/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kubelet

import (
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"strings"

	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"

	"github.com/golang/glog"
)

// TODO: Move all this to a network plugin.
const (
	// TODO: The location of default docker options is distro specific, so this
	// probably won't work on anything other than debian/ubuntu. This is a
	// short-term compromise till we've moved overlay setup into a plugin.
	dockerOptsFile    = "/etc/default/docker"
	flannelSubnetKey  = "FLANNEL_SUBNET"
	flannelNetworkKey = "FLANNEL_NETWORK"
	flannelMtuKey     = "FLANNEL_MTU"
	dockerOptsKey     = "DOCKER_OPTS"
	flannelSubnetFile = "/var/run/flannel/subnet.env"
)

// A Kubelet to flannel bridging helper.
type FlannelHelper struct {
	subnetFile     string
	iptablesHelper utiliptables.Interface
}

// NewFlannelHelper creates a new flannel helper.
func NewFlannelHelper() *FlannelHelper {
	return &FlannelHelper{
		subnetFile:     flannelSubnetFile,
		iptablesHelper: utiliptables.New(utilexec.New(), utildbus.New(), utiliptables.ProtocolIpv4),
	}
}

// Ensure the required MASQUERADE rules exist for the given network/cidr.
func (f *FlannelHelper) ensureFlannelMasqRule(kubeNetwork, podCIDR string) error {
	// TODO: Investigate delegation to flannel via -ip-masq=true once flannel
	// issue #374 is resolved.
	comment := "Flannel masquerade facilitates pod<->node traffic."
	args := []string{
		"-m", "comment", "--comment", comment,
		"!", "-d", kubeNetwork, "-s", podCIDR, "-j", "MASQUERADE",
	}
	_, err := f.iptablesHelper.EnsureRule(
		utiliptables.Append,
		utiliptables.TableNAT,
		utiliptables.ChainPostrouting,
		args...)
	return err
}

// Handshake waits for the flannel subnet file and installs a few IPTables
// rules, returning the pod CIDR allocated for this node.
func (f *FlannelHelper) Handshake() (podCIDR string, err error) {
	// TODO: Using a file to communicate is brittle
	if _, err = os.Stat(f.subnetFile); err != nil {
		return "", fmt.Errorf("Waiting for subnet file %v", f.subnetFile)
	}
	glog.Infof("Found flannel subnet file %v", f.subnetFile)

	config, err := parseKVConfig(f.subnetFile)
	if err != nil {
		return "", err
	}
	if err = writeDockerOptsFromFlannelConfig(config); err != nil {
		return "", err
	}
	podCIDR, ok := config[flannelSubnetKey]
	if !ok {
		return "", fmt.Errorf("No flannel subnet, config %+v", config)
	}
	kubeNetwork, ok := config[flannelNetworkKey]
	if !ok {
		return "", fmt.Errorf("No flannel network, config %+v", config)
	}
	if f.ensureFlannelMasqRule(kubeNetwork, podCIDR); err != nil {
		return "", fmt.Errorf("Unable to install flannel masquerade %v", err)
	}
	return podCIDR, nil
}

// Take env variables from flannel subnet env and write to /etc/docker/defaults.
func writeDockerOptsFromFlannelConfig(flannelConfig map[string]string) error {
	// TODO: Write dockeropts to unit file on systemd machines
	// https://github.com/docker/docker/issues/9889
	mtu, ok := flannelConfig[flannelMtuKey]
	if !ok {
		return fmt.Errorf("No flannel mtu, flannel config %+v", flannelConfig)
	}
	dockerOpts, err := parseKVConfig(dockerOptsFile)
	if err != nil {
		return err
	}
	opts, ok := dockerOpts[dockerOptsKey]
	if !ok {
		glog.Errorf("Did not find docker opts, writing them")
		opts = fmt.Sprintf(
			" --bridge=cbr0 --iptables=false --ip-masq=false")
	} else {
		opts, _ = strconv.Unquote(opts)
	}
	dockerOpts[dockerOptsKey] = fmt.Sprintf("\"%v --mtu=%v\"", opts, mtu)
	if err = writeKVConfig(dockerOptsFile, dockerOpts); err != nil {
		return err
	}
	return nil
}

// parseKVConfig takes a file with key-value env variables and returns a dictionary mapping the same.
func parseKVConfig(filename string) (map[string]string, error) {
	config := map[string]string{}
	if _, err := os.Stat(filename); err != nil {
		return config, err
	}
	buff, err := ioutil.ReadFile(filename)
	if err != nil {
		return config, err
	}
	str := string(buff)
	glog.Infof("Read kv options %+v from %v", str, filename)
	for _, line := range strings.Split(str, "\n") {
		kv := strings.Split(line, "=")
		if len(kv) != 2 {
			glog.Warningf("Ignoring non key-value pair %v", kv)
			continue
		}
		config[string(kv[0])] = string(kv[1])
	}
	return config, nil
}

// writeKVConfig writes a kv map as env variables into the given file.
func writeKVConfig(filename string, kv map[string]string) error {
	if _, err := os.Stat(filename); err != nil {
		return err
	}
	content := ""
	for k, v := range kv {
		content += fmt.Sprintf("%v=%v\n", k, v)
	}
	glog.Warningf("Writing kv options %+v to %v", content, filename)
	return ioutil.WriteFile(filename, []byte(content), 0644)
}
