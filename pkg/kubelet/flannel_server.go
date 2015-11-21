package kubelet

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/golang/glog"
)

const (
	networkType       = "vxlan"
	dockerOptsFile    = "/etc/default/docker"
	flannelSubnetKey  = "FLANNEL_SUBNET"
	flannelNetworkKey = "FLANNEL_NETWORK"
	flannelMtuKey     = "FLANNEL_MTU"
	dockerOptsKey     = "DOCKER_OPTS"
	flannelSubnetFile = "/var/run/flannel/subnet.env"
)

type FlannelServer struct {
	subnetFile string
	// TODO: Manage subnet file.
}

func NewFlannelServer() *FlannelServer {
	return &FlannelServer{flannelSubnetFile}
}

func (f *FlannelServer) Handshake() (podCIDR string, err error) {
	// Flannel daemon will hang till the server comes up, kubelet will hang until
	// flannel daemon has written subnet env variables. This is the kubelet handshake.
	// To improve performance, we could defer just the configuration of the container
	// bridge till after subnet.env is written. Keeping it local is clearer for now.
	// TODO: Using a file to communicate is brittle
	if _, err = os.Stat(f.subnetFile); err != nil {
		return "", fmt.Errorf("Waiting for subnet file %v", f.subnetFile)
	}
	glog.Infof("(kubelet)Found flannel subnet file %v", f.subnetFile)

	// TODO: Rest of this function is a hack.
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
	if err := exec.Command("iptables",
		"-t", "nat",
		"-A", "POSTROUTING",
		"!", "-d", kubeNetwork,
		"-s", podCIDR,
		"-j", "MASQUERADE").Run(); err != nil {
		return "", fmt.Errorf("Unable to install iptables rule for flannel.")
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
		glog.Errorf("(kubelet)Did not find docker opts, writing them")
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
	glog.Infof("(kubelet) Read kv options %+v from %v", str, filename)
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
	glog.Warningf("(kubelet)Writing kv options %+v to %v", content, filename)
	return ioutil.WriteFile(filename, []byte(content), 0644)
}
