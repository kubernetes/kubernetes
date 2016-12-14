/*
Copyright 2016 The Kubernetes Authors.

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

package remote

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/golang/glog"
)

// utils.go contains functions used accross test suites.

const (
	cniRelease   = "07a8a28637e97b22eb8dfe710eeae1344f69d16e"
	cniDirectory = "cni"
	cniURL       = "https://storage.googleapis.com/kubernetes-release/network-plugins/cni-" + cniRelease + ".tar.gz"
)

// Install the cni plugin.
func installCNI(host, workspace string) error {
	glog.V(2).Infof("Install CNI on %q", host)
	cniPath := filepath.Join(workspace, cniDirectory)
	cmd := getSSHCommand(" ; ",
		fmt.Sprintf("mkdir -p %s", cniPath),
		fmt.Sprintf("wget -O - %s | tar -xz -C %s", cniURL, cniPath),
	)
	if output, err := SSH(host, "sh", "-c", cmd); err != nil {
		return fmt.Errorf("failed to install cni plugin on %q: %v output: %q", host, err, output)
	}
	return nil
}

// configureFirewall configures iptable firewall rules.
func configureFirewall(host string) error {
	glog.V(2).Infof("Configure iptables firewall rules on %q", host)
	// TODO: consider calling bootstrap script to configure host based on OS
	output, err := SSH(host, "iptables", "-L", "INPUT")
	if err != nil {
		return fmt.Errorf("failed to get iptables INPUT on %q: %v output: %q", host, err, output)
	}
	if strings.Contains(output, "Chain INPUT (policy DROP)") {
		cmd := getSSHCommand("&&",
			"(iptables -C INPUT -w -p TCP -j ACCEPT || iptables -A INPUT -w -p TCP -j ACCEPT)",
			"(iptables -C INPUT -w -p UDP -j ACCEPT || iptables -A INPUT -w -p UDP -j ACCEPT)",
			"(iptables -C INPUT -w -p ICMP -j ACCEPT || iptables -A INPUT -w -p ICMP -j ACCEPT)")
		output, err := SSH(host, "sh", "-c", cmd)
		if err != nil {
			return fmt.Errorf("failed to configured firewall on %q: %v output: %v", host, err, output)
		}
	}
	output, err = SSH(host, "iptables", "-L", "FORWARD")
	if err != nil {
		return fmt.Errorf("failed to get iptables FORWARD on %q: %v output: %q", host, err, output)
	}
	if strings.Contains(output, "Chain FORWARD (policy DROP)") {
		cmd := getSSHCommand("&&",
			"(iptables -C FORWARD -w -p TCP -j ACCEPT || iptables -A FORWARD -w -p TCP -j ACCEPT)",
			"(iptables -C FORWARD -w -p UDP -j ACCEPT || iptables -A FORWARD -w -p UDP -j ACCEPT)",
			"(iptables -C FORWARD -w -p ICMP -j ACCEPT || iptables -A FORWARD -w -p ICMP -j ACCEPT)")
		output, err = SSH(host, "sh", "-c", cmd)
		if err != nil {
			return fmt.Errorf("failed to configured firewall on %q: %v output: %v", host, err, output)
		}
	}
	return nil
}

// cleanupNodeProcesses kills all running node processes may conflict with the test.
func cleanupNodeProcesses(host string) {
	glog.V(2).Infof("Killing any existing node processes on %q", host)
	cmd := getSSHCommand(" ; ",
		"pkill kubelet",
		"pkill kube-apiserver",
		"pkill etcd",
		"pkill e2e_node.test",
	)
	// No need to log an error if pkill fails since pkill will fail if the commands are not running.
	// If we are unable to stop existing running k8s processes, we should see messages in the kubelet/apiserver/etcd
	// logs about failing to bind the required ports.
	SSH(host, "sh", "-c", cmd)
}
