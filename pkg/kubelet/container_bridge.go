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
	"net"
	"os"
	"os/exec"
	"syscall"

	"github.com/golang/glog"
	"github.com/vishvananda/netlink"

	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/iptables"
	"k8s.io/kubernetes/pkg/util/procfs"
)

// ensureBridge and ensureIPTablesMasqRule are the only methods that are called from outside this file
func ensureBridge(brName string, brMtu int, wantCIDR *net.IPNet, promiscuous, babysitDaemons bool) error {
	br := &netlink.Bridge{
		LinkAttrs: netlink.LinkAttrs{
			Name: brName,
			MTU:  brMtu,
			// Let kernel use default txqueuelen; leaving it unset
			// means 0, and a zero-length TX queue messes up FIFO
			// traffic shapers which use TX queue length as the
			// default packet limit
			TxQLen: -1,
		},
	}

	exists, err := bridgeExists(br.Name)
	if err != nil {
		return err
	}
	if !exists {
		glog.V(2).Infof("%s doesn't exist, attempting to create it with range: %s", br.Name, wantCIDR)
		return createBridge(br, wantCIDR, babysitDaemons)
	}
	if !bridgeCidrCorrect(br, wantCIDR) {
		glog.V(2).Infof("Attempting to recreate %s with address range: %s", br.Name, wantCIDR)

		if err := netlink.LinkSetDown(br); err != nil {
			return fmt.Errorf("could not set down bridge %s: %v", br.Name, err)
		}

		if err := netlink.LinkDel(br); err != nil {
			return fmt.Errorf("could not delete bridge %s: %v", br.Name, err)
		}

		return createBridge(br, wantCIDR, babysitDaemons)
	}
	// Put the container bridge into promiscuous mode to force it to accept hairpin packets.
	// TODO: Remove this once the kernel bug (#20096) is fixed.
	if promiscuous {
		// Checking if the bridge is in promiscuous mode is as expensive and more brittle than
		// simply setting the flag every time.
		// TODO: check and set promiscuous mode with netlink once vishvananda/netlink supports it
		if err := exec.Command("ip", "link", "set", br.Name, "promisc", "on").Run(); err != nil {
			glog.Error(err)
			return err
		}
	}
	return nil
}

func createBridge(br *netlink.Bridge, wantCIDR *net.IPNet, babysitDaemons bool) error {
	addr := &netlink.Addr{
		IPNet: wantCIDR,
		Label: "",
	}

	if err := netlink.LinkAdd(br); err != nil {
		return fmt.Errorf("could not add create bridge %s: %v", br.Name, err)
	}

	if err := netlink.AddrAdd(br, addr); err != nil {
		return fmt.Errorf("could not add IP address to bridge %s: %v", br.Name, err)
	}

	if err := netlink.LinkSetUp(br); err != nil {
		return fmt.Errorf("could not set up bridge %s: %v", br.Name, err)
	}

	// Stop docker so that babysitter process can restart it again with proper configurations and
	// checkpoint file (https://github.com/docker/docker/issues/18283). It is safe to kill docker
	// process here since CIDR can be changed only once for a given node object, and node is marked
	// as NotReady until the docker daemon is restarted with the newly configured custom bridge.
	// TODO (dawnchen): Remove this once corrupted checkpoint issue is fixed.
	//
	// For now just log the error. The containerRuntime check will catch docker failures.
	// TODO (dawnchen) figure out what we should do for rkt here.
	if babysitDaemons {
		if err := procfs.PKill("docker", syscall.SIGKILL); err != nil {
			glog.Error(err)
		}
	} else if util.UsingSystemdInitSystem() {
		if err := exec.Command("systemctl", "restart", "docker").Run(); err != nil {
			glog.Error(err)
		}
	} else {
		if err := exec.Command("service", "docker", "restart").Run(); err != nil {
			glog.Error(err)
		}
	}
	glog.V(2).Info("Recreated %s and restarted docker", br.Name)
	return nil
}

// Check if the bridge network interface is configured or not, and take action
// when the configuration is missing on the node, and propagate the rest
// error to kubelet to handle.
func bridgeExists(brName string) (bool, error) {
	if _, err := os.Stat("/sys/class/net/" + brName); err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// Check if some of the IP addresses on this bridge matches the CIDR we want
func bridgeCidrCorrect(br *netlink.Bridge, wantCIDR *net.IPNet) bool {
	addrs, err := netlink.AddrList(br, syscall.AF_INET)
	if err != nil {
		glog.Errorf("could not get list of IP addresses from bridge %s: %v", br.Name, err)
		return false
	}

	for _, addr := range addrs {
		if addr.IPNet.String() == wantCIDR.String() {
			return true
		}
	}

	return false
}

// nonMasqueradeCIDR is the CIDR for our internal IP range; traffic to IPs
// outside this range will use IP masquerade.
func ensureIPTablesMasqRule(client iptables.Interface, nonMasqueradeCIDR string) error {
	if _, err := client.EnsureRule(iptables.Append, iptables.TableNAT,
		iptables.ChainPostrouting,
		"-m", "comment", "--comment", "kubelet: SNAT outbound cluster traffic",
		"-m", "addrtype", "!", "--dst-type", "LOCAL",
		"!", "-d", nonMasqueradeCIDR,
		"-j", "MASQUERADE"); err != nil {
		return fmt.Errorf("Failed to ensure masquerading for %s chain %s: %v",
			iptables.TableNAT, iptables.ChainPostrouting, err)
	}
	return nil
}
