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
	"bytes"
	"fmt"
	"net"
	"os"
	"os/exec"
	"regexp"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/iptables"
	"k8s.io/kubernetes/pkg/util/procfs"
	"syscall"
)

var cidrRegexp = regexp.MustCompile(`inet ([0-9a-fA-F.:]*/[0-9]*)`)

func createCBR0(wantCIDR *net.IPNet, babysitDaemons bool) error {
	// recreate cbr0 with wantCIDR
	if err := exec.Command("brctl", "addbr", "cbr0").Run(); err != nil {
		glog.Error(err)
		return err
	}
	if err := exec.Command("ip", "addr", "add", wantCIDR.String(), "dev", "cbr0").Run(); err != nil {
		glog.Error(err)
		return err
	}
	if err := exec.Command("ip", "link", "set", "dev", "cbr0", "mtu", "1460", "up").Run(); err != nil {
		glog.Error(err)
		return err
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
	glog.V(2).Info("Recreated cbr0 and restarted docker")
	return nil
}

func ensureCbr0(wantCIDR *net.IPNet, promiscuous, babysitDaemons bool) error {
	exists, err := cbr0Exists()
	if err != nil {
		return err
	}
	if !exists {
		glog.V(2).Infof("CBR0 doesn't exist, attempting to create it with range: %s", wantCIDR)
		return createCBR0(wantCIDR, babysitDaemons)
	}
	if !cbr0CidrCorrect(wantCIDR) {
		glog.V(2).Infof("Attempting to recreate cbr0 with address range: %s", wantCIDR)

		// delete cbr0
		if err := exec.Command("ip", "link", "set", "dev", "cbr0", "down").Run(); err != nil {
			glog.Error(err)
			return err
		}
		if err := exec.Command("brctl", "delbr", "cbr0").Run(); err != nil {
			glog.Error(err)
			return err
		}
		if err := createCBR0(wantCIDR, babysitDaemons); err != nil {
			glog.Error(err)
			return err
		}
	}
	// Put the container bridge into promiscuous mode to force it to accept hairpin packets.
	// TODO: Remove this once the kernel bug (#20096) is fixed.
	if promiscuous {
		// Checking if the bridge is in promiscuous mode is as expensive and more brittle than
		// simply setting the flag every time.
		if err := exec.Command("ip", "link", "set", "cbr0", "promisc", "on").Run(); err != nil {
			glog.Error(err)
			return err
		}
	}
	return nil
}

// Check if cbr0 network interface is configured or not, and take action
// when the configuration is missing on the node, and propagate the rest
// error to kubelet to handle.
func cbr0Exists() (bool, error) {
	if _, err := os.Stat("/sys/class/net/cbr0"); err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

func cbr0CidrCorrect(wantCIDR *net.IPNet) bool {
	output, err := exec.Command("ip", "addr", "show", "cbr0").Output()
	if err != nil {
		return false
	}
	match := cidrRegexp.FindSubmatch(output)
	if len(match) < 2 {
		return false
	}
	cbr0IP, cbr0CIDR, err := net.ParseCIDR(string(match[1]))
	if err != nil {
		glog.Errorf("Couldn't parse CIDR: %q", match[1])
		return false
	}
	cbr0CIDR.IP = cbr0IP

	glog.V(5).Infof("Want cbr0 CIDR: %s, have cbr0 CIDR: %s", wantCIDR, cbr0CIDR)
	return wantCIDR.IP.Equal(cbr0IP) && bytes.Equal(wantCIDR.Mask, cbr0CIDR.Mask)
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
