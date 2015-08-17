/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"net"
	"os"
	"os/exec"
	"regexp"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util"
)

var cidrRegexp = regexp.MustCompile(`inet ([0-9a-fA-F.:]*/[0-9]*)`)

func createCBR0(wantCIDR *net.IPNet) error {
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
	// restart docker
	// For now just log the error. The containerRuntime check will catch docker failures.
	// TODO (dawnchen) figure out what we should do for rkt here.
	if util.UsingSystemdInitSystem() {
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

func ensureCbr0(wantCIDR *net.IPNet) error {
	exists, err := cbr0Exists()
	if err != nil {
		return err
	}
	if !exists {
		glog.V(2).Infof("CBR0 doesn't exist, attempting to create it with range: %s", wantCIDR)
		return createCBR0(wantCIDR)
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
		return createCBR0(wantCIDR)
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

// TODO(dawnchen): Using pkg/util/iptables
func ensureIPTablesMasqRule() error {
	// Check if the MASQUERADE rule exist or not
	if err := exec.Command("iptables", "-t", "nat", "-C", "POSTROUTING", "-o", "eth0", "-j", "MASQUERADE", "!", "-d", "10.0.0.0/8").Run(); err == nil {
		// The MASQUERADE rule exists
		return nil
	}

	glog.Infof("MASQUERADE rule doesn't exist, recreate it")
	if err := exec.Command("iptables", "-t", "nat", "-A", "POSTROUTING", "-o", "eth0", "-j", "MASQUERADE", "!", "-d", "10.0.0.0/8").Run(); err != nil {
		return err
	}
	return nil
}
