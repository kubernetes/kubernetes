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
	"os/exec"
	"regexp"

	"github.com/golang/glog"
)

var cidrRegexp = regexp.MustCompile(`inet ([0-9a-fA-F.:]*/[0-9]*)`)

func ensureCbr0(wantCIDR *net.IPNet) error {
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
		// recreate cbr0 with wantCIDR
		if err := exec.Command("brctl", "addbr", "cbr0").Run(); err != nil {
			glog.Error(err)
			return err
		}
		if err := exec.Command("ip", "addr", "add", wantCIDR.String(), "dev", "cbr0").Run(); err != nil {
			glog.Error(err)
			return err
		}
		if err := exec.Command("ip", "link", "set", "dev", "cbr0", "up").Run(); err != nil {
			glog.Error(err)
			return err
		}
		// restart docker
		if err := exec.Command("service", "docker", "restart").Run(); err != nil {
			glog.Error(err)
			// For now just log the error. The containerRuntime check will catch docker failures.
			// TODO (dawnchen) figure out what we should do for rkt here.
		}
		glog.V(2).Info("Recreated cbr0 and restarted docker")
	}
	return nil
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
