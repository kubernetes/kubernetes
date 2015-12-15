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
	"fmt"
	"net"
	"os"
	"regexp"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util"
)

var (
	iptablesMasqCheck = []string{
		"-t", "nat",
		"-C", "POSTROUTING",
		"!", "-d", "10.0.0.0/8",
		"-m", "addrtype", "!", "--dst-type", "LOCAL",
		"-j", "MASQUERADE",
	}
	iptablesMasqAppend = []string{
		"-t", "nat",
		"-A", "POSTROUTING",
		"!", "-d", "10.0.0.0/8",
		"-m", "addrtype", "!", "--dst-type", "LOCAL",
		"-j", "MASQUERADE",
	}

	cidrRegexp = regexp.MustCompile(`inet ([0-9a-fA-F.:]*/[0-9]*)`)
)

// containerBridgeReconciler is used to setup basic node networking.
type containerBridgeReconciler struct {
	// execer will be used to exec commands.
	execer
	// name is the name of the container bridge, eg: cbr0
	name string
	// bridgeExistsFunc will be called to determine if bridge exists.
	bridgeExistsFunc func(string) (bool, error)
}

// netContainerBridgeReconciler creates a containerBridgeReconciler.
func newContainerBridgeReconciler(bridgeName string) *containerBridgeReconciler {
	return &containerBridgeReconciler{
		execer:           execerImpl{},
		name:             bridgeName,
		bridgeExistsFunc: brExists,
	}
}

func (c *containerBridgeReconciler) reconcile(podCIDR string) error {
	if podCIDR == "" {
		glog.V(5).Infof("PodCIDR not set. Will not configure bridge %v.", c.name)
		return nil
	}
	glog.V(5).Infof("PodCIDR is set to %q", podCIDR)
	_, cidr, err := net.ParseCIDR(podCIDR)
	if err != nil {
		return err
	}
	// Set bridge interface address to first address in IPNet
	cidr.IP.To4()[3] += 1
	return c.ensure(cidr)
}

func (c *containerBridgeReconciler) create(wantCIDR *net.IPNet) error {
	// recreate bridge with wantCIDR
	if _, err := c.execCmd("brctl", "addbr", c.name); err != nil {
		glog.Error(err)
		return err
	}
	if _, err := c.execCmd("ip", "addr", "add", wantCIDR.String(), "dev", c.name); err != nil {
		glog.Error(err)
		return err
	}
	if _, err := c.execCmd("ip", "link", "set", "dev", c.name, "mtu", "1460", "up"); err != nil {
		glog.Error(err)
		return err
	}
	// restart docker
	// For now just log the error. The containerRuntime check will catch docker failures.
	// TODO (dawnchen) figure out what we should do for rkt here.
	if util.UsingSystemdInitSystem() {
		if _, err := c.execCmd("systemctl", "restart", "docker"); err != nil {
			glog.Error(err)
		}
	} else {
		if _, err := c.execCmd("service", "docker", "restart"); err != nil {
			glog.Error(err)
		}
	}
	glog.V(2).Infof("Recreated bridge %v and restarted docker", c.name)
	return nil
}

func (c *containerBridgeReconciler) ensure(wantCIDR *net.IPNet) error {
	exists, err := c.bridgeExistsFunc(c.name)
	if err != nil {
		return err
	}
	if !exists {
		glog.V(2).Infof("bridge %v doesn't exist, attempting to create it with range: %s", c.name, wantCIDR)
		return c.create(wantCIDR)
	}
	if !c.checkCIDR(wantCIDR) {
		glog.V(2).Infof("Attempting to recreate %v with address range: %s", c.name, wantCIDR)

		// delete bridge
		if _, err := c.execCmd("ip", "link", "set", "dev", c.name, "down"); err != nil {
			glog.Error(err)
			return err
		}
		if _, err := c.execCmd("brctl", "delbr", c.name); err != nil {
			glog.Error(err)
			return err
		}
		return c.create(wantCIDR)
	}
	return nil
}

func (c *containerBridgeReconciler) checkCIDR(wantCIDR *net.IPNet) bool {
	output, err := c.execCmd("ip", "addr", "show", c.name)
	if err != nil {
		return false
	}
	match := cidrRegexp.FindSubmatch([]byte(output))
	if len(match) < 2 {
		return false
	}
	bridgeIP, bridgeCIDR, err := net.ParseCIDR(string(match[1]))
	if err != nil {
		glog.Errorf("Couldn't parse CIDR: %q", match[1])
		return false
	}
	bridgeCIDR.IP = bridgeIP

	glog.V(5).Infof("Want bridge CIDR: %s, have bridge CIDR: %s", wantCIDR, bridgeCIDR)
	return wantCIDR.IP.Equal(bridgeIP) && bytes.Equal(wantCIDR.Mask, bridgeCIDR.Mask)
}

// TODO(dawnchen): Using pkg/util/iptables
func ensureIPTablesMasqRule(e execer) error {
	// Check if the MASQUERADE rule exist or not
	if _, err := e.execCmd("iptables", iptablesMasqCheck...); err == nil {
		// The MASQUERADE rule exists
		return nil
	}
	glog.Infof("MASQUERADE rule doesn't exist, recreate it")
	if _, err := e.execCmd("iptables", iptablesMasqAppend...); err != nil {
		return err
	}
	return nil
}

// Check if bridge network interface is configured or not, and take action
// when the configuration is missing on the node, and propagate the rest
// error to kubelet to handle.
func brExists(bridgeName string) (bool, error) {
	if _, err := os.Stat(fmt.Sprintf("/sys/class/net/%v", bridgeName)); err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}
