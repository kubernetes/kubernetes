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

package hairpin

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path"

	"github.com/containernetworking/cni/pkg/ns"
	"github.com/golang/glog"
	"github.com/vishvananda/netlink"
)

const (
	sysfsNetPath            = "/sys/devices/virtual/net"
	brportRelativePath      = "brport"
	hairpinModeRelativePath = "hairpin_mode"
	hairpinEnable           = "1"
)

func SetUpContainerPid(containerPid int, containerInterfaceName string) error {
	netnsPath := fmt.Sprintf("/proc/%d/ns/net", containerPid)
	return SetUpContainerPath(netnsPath, containerInterfaceName)
}

func SetUpContainerPath(netnsPath string, containerInterfaceName string) error {
	hostIfName, err := findPairInterfaceOfContainerInterface(containerInterfaceName, netnsPath)
	if err != nil {
		glog.Infof("Unable to find pair interface, setting up all interfaces: %v", err)
		return setUpAllInterfaces()
	}
	return setUpInterface(hostIfName)
}

func findPairInterfaceOfContainerInterface(containerInterfaceName, netnsPath string) (string, error) {
	var peerIfIndex int

	err := ns.WithNetNSPath(netnsPath, func(hostNs ns.NetNS) error {
		if containerInterface, err := netlink.LinkByName(containerInterfaceName); err != nil {
			return err
		} else {
			peerIfIndex = containerInterface.Attrs().ParentIndex
		}

		return nil
	})
	if err != nil {
		return "", fmt.Errorf("Unable to query interface %s of container %s: %v", containerInterfaceName, netnsPath, err)
	}

	iface, err := net.InterfaceByIndex(peerIfIndex)
	if err != nil {
		return "", err
	}
	return iface.Name, nil
}

func setUpAllInterfaces() error {
	interfaces, err := net.Interfaces()
	if err != nil {
		return err
	}
	for _, netIf := range interfaces {
		setUpInterface(netIf.Name) // ignore errors
	}
	return nil
}

func setUpInterface(ifName string) error {
	glog.V(3).Infof("Enabling hairpin on interface %s", ifName)
	ifPath := path.Join(sysfsNetPath, ifName)
	if _, err := os.Stat(ifPath); err != nil {
		return err
	}
	brportPath := path.Join(ifPath, brportRelativePath)
	if _, err := os.Stat(brportPath); err != nil && os.IsNotExist(err) {
		// Device is not on a bridge, so doesn't need hairpin mode
		return nil
	}
	hairpinModeFile := path.Join(brportPath, hairpinModeRelativePath)
	return ioutil.WriteFile(hairpinModeFile, []byte(hairpinEnable), 0644)
}
