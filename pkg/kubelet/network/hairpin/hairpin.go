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

package hairpin

import (
	"fmt"
	"io/ioutil"
	"net"
	"path"
	"regexp"
	"strconv"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/exec"
)

const (
	sysfsNetPath            = "/sys/devices/virtual/net"
	hairpinModeRelativePath = "brport/hairpin_mode"
	hairpinEnable           = "1"
)

var (
	ethtoolOutputRegex = regexp.MustCompile("peer_ifindex: (\\d+)")
)

func SetUpContainer(containerPid int, containerInterfaceName string) error {
	e := exec.New()
	return setUpContainerInternal(e, containerPid, containerInterfaceName)
}

func setUpContainerInternal(e exec.Interface, containerPid int, containerInterfaceName string) error {
	hostIfName, err := findPairInterfaceOfContainerInterface(e, containerPid, containerInterfaceName)
	if err != nil {
		glog.Infof("Unable to find pair interface, setting up all interfaces: %v", err)
		return setUpAllInterfaces()
	}
	return setUpInterface(hostIfName)
}

func findPairInterfaceOfContainerInterface(e exec.Interface, containerPid int, containerInterfaceName string) (string, error) {
	nsenterPath, err := e.LookPath("nsenter")
	if err != nil {
		return "", err
	}
	ethtoolPath, err := e.LookPath("ethtool")
	if err != nil {
		return "", err
	}
	// Get container's interface index
	output, err := e.Command(nsenterPath, "-t", fmt.Sprintf("%d", containerPid), "-n", "-F", "--", ethtoolPath, "--statistics", containerInterfaceName).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("Unable to query interface %s of container %d: %v", containerInterfaceName, containerPid, err)
	}
	// look for peer_ifindex
	match := ethtoolOutputRegex.FindSubmatch(output)
	if match == nil {
		return "", fmt.Errorf("No peer_ifindex in interface statistics for %s of container %d", containerInterfaceName, containerPid)
	}
	peerIfIndex, err := strconv.Atoi(string(match[1]))
	if err != nil { // seems impossible (\d+ not numeric)
		return "", fmt.Errorf("peer_ifindex wasn't numeric: %s: %v", match[1], err)
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
	hairpinModeFile := path.Join(sysfsNetPath, ifName, hairpinModeRelativePath)
	return ioutil.WriteFile(hairpinModeFile, []byte(hairpinEnable), 0644)
}
