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
	"regexp"
	"strconv"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/exec"
)

const (
	sysfsNetPath            = "/sys/devices/virtual/net"
	brportRelativePath      = "brport"
	hairpinModeRelativePath = "hairpin_mode"
	hairpinEnable           = "1"
)

var (
	ethtoolOutputRegex = regexp.MustCompile("peer_ifindex: (\\d+)")
)

func SetUpContainerPid(containerPid int, containerInterfaceName string) error {
	pidStr := fmt.Sprintf("%d", containerPid)
	nsenterArgs := []string{"-t", pidStr, "-n"}
	return setUpContainerInternal(containerInterfaceName, pidStr, nsenterArgs)
}

func SetUpContainerPath(netnsPath string, containerInterfaceName string) error {
	if netnsPath[0] != '/' {
		return fmt.Errorf("netnsPath path '%s' was invalid", netnsPath)
	}
	nsenterArgs := []string{"-n", netnsPath}
	return setUpContainerInternal(containerInterfaceName, netnsPath, nsenterArgs)
}

func setUpContainerInternal(containerInterfaceName, containerDesc string, nsenterArgs []string) error {
	e := exec.New()
	hostIfName, err := findPairInterfaceOfContainerInterface(e, containerInterfaceName, containerDesc, nsenterArgs)
	if err != nil {
		glog.Infof("Unable to find pair interface, setting up all interfaces: %v", err)
		return setUpAllInterfaces()
	}
	return setUpInterface(hostIfName)
}

func findPairInterfaceOfContainerInterface(e exec.Interface, containerInterfaceName, containerDesc string, nsenterArgs []string) (string, error) {
	nsenterPath, err := e.LookPath("nsenter")
	if err != nil {
		return "", err
	}
	ethtoolPath, err := e.LookPath("ethtool")
	if err != nil {
		return "", err
	}

	nsenterArgs = append(nsenterArgs, "-F", "--", ethtoolPath, "--statistics", containerInterfaceName)
	output, err := e.Command(nsenterPath, nsenterArgs...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("Unable to query interface %s of container %s: %v: %s", containerInterfaceName, containerDesc, err, string(output))
	}
	// look for peer_ifindex
	match := ethtoolOutputRegex.FindSubmatch(output)
	if match == nil {
		return "", fmt.Errorf("No peer_ifindex in interface statistics for %s of container %s", containerInterfaceName, containerDesc)
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
