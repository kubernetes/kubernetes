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
	sysfsRegex         = regexp.MustCompile("\\s*(\\d+)\\s*")
)

func SetUpContainer(containerPid int, containerInterfaceName string) error {
	hostIfName, err := findPairInterfaceOfContainerInterface(exec.New(), containerPid, containerInterfaceName)
	if err != nil {
		return err
	}
	return setUpInterface(hostIfName)
}

func getPeerIfindex(e exec.Interface, containerPid int, containerInterfaceName string, regex *regexp.Regexp, cmdArgs ...string) (string, error) {
	nsenterPath, err := e.LookPath("nsenter")
	if err != nil {
		return "", err
	}

	// Get container's interface index
	args := append([]string{"-t", fmt.Sprintf("%d", containerPid), "-n", "-F", "--"}, cmdArgs...)
	output, err := e.Command(nsenterPath, args...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("Unable to query interface %s of container %d: %v: %s", containerInterfaceName, containerPid, err, string(output))
	}

	// look for peer_ifindex
	match := regex.FindSubmatch(output)
	if match == nil {
		return "", fmt.Errorf("Failed to match peer ifindex for %s of container %d in '%s'", containerInterfaceName, containerPid, string(output))
	}
	peerIfIndex, err := strconv.Atoi(string(match[1]))
	if err != nil { // seems impossible (\d+ not numeric)
		return "", fmt.Errorf("peer ifindex wasn't numeric: %s: %v", match[1], err)
	}
	iface, err := net.InterfaceByIndex(peerIfIndex)
	if err != nil {
		return "", err
	}
	return iface.Name, nil
}

func findPairInterfaceOfContainerInterfaceSysfs(e exec.Interface, containerPid int, containerInterfaceName string) (string, error) {
	catPath, err := e.LookPath("cat")
	if err != nil {
		return "", err
	}

	sysfsPath := fmt.Sprintf("/sys/class/net/%s/iflink", containerInterfaceName)
	return getPeerIfindex(e, containerPid, containerInterfaceName, sysfsRegex, catPath, sysfsPath)
}

func findPairInterfaceOfContainerInterfaceEthtool(e exec.Interface, containerPid int, containerInterfaceName string) (string, error) {
	ethtoolPath, err := e.LookPath("ethtool")
	if err != nil {
		return "", err
	}

	return getPeerIfindex(e, containerPid, containerInterfaceName, ethtoolOutputRegex, ethtoolPath, "--statistics", containerInterfaceName)
}

func findPairInterfaceOfContainerInterface(e exec.Interface, containerPid int, containerInterfaceName string) (string, error) {
	hostIfName, err := findPairInterfaceOfContainerInterfaceSysfs(e, containerPid, containerInterfaceName)
	if err != nil {
		hostIfName, err = findPairInterfaceOfContainerInterfaceEthtool(e, containerPid, containerInterfaceName)
		if err != nil {
			return "", err
		}
	}
	return hostIfName, nil
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
