// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package testutils

import (
	"bufio"
	"fmt"
	"net"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/vishvananda/netlink"
)

func getDefaultGW(family int) (string, error) {
	routes, err := netlink.RouteList(nil, family)
	if err != nil {
		return "", err
	}

	for _, route := range routes {
		if route.Src == nil && route.Dst == nil {
			return route.Gw.String(), nil
		}
	}

	return "", fmt.Errorf("Default route is not set")
}
func GetDefaultGWv4() (string, error) {
	return getDefaultGW(netlink.FAMILY_V4)
}

func GetDefaultGWv6() (string, error) {
	return getDefaultGW(netlink.FAMILY_V6)
}

func GetIPs(ifaceWanted string, familyWanted int) ([]string, error) {
	ips := make([]string, 0)
	ifaces, err := net.Interfaces()
	if err != nil {
		return ips, err
	}
	for _, iface := range ifaces {
		if iface.Name != ifaceWanted {
			continue
		}

		addrs, _ := iface.Addrs()
		for _, addr := range addrs {
			addrString := addr.String()
			ip, _, err := net.ParseCIDR(addrString)
			if err != nil {
				return ips, err
			}

			if strings.Contains(addrString, ".") && familyWanted == netlink.FAMILY_V4 ||
				strings.Contains(addrString, ":") && familyWanted == netlink.FAMILY_V6 {
				ips = append(ips, ip.String())
			}
		}
	}
	return ips, err
}

func GetIPsv4(iface string) ([]string, error) {
	return GetIPs(iface, netlink.FAMILY_V4)
}
func GetIPsv6(iface string) ([]string, error) {
	return GetIPs(iface, netlink.FAMILY_V6)
}

func GetGW(iface string, family int) (string, error) {
	return "", fmt.Errorf("Not implemented")
}
func GetGWv4(iface string) (string, error) {
	return GetGW(iface, netlink.FAMILY_V4)
}

func GetGWv6(iface string) (string, error) {
	return GetGW(iface, netlink.FAMILY_V4)
}

func GetNonLoIfaceWithAddrs(ipFamily int) (iface net.Interface, addrs []string, err error) {
	ifaces, err := net.Interfaces()
	if err != nil {
		return iface, nil, err
	}

	for _, i := range ifaces {
		if i.Flags&net.FlagLoopback == 0 {
			addrs, err = GetIPs(i.Name, ipFamily)
			if err != nil {
				return iface, addrs, fmt.Errorf("Cannot get IP address for interface %v: %v", i.Name, err)
			}

			if len(addrs) == 0 {
				continue
			}
			iface = i

			ifaceNameLower := strings.ToLower(i.Name)
			// Don't use rkt's interfaces
			if strings.Contains(ifaceNameLower, "cni") ||
				strings.Contains(ifaceNameLower, "veth") {
				continue
			}
			break
		}
	}
	return iface, addrs, err
}

func GetNonLoIfaceIPv4() (string, error) {
	iface, ifaceIPsv4, err := GetNonLoIfaceWithAddrs(netlink.FAMILY_V4)
	if err != nil {
		return "", fmt.Errorf("Error while getting non-lo host interface: %v\n", err)
	}

	if iface.Name == "" || ifaceIPsv4 == nil {
		return "", nil
	}

	return ifaceIPsv4[0], nil
}

func CheckTcp4Port(port int) (bool, error) {
	tcpFile, err := os.Open("/proc/net/tcp")
	if err != nil {
		return false, err
	}
	defer tcpFile.Close()

	re := regexp.MustCompile(`:([A-Z0-9]+) `)
	scanner := bufio.NewScanner(tcpFile)
	for scanner.Scan() {
		line := scanner.Text()
		result := re.FindAllStringSubmatch(line, -1)
		if result != nil {
			i, err := strconv.ParseInt(result[0][1], 16, 32)
			if err != nil {
				return false, err
			}
			if int(i) == port {
				return false, nil
			}
		}
	}
	return true, nil
}

func GetNextFreePort4() (int, error) {
	return GetNextFreePort4Banned(map[int]struct{}{})
}

func GetNextFreePort4Banned(bannedPorts map[int]struct{}) (int, error) {
	for port := 49152; port <= 65535; port++ {
		if _, portBanned := bannedPorts[port]; portBanned {
			continue
		}

		avail, err := CheckTcp4Port(port)
		if err != nil {
			return 0, err
		}
		if avail {
			return port, nil
		}
	}
	return 0, fmt.Errorf("No available ports")
}

func GetIfaceCount() (int, error) {
	ifaces, err := net.Interfaces()
	if err != nil {
		return 0, err
	}
	return len(ifaces), nil
}
