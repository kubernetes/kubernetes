/*
Copyright 2016 The Kubernetes Authors.

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

package net

import (
	"bufio"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"os"

	"strings"

	"github.com/golang/glog"
)

type AddressFamily uint

const (
	familyIPv4 AddressFamily = 4
	familyIPv6 AddressFamily = 6
)

type Route struct {
	Interface   string
	Destination net.IP
	Gateway     net.IP
	// TODO: add more fields here if needed
}

// getRoutes obtains the IPv4 routes, and filters out non-default routes.
func getRoutes(input io.Reader) ([]Route, error) {
	routes := []Route{}
	if input == nil {
		return nil, fmt.Errorf("input is nil")
	}
	scanner := bufio.NewReader(input)
	for {
		line, err := scanner.ReadString('\n')
		if err == io.EOF {
			break
		}
		//ignore the headers in the route info
		if strings.HasPrefix(line, "Iface") {
			continue
		}
		fields := strings.Fields(line)
		dest, err := parseIP(fields[1])
		if err != nil {
			return nil, err
		}
		gw, err := parseIP(fields[2])
		if err != nil {
			return nil, err
		}
		if !dest.Equal(net.IPv4zero) {
			continue
		}
		routes = append(routes, Route{})
		route := &routes[len(routes)-1]
		route.Interface = fields[0]
		route.Destination = dest
		route.Gateway = gw
	}
	return routes, nil
}

func parseIP(str string) (net.IP, error) {
	if str == "" {
		return nil, fmt.Errorf("input is nil")
	}
	bytes, err := hex.DecodeString(str)
	if err != nil {
		return nil, err
	}
	if len(bytes) != net.IPv4len {
		return nil, fmt.Errorf("invalid IPv4 address in route")
	}
	bytes[0], bytes[1], bytes[2], bytes[3] = bytes[3], bytes[2], bytes[1], bytes[0]
	return net.IP(bytes), nil
}

func isInterfaceUp(intf *net.Interface) bool {
	if intf == nil {
		return false
	}
	if intf.Flags&net.FlagUp != 0 {
		glog.V(4).Infof("Interface %v is up", intf.Name)
		return true
	}
	return false
}

func inFamily(ip net.IP, expectedFamily AddressFamily) bool {
	ipFamily := familyIPv4
	if ip.To4() == nil {
		ipFamily = familyIPv6
	}
	return ipFamily == expectedFamily
}

// getFinalIP method checks all the IP addresses of a Interface looking
// for a valid non-loopback/link-local address of the requested family.
func getFinalIP(addrs []net.Addr, family AddressFamily) (net.IP, error) {
	if len(addrs) > 0 {
		for i := range addrs {
			glog.V(4).Infof("Checking addr  %s.", addrs[i].String())
			ip, _, err := net.ParseCIDR(addrs[i].String())
			if err != nil {
				return nil, err
			}
			if inFamily(ip, family) {
				if ip.IsGlobalUnicast() {
					glog.V(4).Infof("IP found %v", ip)
					return ip, nil
				} else {
					glog.V(4).Infof("non-global IP found %v", ip)
				}
			} else {
				glog.V(4).Infof("%v is not an IPv%d address", ip, int(family))
			}

		}
	}
	return nil, nil
}

func getIPFromInterface(intfName string, forFamily AddressFamily, nw networkInterfacer) (net.IP, error) {
	intf, err := nw.InterfaceByName(intfName)
	if err != nil {
		return nil, err
	}
	if isInterfaceUp(intf) {
		addrs, err := nw.Addrs(intf)
		if err != nil {
			return nil, err
		}
		glog.V(4).Infof("Interface %q has %d addresses :%v.", intfName, len(addrs), addrs)
		finalIP, err := getFinalIP(addrs, forFamily)
		if err != nil {
			return nil, err
		}
		if finalIP != nil {
			glog.V(4).Infof("valid IPv%d address for interface %q found as %v.", int(forFamily), intfName, finalIP)
			return finalIP, nil
		}
	}

	return nil, nil
}

func flagsSet(flags net.Flags, test net.Flags) bool {
	return flags&test != 0
}

func flagsClear(flags net.Flags, test net.Flags) bool {
	return flags&test == 0
}

func chooseHostInterfaceNativeGo() (net.IP, error) {
	intfs, err := net.Interfaces()
	if err != nil {
		return nil, err
	}
	i := 0
	var ip net.IP
	for i = range intfs {
		if flagsSet(intfs[i].Flags, net.FlagUp) && flagsClear(intfs[i].Flags, net.FlagLoopback|net.FlagPointToPoint) {
			addrs, err := intfs[i].Addrs()
			if err != nil {
				return nil, err
			}
			if len(addrs) > 0 {
				for _, addr := range addrs {
					if addrIP, _, err := net.ParseCIDR(addr.String()); err == nil {
						if addrIP.To4() != nil {
							ip = addrIP.To4()
							if !ip.IsLinkLocalMulticast() && !ip.IsLinkLocalUnicast() {
								break
							}
						}
					}
				}
				if ip != nil {
					// This interface should suffice.
					break
				}
			}
		}
	}
	if ip == nil {
		return nil, fmt.Errorf("no acceptable interface from host")
	}
	glog.V(4).Infof("Choosing interface %s (IP %v) as default", intfs[i].Name, ip)
	return ip, nil
}

//ChooseHostInterface is a method used fetch an IP for a daemon.
//It uses data from /proc/net/route file.
//For a node with no internet connection ,it returns error
//For a multi n/w interface node it returns the IP of the interface with gateway on it.
func ChooseHostInterface() (net.IP, error) {
	inFile, err := os.Open("/proc/net/route")
	if err != nil {
		if os.IsNotExist(err) {
			return chooseHostInterfaceNativeGo()
		}
		return nil, err
	}
	defer inFile.Close()
	var nw networkInterfacer = networkInterface{}
	return chooseHostInterfaceFromRoute(inFile, nw)
}

type networkInterfacer interface {
	InterfaceByName(intfName string) (*net.Interface, error)
	Addrs(intf *net.Interface) ([]net.Addr, error)
}

type networkInterface struct{}

func (_ networkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	intf, err := net.InterfaceByName(intfName)
	if err != nil {
		return nil, err
	}
	return intf, nil
}

func (_ networkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	addrs, err := intf.Addrs()
	if err != nil {
		return nil, err
	}
	return addrs, nil
}

func chooseHostInterfaceFromRoute(inFile io.Reader, nw networkInterfacer) (net.IP, error) {
	routes, err := getRoutes(inFile)
	if err != nil {
		return nil, err
	}
	if len(routes) == 0 {
		return nil, fmt.Errorf("No default routes.")
	}
	// TODO: append IPv6 routes for processing - currently only have IPv4 routes
	for _, family := range []AddressFamily{familyIPv4, familyIPv6} {
		glog.V(4).Infof("Looking for default routes with IPv%d addresses", uint(family))
		for _, route := range routes {
			// TODO: When have IPv6 routes, filter here to speed up processing
			// if route.Family != family {
			// 	continue
			// }
			glog.V(4).Infof("Default route transits interface %q", route.Interface)
			finalIP, err := getIPFromInterface(route.Interface, family, nw)
			if err != nil {
				return nil, err
			}
			if finalIP != nil {
				glog.V(4).Infof("Found active IP %v ", finalIP)
				return finalIP, nil
			}
		}
	}
	glog.V(4).Infof("No active IP found by looking at default routes")
	return nil, fmt.Errorf("unable to select an IP from default routes.")
}

// If bind-address is usable, return it directly
// If bind-address is not usable (unset, 0.0.0.0, or loopback), we will use the host's default
// interface.
func ChooseBindAddress(bindAddress net.IP) (net.IP, error) {
	if bindAddress == nil || bindAddress.IsUnspecified() || bindAddress.IsLoopback() {
		hostIP, err := ChooseHostInterface()
		if err != nil {
			return nil, err
		}
		bindAddress = hostIP
	}
	return bindAddress, nil
}
