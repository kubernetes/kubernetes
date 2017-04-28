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
	"fmt"
	"net"
	"os/exec"

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
	Family      AddressFamily
}

type FamilyFlags uint

const (
	ipCommand   = "/usr/sbin/ip"
	grepCommand = "/usr/bin/grep"
)

var execCommand = exec.Command

// buildRouteCommands forms IP route command(s) to obtain routes and then filter
// on the default routes.
func buildRouteCommands() string {
	return fmt.Sprintf("{ %s route; %s -6 route; } | %s default", ipCommand, ipCommand, grepCommand)
}

// getDefaultRoutes will return the raw IPv4 and IPv6 default routes.
func getDefaultRoutes(usingCmd string) (string, error) {
	routes, err := execCommand("sh", "-c", usingCmd).Output()
	if err != nil {
		return "", fmt.Errorf("unable to obtain route information: '%s'.", err)
	}
	return string(routes), nil
}

// parseDefaultRoutes extracts out the interface and gateway from the raw default
// route info. Entry is skipped, if GW is not global.
func parseDefaultRoutes(route_lines string) ([]Route, error) {
	routes := []Route{}
	rows := strings.Split(route_lines, "\n")
	for _, row := range rows {
		fields := strings.Split(row, " ")
		if len(fields) < 5 {
			continue // skip blank and malformed lines
		}
		gw := net.ParseIP(fields[2])
		if gw == nil {
			return nil, fmt.Errorf("unable to parse gateway IP %q.", fields[2])
		}
		if !gw.IsGlobalUnicast() {
			glog.V(4).Infof("Skipping default route with local gateway IP %q.", gw)
			continue
		}
		dest := net.IPv4zero
		family := familyIPv4
		if gw.To4() == nil {
			dest = net.IPv6zero
			family = familyIPv6
		}
		routes = append(routes, Route{Interface: fields[4], Destination: dest, Gateway: gw, Family: family})
	}
	return routes, nil
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

func isLoopbackOrPointToPoint(intf *net.Interface) bool {
	if intf.Flags&(net.FlagLoopback|net.FlagPointToPoint) != 0 {
		return true
	}
	return false
}

// findMatchingIP checks all the interface's addresses and finds the one that
// is on the same network as the default route's GW. This implies that the
// IP is of the correct family (since caller filters GW by family), and
// is global (only default routes with global GW IP are selected previously).
func findMatchingIP(route Route, addrs []net.Addr) (net.IP, error) {
	if len(addrs) > 0 {
		for _, a := range addrs {
			glog.V(4).Infof("Checking address %q", a.String())
			ip, net, err := net.ParseCIDR(a.String())
			if err != nil {
				return nil, err
			}
			if net.Contains(route.Gateway) {
				glog.V(4).Infof("IP found '%v'", ip)
				return ip, nil
			}
			glog.V(4).Info("Skipping: IP not on same network as default route GW %q", route.Gateway)
		}
	}
	return nil, nil
}

// getIPFromInterface gets the interface IP that is on the network of the
// default route. Interface must be up.
func getIPFromInterface(route Route, nw networkInterfacer) (net.IP, error) {
	intf, err := nw.InterfaceByName(route.Interface)
	if err != nil {
		return nil, err
	}
	if isInterfaceUp(intf) {
		// Get all unicast addresses for interface
		addrs, err := nw.Addrs(intf)
		if err != nil {
			return nil, err
		}
		glog.V(4).Infof("Interface %q has %d addresses :%v.", route.Interface, len(addrs), addrs)
		ip, err := findMatchingIP(route, addrs)
		if err != nil {
			return nil, err
		}
		if ip != nil {
			glog.V(4).Infof("Found IP %q on interface %q matching default route.", ip, route.Interface)
			return ip, nil
		}
	}
	return nil, nil
}

// memberOF tells if the IP is of the desired family. Used for checking interface addresses.
func memberOf(ip net.IP, family AddressFamily) bool {
	ipFamily := familyIPv4
	if ip.To4() == nil {
		ipFamily = familyIPv6
	}
	return ipFamily == family
}

// chooseIPFromHostInterfaces looks at all system interfaces, trying to find one that is up that
// has a global IP, and returns the IP. Searches for IPv4 addresses, and then IPv6 addresses.
func chooseIPFromHostInterfaces(nw networkInterfacer) (net.IP, error) {
	intfs, err := nw.Interfaces()
	if err != nil {
		return nil, err
	}
	if len(intfs) == 0 {
		return nil, fmt.Errorf("no interfaces found on host.")
	}
	skipReason := ""
	for _, family := range []AddressFamily{familyIPv4, familyIPv6} {
		glog.V(4).Infof("Looking for system interface with a global IPv%d address", uint(family))
		for _, intf := range intfs {
			if !isInterfaceUp(&intf) {
				skipReason = "down interface"
				glog.V(4).Infof("Skipping: %s %q", skipReason, intf.Name)
				continue
			}
			if isLoopbackOrPointToPoint(&intf) {
				skipReason = "LB or P2P interface"
				glog.V(4).Infof("Skipping: %s %q", skipReason, intf.Name)
				continue
			}
			cidrs, err := nw.Addrs(&intf)
			if err != nil {
				return nil, err
			}
			if len(cidrs) == 0 {
				skipReason = "no addresses on interface"
				glog.V(4).Infof("Skipping: %s %q", skipReason, intf.Name)
				continue
			}
			for _, cidr := range cidrs {
				ip, _, err := net.ParseCIDR(cidr.String())
				if err != nil {
					return nil, err
				}
				if !memberOf(ip, family) {
					skipReason = "no address family match"
					glog.V(4).Infof("Skipping: %s for %q on interface %q.", skipReason, ip, intf.Name)
					continue
				}
				if !ip.IsGlobalUnicast() {
					skipReason = "non-global address"
					glog.V(4).Infof("Skipping: %s %q on interface %q.", skipReason, ip, intf.Name)
					continue
				}
				glog.V(4).Infof("Found global IP %q on interface %q.", ip, intf.Name)
				return ip, nil
			}
		}
	}
	return nil, fmt.Errorf("no acceptable interface from host (last cause: %s).", skipReason)
}

// ChooseHostInterface attempts to determine the (global) IP for the node.
// First, it looks at IPs on interfaces referenced by default routes. If
// that fails, it will look at all system interfaces. In all cases, the
// interface must be up, and the IP must be a global address. Preference
// is given to IPv4 addresses.
func ChooseHostInterface() (net.IP, error) {
	nw := networkInterface{}
	ip, err := chooseHostInterfaceFromRoute(nw)
	if err == nil {
		return ip, nil
	}
	return chooseIPFromHostInterfaces(nw)
}

type networkInterfacer interface {
	InterfaceByName(intfName string) (*net.Interface, error)
	Addrs(intf *net.Interface) ([]net.Addr, error)
	Interfaces() ([]net.Interface, error)
}

type networkInterface struct{}

func (_ networkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	return net.InterfaceByName(intfName)
}

func (_ networkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	return intf.Addrs()
}

func (_ networkInterface) Interfaces() ([]net.Interface, error) {
	return net.Interfaces()
}

// chooseHostInterfaceFromRoute looks at interfaces with default routes and sees if there
// is a global IP on the same network as the gateway. Search for IPv4 and then IPv6 default
// routes.
func chooseHostInterfaceFromRoute(nw networkInterfacer) (net.IP, error) {
	cmd := buildRouteCommands()
	raw, err := getDefaultRoutes(cmd)
	if err != nil {
		return nil, err
	}
	routes, err := parseDefaultRoutes(raw)
	if err != nil {
		return nil, err
	}
	for _, family := range []AddressFamily{familyIPv4, familyIPv6} {
		glog.V(4).Infof("Looking for default routes with IPv%d addresses", uint(family))
		for _, route := range routes {
			if route.Family != family {
				continue
			}
			glog.V(4).Infof("Default route transits interface %q", route.Interface)
			ip, err := getIPFromInterface(route, nw)
			if err != nil {
				return nil, err
			}
			if ip != nil {
				glog.V(4).Infof("Found active IP %v ", ip)
				return ip, nil
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
