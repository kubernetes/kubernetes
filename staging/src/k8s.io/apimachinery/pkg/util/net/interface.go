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

type Route struct {
	Interface   string
	Destination net.IP
	Gateway     net.IP
	// TODO: add more fields here if needed
}

const (
	ipCommand   = "/usr/sbin/ip"
	grepCommand = "/usr/bin/grep"
)

var (
	ipv4_zero = net.IP{0, 0, 0, 0}
	ipv6_zero = net.ParseIP("::")
)

var execCommand = exec.Command

// buildRouteCommand forms IP route command(s) to obtain routes and filter
// on the default routes. Can do IPv4 only, or IPv4 + IPv6.
func buildRouteCommand(withIPv6 bool) string {
	v6Command := ""
	if withIPv6 {
		v6Command = ipCommand + " -6 route; "
	}
	return fmt.Sprintf("{ %s route; %s} | %s default", ipCommand, v6Command, grepCommand)
}

// getRoutes will return the raw IPv4, and optionally IPv6, default routes.
func getDefaultRoutes(usingCmd string) (string, error) {
	routes, err := execCommand("sh", "-c", usingCmd).Output()
	if err != nil {
		return "", fmt.Errorf("unable to obtain route information: '%s'.", err)
	}
	return string(routes), nil
}

// parseDefaultRoutes extracts out the interface and gateway from the raw route info.
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
			return nil, fmt.Errorf("unable to parse gateway IP '%s'.", fields[2])
		}
		dest := ipv4_zero
		if gw.To4() == nil {
			dest = ipv6_zero
		}
		routes = append(routes, Route{Interface: fields[4], Destination: dest, Gateway: gw})
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

// findGlobalIP returns the first IP on interface that is not a loopback or
// link local address. Can optionally allow IPv6 addresses. Note: ordering
// of addresses processed is undefined.
func findGlobalIP(addrs []net.Addr, allowIPv6 bool) (net.IP, error) {
	if len(addrs) > 0 {
		for _, a := range addrs {
			glog.V(4).Infof("Checking address %q", a.String())
			ip, _, err := net.ParseCIDR(a.String())
			if err != nil {
				return nil, err
			}
			if ip.To4() == nil && !allowIPv6 {
				glog.V(4).Infof("Ignoring IPv6 address '%v' for IPv4 only mode", ip)
				continue
			}
			// TODO: Do we need to test IsLinkLocalMulticast or IsInterfaceLocalMulticast?
			// net.Addrs() indicates that it returns *unicast* addresses (and not multicast).
			if ip.IsLoopback() || ip.IsLinkLocalUnicast() {
				glog.V(4).Infof("Ignoring loopback/link-local '%v'", ip)
				continue
			}
			glog.V(4).Infof("IP found '%v'", ip)
			return ip, nil
		}
	}
	return nil, nil
}

// getIPFromInterface looks for a global address on an interface that is up
func getIPFromInterface(intfName string, nw networkInterfacer, allowIPv6 bool) (net.IP, error) {
	intf, err := nw.InterfaceByName(intfName)
	if err != nil {
		return nil, err
	}
	if isInterfaceUp(intf) {
		// Get all unicast addresses for interface
		addrs, err := nw.Addrs(intf)
		if err != nil {
			return nil, err
		}
		glog.V(4).Infof("Interface %q has %d addresses :%v.", intfName, len(addrs), addrs)
		ip, err := findGlobalIP(addrs, allowIPv6)
		if err != nil {
			return nil, err
		}
		if ip != nil {
			glog.V(4).Infof("valid IPv4 address for interface %q found as %v.", intfName, ip)
			return ip, nil
		}
	}
	return nil, nil
}

func chooseIPFromHostInterfaces(allowIPv6 bool, nw networkInterfacer) (net.IP, error) {
	intfs, err := nw.Interfaces()
	if err != nil {
		return nil, err
	}
	if len(intfs) == 0 {
		return nil, fmt.Errorf("no interfaces found on host.")
	}
	skipReason := ""
	for _, intf := range intfs {
		if !isInterfaceUp(&intf) {
			skipReason = "down interface"
			glog.V(4).Infof("Skipping %s %q", skipReason, intf.Name)
			continue
		}
		if isLoopbackOrPointToPoint(&intf) {
			skipReason = "LB or P2P interface"
			glog.V(4).Infof("Skipping %s %q", skipReason, intf.Name)
			continue
		}

		cidrs, err := nw.Addrs(&intf)
		if err != nil {
			return nil, err
		}
		if len(cidrs) == 0 {
			skipReason = "no addresses on interface"
			glog.V(4).Infof("Skipping %s %q", skipReason, intf.Name)
			continue
		}
		for _, cidr := range cidrs {
			ip, _, err := net.ParseCIDR(cidr.String())
			if err != nil {
				return nil, err
			}
			if !allowIPv6 && ip.To4() == nil {
				skipReason = "non-IPv4 address"
				glog.V(4).Infof("Skipping %s %q on interface %q", skipReason, ip, intf.Name)
				continue
			}
			if ip.IsLinkLocalUnicast() {
				skipReason = "link local address"
				glog.V(4).Infof("Skipping %s %q on interface %q", skipReason, ip, intf.Name)
				continue
			}
			return ip, nil
		}
	}
	return nil, fmt.Errorf("no acceptable interface from host (last cause: %s).", skipReason)
}

// ChooseHostInterface attempts to determine the IP for the node.
// It uses routing table data to find the interface with a default GW,
// checks if it is up, and returns IP from interface that is not a loopback
// or LL unicast address. If this fails, it will look through all system
// interfaces. If no interface is found (e.g. not connected to internet),
// an error is returned.
func ChooseHostInterface() (net.IP, error) {
	nw := networkInterface{}
	ip, err := chooseHostInterfaceFromRoute(false, nw)
	if err == nil {
		return ip, nil
	}
	return chooseIPFromHostInterfaces(false, nw)
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

func chooseHostInterfaceFromRoute(include_ipv6 bool, nw networkInterfacer) (net.IP, error) {
	cmd := buildRouteCommand(include_ipv6)
	raw, err := getDefaultRoutes(cmd)
	if err != nil {
		return nil, err
	}
	routes, err := parseDefaultRoutes(raw)
	if err != nil {
		return nil, err
	}

	for _, route := range routes {
		glog.V(4).Infof("Default route transits interface %q", route.Interface)
		ip, err := getIPFromInterface(route.Interface, nw, include_ipv6)
		if err != nil {
			return nil, err
		}
		if ip != nil {
			glog.V(4).Infof("Choosing IP %v ", ip)
			return ip, nil
		}
	}
	glog.V(4).Infof("No valid IP found")
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
