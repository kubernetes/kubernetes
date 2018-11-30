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

	"k8s.io/klog"
)

type AddressFamily uint

const (
	familyIPv4 AddressFamily = 4
	familyIPv6 AddressFamily = 6
)

const (
	ipv4RouteFile = "/proc/net/route"
	ipv6RouteFile = "/proc/net/ipv6_route"
)

type Route struct {
	Interface   string
	Destination net.IP
	Gateway     net.IP
	Family      AddressFamily
}

type RouteFile struct {
	name  string
	parse func(input io.Reader) ([]Route, error)
}

// noRoutesError can be returned by ChooseBindAddress() in case of no routes
type noRoutesError struct {
	message string
}

func (e noRoutesError) Error() string {
	return e.message
}

// IsNoRoutesError checks if an error is of type noRoutesError
func IsNoRoutesError(err error) bool {
	if err == nil {
		return false
	}
	switch err.(type) {
	case noRoutesError:
		return true
	default:
		return false
	}
}

var (
	v4File = RouteFile{name: ipv4RouteFile, parse: getIPv4DefaultRoutes}
	v6File = RouteFile{name: ipv6RouteFile, parse: getIPv6DefaultRoutes}
)

func (rf RouteFile) extract() ([]Route, error) {
	file, err := os.Open(rf.name)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return rf.parse(file)
}

// getIPv4DefaultRoutes obtains the IPv4 routes, and filters out non-default routes.
func getIPv4DefaultRoutes(input io.Reader) ([]Route, error) {
	routes := []Route{}
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
		// Interested in fields:
		//  0 - interface name
		//  1 - destination address
		//  2 - gateway
		dest, err := parseIP(fields[1], familyIPv4)
		if err != nil {
			return nil, err
		}
		gw, err := parseIP(fields[2], familyIPv4)
		if err != nil {
			return nil, err
		}
		if !dest.Equal(net.IPv4zero) {
			continue
		}
		routes = append(routes, Route{
			Interface:   fields[0],
			Destination: dest,
			Gateway:     gw,
			Family:      familyIPv4,
		})
	}
	return routes, nil
}

func getIPv6DefaultRoutes(input io.Reader) ([]Route, error) {
	routes := []Route{}
	scanner := bufio.NewReader(input)
	for {
		line, err := scanner.ReadString('\n')
		if err == io.EOF {
			break
		}
		fields := strings.Fields(line)
		// Interested in fields:
		//  0 - destination address
		//  4 - gateway
		//  9 - interface name
		dest, err := parseIP(fields[0], familyIPv6)
		if err != nil {
			return nil, err
		}
		gw, err := parseIP(fields[4], familyIPv6)
		if err != nil {
			return nil, err
		}
		if !dest.Equal(net.IPv6zero) {
			continue
		}
		if gw.Equal(net.IPv6zero) {
			continue // loopback
		}
		routes = append(routes, Route{
			Interface:   fields[9],
			Destination: dest,
			Gateway:     gw,
			Family:      familyIPv6,
		})
	}
	return routes, nil
}

// parseIP takes the hex IP address string from route file and converts it
// to a net.IP address. For IPv4, the value must be converted to big endian.
func parseIP(str string, family AddressFamily) (net.IP, error) {
	if str == "" {
		return nil, fmt.Errorf("input is nil")
	}
	bytes, err := hex.DecodeString(str)
	if err != nil {
		return nil, err
	}
	if family == familyIPv4 {
		if len(bytes) != net.IPv4len {
			return nil, fmt.Errorf("invalid IPv4 address in route")
		}
		return net.IP([]byte{bytes[3], bytes[2], bytes[1], bytes[0]}), nil
	}
	// Must be IPv6
	if len(bytes) != net.IPv6len {
		return nil, fmt.Errorf("invalid IPv6 address in route")
	}
	return net.IP(bytes), nil
}

func isInterfaceUp(intf *net.Interface) bool {
	if intf == nil {
		return false
	}
	if intf.Flags&net.FlagUp != 0 {
		klog.V(4).Infof("Interface %v is up", intf.Name)
		return true
	}
	return false
}

func isLoopbackOrPointToPoint(intf *net.Interface) bool {
	return intf.Flags&(net.FlagLoopback|net.FlagPointToPoint) != 0
}

// getMatchingGlobalIP returns the first valid global unicast address of the given
// 'family' from the list of 'addrs'.
func getMatchingGlobalIP(addrs []net.Addr, family AddressFamily) (net.IP, error) {
	if len(addrs) > 0 {
		for i := range addrs {
			klog.V(4).Infof("Checking addr  %s.", addrs[i].String())
			ip, _, err := net.ParseCIDR(addrs[i].String())
			if err != nil {
				return nil, err
			}
			if memberOf(ip, family) {
				if ip.IsGlobalUnicast() {
					klog.V(4).Infof("IP found %v", ip)
					return ip, nil
				} else {
					klog.V(4).Infof("Non-global unicast address found %v", ip)
				}
			} else {
				klog.V(4).Infof("%v is not an IPv%d address", ip, int(family))
			}

		}
	}
	return nil, nil
}

// getIPFromInterface gets the IPs on an interface and returns a global unicast address, if any. The
// interface must be up, the IP must in the family requested, and the IP must be a global unicast address.
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
		klog.V(4).Infof("Interface %q has %d addresses :%v.", intfName, len(addrs), addrs)
		matchingIP, err := getMatchingGlobalIP(addrs, forFamily)
		if err != nil {
			return nil, err
		}
		if matchingIP != nil {
			klog.V(4).Infof("Found valid IPv%d address %v for interface %q.", int(forFamily), matchingIP, intfName)
			return matchingIP, nil
		}
	}
	return nil, nil
}

// memberOF tells if the IP is of the desired family. Used for checking interface addresses.
func memberOf(ip net.IP, family AddressFamily) bool {
	if ip.To4() != nil {
		return family == familyIPv4
	} else {
		return family == familyIPv6
	}
}

// chooseIPFromHostInterfaces looks at all system interfaces, trying to find one that is up that
// has a global unicast address (non-loopback, non-link local, non-point2point), and returns the IP.
// Searches for IPv4 addresses, and then IPv6 addresses.
func chooseIPFromHostInterfaces(nw networkInterfacer) (net.IP, error) {
	intfs, err := nw.Interfaces()
	if err != nil {
		return nil, err
	}
	if len(intfs) == 0 {
		return nil, fmt.Errorf("no interfaces found on host.")
	}
	for _, family := range []AddressFamily{familyIPv4, familyIPv6} {
		klog.V(4).Infof("Looking for system interface with a global IPv%d address", uint(family))
		for _, intf := range intfs {
			if !isInterfaceUp(&intf) {
				klog.V(4).Infof("Skipping: down interface %q", intf.Name)
				continue
			}
			if isLoopbackOrPointToPoint(&intf) {
				klog.V(4).Infof("Skipping: LB or P2P interface %q", intf.Name)
				continue
			}
			addrs, err := nw.Addrs(&intf)
			if err != nil {
				return nil, err
			}
			if len(addrs) == 0 {
				klog.V(4).Infof("Skipping: no addresses on interface %q", intf.Name)
				continue
			}
			for _, addr := range addrs {
				ip, _, err := net.ParseCIDR(addr.String())
				if err != nil {
					return nil, fmt.Errorf("Unable to parse CIDR for interface %q: %s", intf.Name, err)
				}
				if !memberOf(ip, family) {
					klog.V(4).Infof("Skipping: no address family match for %q on interface %q.", ip, intf.Name)
					continue
				}
				// TODO: Decide if should open up to allow IPv6 LLAs in future.
				if !ip.IsGlobalUnicast() {
					klog.V(4).Infof("Skipping: non-global address %q on interface %q.", ip, intf.Name)
					continue
				}
				klog.V(4).Infof("Found global unicast address %q on interface %q.", ip, intf.Name)
				return ip, nil
			}
		}
	}
	return nil, fmt.Errorf("no acceptable interface with global unicast address found on host")
}

// ChooseHostInterface is a method used fetch an IP for a daemon.
// If there is no routing info file, it will choose a global IP from the system
// interfaces. Otherwise, it will use IPv4 and IPv6 route information to return the
// IP of the interface with a gateway on it (with priority given to IPv4). For a node
// with no internet connection, it returns error.
func ChooseHostInterface() (net.IP, error) {
	var nw networkInterfacer = networkInterface{}
	if _, err := os.Stat(ipv4RouteFile); os.IsNotExist(err) {
		return chooseIPFromHostInterfaces(nw)
	}
	routes, err := getAllDefaultRoutes()
	if err != nil {
		return nil, err
	}
	return chooseHostInterfaceFromRoute(routes, nw)
}

// networkInterfacer defines an interface for several net library functions. Production
// code will forward to net library functions, and unit tests will override the methods
// for testing purposes.
type networkInterfacer interface {
	InterfaceByName(intfName string) (*net.Interface, error)
	Addrs(intf *net.Interface) ([]net.Addr, error)
	Interfaces() ([]net.Interface, error)
}

// networkInterface implements the networkInterfacer interface for production code, just
// wrapping the underlying net library function calls.
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

// getAllDefaultRoutes obtains IPv4 and IPv6 default routes on the node. If unable
// to read the IPv4 routing info file, we return an error. If unable to read the IPv6
// routing info file (which is optional), we'll just use the IPv4 route information.
// Using all the routing info, if no default routes are found, an error is returned.
func getAllDefaultRoutes() ([]Route, error) {
	routes, err := v4File.extract()
	if err != nil {
		return nil, err
	}
	v6Routes, _ := v6File.extract()
	routes = append(routes, v6Routes...)
	if len(routes) == 0 {
		return nil, noRoutesError{
			message: fmt.Sprintf("no default routes found in %q or %q", v4File.name, v6File.name),
		}
	}
	return routes, nil
}

// chooseHostInterfaceFromRoute cycles through each default route provided, looking for a
// global IP address from the interface for the route. Will first look all each IPv4 route for
// an IPv4 IP, and then will look at each IPv6 route for an IPv6 IP.
func chooseHostInterfaceFromRoute(routes []Route, nw networkInterfacer) (net.IP, error) {
	for _, family := range []AddressFamily{familyIPv4, familyIPv6} {
		klog.V(4).Infof("Looking for default routes with IPv%d addresses", uint(family))
		for _, route := range routes {
			if route.Family != family {
				continue
			}
			klog.V(4).Infof("Default route transits interface %q", route.Interface)
			finalIP, err := getIPFromInterface(route.Interface, family, nw)
			if err != nil {
				return nil, err
			}
			if finalIP != nil {
				klog.V(4).Infof("Found active IP %v ", finalIP)
				return finalIP, nil
			}
		}
	}
	klog.V(4).Infof("No active IP found by looking at default routes")
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
