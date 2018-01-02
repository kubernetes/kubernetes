// +build solaris

package netutils

import (
	"fmt"
	"net"
	"os/exec"
	"strings"

	"github.com/docker/libnetwork/ipamutils"
	"github.com/vishvananda/netlink"
)

var (
	networkGetRoutesFct func(netlink.Link, int) ([]netlink.Route, error)
)

// CheckRouteOverlaps checks whether the passed network overlaps with any existing routes
func CheckRouteOverlaps(toCheck *net.IPNet) error {
	return nil
}

// ElectInterfaceAddresses looks for an interface on the OS with the specified name
// and returns returns all its IPv4 and IPv6 addresses in CIDR notation.
// If a failure in retrieving the addresses or no IPv4 address is found, an error is returned.
// If the interface does not exist, it chooses from a predefined
// list the first IPv4 address which does not conflict with other
// interfaces on the system.
func ElectInterfaceAddresses(name string) ([]*net.IPNet, []*net.IPNet, error) {
	var (
		v4Net *net.IPNet
	)

	out, err := exec.Command("/usr/sbin/ipadm", "show-addr",
		"-p", "-o", "addrobj,addr").Output()
	if err != nil {
		fmt.Println("failed to list interfaces on system")
		return nil, nil, err
	}
	alist := strings.Fields(string(out))
	for _, a := range alist {
		linkandaddr := strings.SplitN(a, ":", 2)
		if len(linkandaddr) != 2 {
			fmt.Println("failed to check interfaces on system: ", a)
			continue
		}
		gw := fmt.Sprintf("%s_gw0", name)
		link := strings.Split(linkandaddr[0], "/")[0]
		addr := linkandaddr[1]
		if gw != link {
			continue
		}
		_, ipnet, err := net.ParseCIDR(addr)
		if err != nil {
			fmt.Println("failed to parse address: ", addr)
			continue
		}
		v4Net = ipnet
		break
	}
	if v4Net == nil {
		v4Net, err = FindAvailableNetwork(ipamutils.PredefinedBroadNetworks)
		if err != nil {
			return nil, nil, err
		}
	}
	return []*net.IPNet{v4Net}, nil, nil
}

// FindAvailableNetwork returns a network from the passed list which does not
// overlap with existing interfaces in the system
func FindAvailableNetwork(list []*net.IPNet) (*net.IPNet, error) {
	out, err := exec.Command("/usr/sbin/ipadm", "show-addr",
		"-p", "-o", "addr").Output()

	if err != nil {
		fmt.Println("failed to list interfaces on system")
		return nil, err
	}
	ipaddrs := strings.Fields(string(out))
	inuse := []*net.IPNet{}
	for _, ip := range ipaddrs {
		_, ipnet, err := net.ParseCIDR(ip)
		if err != nil {
			fmt.Println("failed to check interfaces on system: ", ip)
			continue
		}
		inuse = append(inuse, ipnet)
	}
	for _, avail := range list {
		is_avail := true
		for _, ipnet := range inuse {
			if NetworkOverlaps(avail, ipnet) {
				is_avail = false
				break
			}
		}
		if is_avail {
			return avail, nil
		}
	}
	return nil, fmt.Errorf("no available network")
}
