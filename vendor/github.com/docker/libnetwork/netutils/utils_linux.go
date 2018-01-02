// +build linux
// Network utility functions.

package netutils

import (
	"fmt"
	"net"
	"strings"

	"github.com/docker/libnetwork/ipamutils"
	"github.com/docker/libnetwork/ns"
	"github.com/docker/libnetwork/osl"
	"github.com/docker/libnetwork/resolvconf"
	"github.com/docker/libnetwork/types"
	"github.com/vishvananda/netlink"
)

var (
	networkGetRoutesFct func(netlink.Link, int) ([]netlink.Route, error)
)

// CheckRouteOverlaps checks whether the passed network overlaps with any existing routes
func CheckRouteOverlaps(toCheck *net.IPNet) error {
	if networkGetRoutesFct == nil {
		networkGetRoutesFct = ns.NlHandle().RouteList
	}
	networks, err := networkGetRoutesFct(nil, netlink.FAMILY_V4)
	if err != nil {
		return err
	}
	for _, network := range networks {
		if network.Dst != nil && NetworkOverlaps(toCheck, network.Dst) {
			return ErrNetworkOverlaps
		}
	}
	return nil
}

// GenerateIfaceName returns an interface name using the passed in
// prefix and the length of random bytes. The api ensures that the
// there are is no interface which exists with that name.
func GenerateIfaceName(nlh *netlink.Handle, prefix string, len int) (string, error) {
	linkByName := netlink.LinkByName
	if nlh != nil {
		linkByName = nlh.LinkByName
	}
	for i := 0; i < 3; i++ {
		name, err := GenerateRandomName(prefix, len)
		if err != nil {
			continue
		}
		_, err = linkByName(name)
		if err != nil {
			if strings.Contains(err.Error(), "not found") {
				return name, nil
			}
			return "", err
		}
	}
	return "", types.InternalErrorf("could not generate interface name")
}

// ElectInterfaceAddresses looks for an interface on the OS with the
// specified name and returns returns all its IPv4 and IPv6 addresses in CIDR notation.
// If a failure in retrieving the addresses or no IPv4 address is found, an error is returned.
// If the interface does not exist, it chooses from a predefined
// list the first IPv4 address which does not conflict with other
// interfaces on the system.
func ElectInterfaceAddresses(name string) ([]*net.IPNet, []*net.IPNet, error) {
	var (
		v4Nets []*net.IPNet
		v6Nets []*net.IPNet
	)

	defer osl.InitOSContext()()

	link, _ := ns.NlHandle().LinkByName(name)
	if link != nil {
		v4addr, err := ns.NlHandle().AddrList(link, netlink.FAMILY_V4)
		if err != nil {
			return nil, nil, err
		}
		v6addr, err := ns.NlHandle().AddrList(link, netlink.FAMILY_V6)
		if err != nil {
			return nil, nil, err
		}
		for _, nlAddr := range v4addr {
			v4Nets = append(v4Nets, nlAddr.IPNet)
		}
		for _, nlAddr := range v6addr {
			v6Nets = append(v6Nets, nlAddr.IPNet)
		}
	}

	if link == nil || len(v4Nets) == 0 {
		// Choose from predefined broad networks
		v4Net, err := FindAvailableNetwork(ipamutils.PredefinedBroadNetworks)
		if err != nil {
			return nil, nil, err
		}
		v4Nets = append(v4Nets, v4Net)
	}

	return v4Nets, v6Nets, nil
}

// FindAvailableNetwork returns a network from the passed list which does not
// overlap with existing interfaces in the system
func FindAvailableNetwork(list []*net.IPNet) (*net.IPNet, error) {
	// We don't check for an error here, because we don't really care if we
	// can't read /etc/resolv.conf. So instead we skip the append if resolvConf
	// is nil. It either doesn't exist, or we can't read it for some reason.
	var nameservers []string
	if rc, err := resolvconf.Get(); err == nil {
		nameservers = resolvconf.GetNameserversAsCIDR(rc.Content)
	}
	for _, nw := range list {
		if err := CheckNameserverOverlaps(nameservers, nw); err == nil {
			if err := CheckRouteOverlaps(nw); err == nil {
				return nw, nil
			}
		}
	}
	return nil, fmt.Errorf("no available network")
}
