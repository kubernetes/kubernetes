package netutils

import (
	"net"

	"github.com/docker/libnetwork/types"
)

// ElectInterfaceAddresses looks for an interface on the OS with the specified name
// and returns returns all its IPv4 and IPv6 addresses in CIDR notation.
// If a failure in retrieving the addresses or no IPv4 address is found, an error is returned.
// If the interface does not exist, it chooses from a predefined
// list the first IPv4 address which does not conflict with other
// interfaces on the system.
func ElectInterfaceAddresses(name string) ([]*net.IPNet, []*net.IPNet, error) {
	return nil, nil, types.NotImplementedErrorf("not supported on freebsd")
}

// FindAvailableNetwork returns a network from the passed list which does not
// overlap with existing interfaces in the system
func FindAvailableNetwork(list []*net.IPNet) (*net.IPNet, error) {
	return nil, types.NotImplementedErrorf("not supported on freebsd")
}
