// Package netlink provides a simple library for netlink. Netlink is
// the interface a user-space program in linux uses to communicate with
// the kernel. It can be used to add and remove interfaces, set up ip
// addresses and routes, and confiugre ipsec. Netlink communication
// requires elevated privileges, so in most cases this code needs to
// be run as root. The low level primitives for netlink are contained
// in the nl subpackage. This package attempts to provide a high-level
// interface that is loosly modeled on the iproute2 cli.
package netlink

import (
	"net"

	"github.com/vishvananda/netlink/nl"
)

const (
	// Family type definitions
	FAMILY_ALL = nl.FAMILY_ALL
	FAMILY_V4  = nl.FAMILY_V4
	FAMILY_V6  = nl.FAMILY_V6
)

// ParseIPNet parses a string in ip/net format and returns a net.IPNet.
// This is valuable because addresses in netlink are often IPNets and
// ParseCIDR returns an IPNet with the IP part set to the base IP of the
// range.
func ParseIPNet(s string) (*net.IPNet, error) {
	ip, ipNet, err := net.ParseCIDR(s)
	if err != nil {
		return nil, err
	}
	return &net.IPNet{IP: ip, Mask: ipNet.Mask}, nil
}

// NewIPNet generates an IPNet from an ip address using a netmask of 32.
func NewIPNet(ip net.IP) *net.IPNet {
	return &net.IPNet{IP: ip, Mask: net.CIDRMask(32, 32)}
}
