/*
Copyright 2024 The Kubernetes Authors.

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

package ip

import (
	"net"
	"net/netip"
	"strings"

	netutils "k8s.io/utils/net"
)

// AddrFromIP converts a net.IP to a netip.Addr. Given a valid input this will always
// succeed; it will return the zero netip.Addr on nil or garbage input.
//
// Use this rather than netip.AddrFromSlice(), which does not correctly handle net.IP's
// 16-byte encoding of IPv4 addresses.
func AddrFromIP(ip net.IP) netip.Addr {
	// The net.IP and netip.Addr parsers/stringers both round-trip correctly with
	// respect to themselves:
	//
	//     net.ParseIP("1.2.3.4").String() => "1.2.3.4"
	//     netip.MustParseAddr("1.2.3.4").String() => "1.2.3.4"
	//
	// But if you parse an IPv4 string as a net.IP and naively convert it to
	// netip.Addr, you get something that netip.Addr considers to be an IPv6 address:
	//
	//     netip.AddrFromSlice(net.ParseIP("1.2.3.4")).String() => "::ffff:1.2.3.4"
	//     netip.AddrFromSlice(net.ParseIP("1.2.3.4")).Is4() => false
	//     netip.AddrFromSlice(net.ParseIP("1.2.3.4")).Is6() => true
	//
	// To get the desired results, you have to convert IPv4-mapped IPv6 addresses to
	// 4-byte IPv4 addresses first.
	if ip4 := ip.To4(); ip4 != nil {
		ip = ip4
	}
	addr, _ := netip.AddrFromSlice(ip)
	return addr
}

// IPFromAddr converts a netip.Addr to a net.IP. Given a valid input this will always
// succeed; it will return nil if addr is the zero netip.Addr.
func IPFromAddr(addr netip.Addr) net.IP {
	// addr.AsSlice() returns:
	//   - a []byte of length 4 if addr is an IPv4 address
	//   - a []byte of length 16 if addr is an IPv6 address
	//   - nil if addr is the zero Addr (which is the only other possibility)
	//
	// Any of those values can be correctly cast directly to a net.IP.
	return net.IP(addr.AsSlice())
}

// AddrFromInterfaceAddr converts a net.Addr returned from net.InterfaceAddrs(),
// net.Interface.Addrs(), or net.Interface.MulticastAddrs() to a netip.Addr. Calling it on
// other kinds of net.Addr values (such as a net.TCPAddr) will generally fail and return
// the zero netip.Addr.
func AddrFromInterfaceAddr(ifaddr net.Addr) netip.Addr {
	return AddrFromIP(IPFromInterfaceAddr(ifaddr))
}

// IPFromInterfaceAddr converts a net.Addr returned from net.InterfaceAddrs(),
// net.Interface.Addrs(), or net.Interface.MulticastAddrs() to a net.IP. Calling it on
// other kinds of net.Addr values (such as a net.TCPAddr) will generally fail and return
// nil.
func IPFromInterfaceAddr(ifaddr net.Addr) net.IP {
	// On both Linux and Windows, the values returned from the "interface addr"
	// methods are currently *net.IPNet for unicast addresses or *net.IPAddr for
	// multicast addresses.
	if ipnet, ok := ifaddr.(*net.IPNet); ok {
		return ipnet.IP
	} else if ipaddr, ok := ifaddr.(*net.IPAddr); ok {
		return ipaddr.IP
	}

	// Try to deal with other similar types... in particular, this is needed for
	// some existing unit tests...
	addrStr := ifaddr.String()
	// If it has a subnet length (like net.IPNet) or optional zone identifier (like
	// net.IPAddr), trim that away.
	if end := strings.IndexAny(addrStr, "/%"); end != -1 {
		addrStr = addrStr[:end]
	}
	// What's left is either an IP address, or something we can't parse.
	return netutils.ParseIPSloppy(addrStr)
}

// PrefixFromIPNet converts a *net.IPNet to a netip.Prefix. Given a valid input this will
// always succeed; it will return the zero netip.Prefix on nil or garbage input.
func PrefixFromIPNet(ipnet *net.IPNet) netip.Prefix {
	if ipnet == nil {
		return netip.Prefix{}
	}

	addr := AddrFromIP(ipnet.IP)
	if !addr.IsValid() {
		return netip.Prefix{}
	}

	prefixLen, bits := ipnet.Mask.Size()
	if prefixLen == 0 && bits == 0 {
		// non-CIDR Mask representation
		return netip.Prefix{}
	}
	if bits == 128 && addr.Is4() && (bits-prefixLen <= 32) {
		// Given an IPv4 IP and a 128-bit mask whose top bits are all "1",
		// net.IPNet/net.IP uses the lower bits as a 32-bit mask.
		prefixLen -= 128 - 32
	} else if bits != addr.BitLen() {
		// invalid IPv4/IPv6 mix
		return netip.Prefix{}
	}

	return netip.PrefixFrom(addr, prefixLen)
}

// IPNetFromPrefix converts a netip.Prefix to a *net.IPNet. Given a valid input this will
// always succeed; it will return nil if prefix is the zero netip.Prefix.
func IPNetFromPrefix(prefix netip.Prefix) *net.IPNet {
	addr := prefix.Addr()
	bits := prefix.Bits()

	// netip.Prefix allows you to construct a prefix using an IPv4-mapped IPv6
	// address, but it has broken semantics. (It requires the prefix length to be
	// between 0 and 32, as though it was an IPv4 CIDR, but if you call .Mask() on it,
	// it applies the prefix length as though it was an IPv6 CIDR, meaning you always
	// get back `::` regardless of the input, since the top 32 bits of an IPv4-mapped
	// IPv6 address are always 0.) So we just treat that as invalid.
	if bits == -1 || !addr.IsValid() || addr.Is4In6() {
		return nil
	}

	return &net.IPNet{
		IP:   IPFromAddr(addr),
		Mask: net.CIDRMask(bits, addr.BitLen()),
	}
}
