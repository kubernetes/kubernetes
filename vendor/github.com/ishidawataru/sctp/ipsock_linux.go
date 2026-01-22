// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the GO_LICENSE file.

package sctp

import (
	"net"
	"os"
	"sync"
	"syscall"
)

//from https://github.com/golang/go
// Boolean to int.
func boolint(b bool) int {
	if b {
		return 1
	}
	return 0
}

//from https://github.com/golang/go
func ipToSockaddr(family int, ip net.IP, port int, zone string) (syscall.Sockaddr, error) {
	switch family {
	case syscall.AF_INET:
		if len(ip) == 0 {
			ip = net.IPv4zero
		}
		ip4 := ip.To4()
		if ip4 == nil {
			return nil, &net.AddrError{Err: "non-IPv4 address", Addr: ip.String()}
		}
		sa := &syscall.SockaddrInet4{Port: port}
		copy(sa.Addr[:], ip4)
		return sa, nil
	case syscall.AF_INET6:
		// In general, an IP wildcard address, which is either
		// "0.0.0.0" or "::", means the entire IP addressing
		// space. For some historical reason, it is used to
		// specify "any available address" on some operations
		// of IP node.
		//
		// When the IP node supports IPv4-mapped IPv6 address,
		// we allow an listener to listen to the wildcard
		// address of both IP addressing spaces by specifying
		// IPv6 wildcard address.
		if len(ip) == 0 || ip.Equal(net.IPv4zero) {
			ip = net.IPv6zero
		}
		// We accept any IPv6 address including IPv4-mapped
		// IPv6 address.
		ip6 := ip.To16()
		if ip6 == nil {
			return nil, &net.AddrError{Err: "non-IPv6 address", Addr: ip.String()}
		}
		//we set ZoneId to 0, as currently we use this functon only to probe the IP capabilities of the host
		//if real Zone handling is required, the zone cache implementation in golang/net should be pulled here
		sa := &syscall.SockaddrInet6{Port: port, ZoneId: 0}
		copy(sa.Addr[:], ip6)
		return sa, nil
	}
	return nil, &net.AddrError{Err: "invalid address family", Addr: ip.String()}
}

//from https://github.com/golang/go
func sockaddr(a *net.TCPAddr, family int) (syscall.Sockaddr, error) {
	if a == nil {
		return nil, nil
	}
	return ipToSockaddr(family, a.IP, a.Port, a.Zone)
}

//from https://github.com/golang/go
type ipStackCapabilities struct {
	sync.Once             // guards following
	ipv4Enabled           bool
	ipv6Enabled           bool
	ipv4MappedIPv6Enabled bool
}

//from https://github.com/golang/go
var ipStackCaps ipStackCapabilities

//from https://github.com/golang/go
// supportsIPv4 reports whether the platform supports IPv4 networking
// functionality.
func supportsIPv4() bool {
	ipStackCaps.Once.Do(ipStackCaps.probe)
	return ipStackCaps.ipv4Enabled
}

//from https://github.com/golang/go
// supportsIPv6 reports whether the platform supports IPv6 networking
// functionality.
func supportsIPv6() bool {
	ipStackCaps.Once.Do(ipStackCaps.probe)
	return ipStackCaps.ipv6Enabled
}

//from https://github.com/golang/go
// supportsIPv4map reports whether the platform supports mapping an
// IPv4 address inside an IPv6 address at transport layer
// protocols. See RFC 4291, RFC 4038 and RFC 3493.
func supportsIPv4map() bool {
	ipStackCaps.Once.Do(ipStackCaps.probe)
	return ipStackCaps.ipv4MappedIPv6Enabled
}

//from https://github.com/golang/go
// Probe probes IPv4, IPv6 and IPv4-mapped IPv6 communication
// capabilities which are controlled by the IPV6_V6ONLY socket option
// and kernel configuration.
//
// Should we try to use the IPv4 socket interface if we're only
// dealing with IPv4 sockets? As long as the host system understands
// IPv4-mapped IPv6, it's okay to pass IPv4-mapeed IPv6 addresses to
// the IPv6 interface. That simplifies our code and is most
// general. Unfortunately, we need to run on kernels built without
// IPv6 support too. So probe the kernel to figure it out.
func (p *ipStackCapabilities) probe() {
	s, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_STREAM, syscall.IPPROTO_TCP)
	switch err {
	case syscall.EAFNOSUPPORT, syscall.EPROTONOSUPPORT:
	case nil:
		syscall.Close(s)
		p.ipv4Enabled = true
	}
	var probes = []struct {
		laddr net.TCPAddr
		value int
	}{
		// IPv6 communication capability
		{laddr: net.TCPAddr{IP: net.IPv6loopback}, value: 1},
		// IPv4-mapped IPv6 address communication capability
		{laddr: net.TCPAddr{IP: net.IPv4(127, 0, 0, 1)}, value: 0},
	}

	for i := range probes {
		s, err := syscall.Socket(syscall.AF_INET6, syscall.SOCK_STREAM, syscall.IPPROTO_TCP)
		if err != nil {
			continue
		}
		defer syscall.Close(s)
		syscall.SetsockoptInt(s, syscall.IPPROTO_IPV6, syscall.IPV6_V6ONLY, probes[i].value)
		sa, err := sockaddr(&(probes[i].laddr), syscall.AF_INET6)
		if err != nil {
			continue
		}
		if err := syscall.Bind(s, sa); err != nil {
			continue
		}
		if i == 0 {
			p.ipv6Enabled = true
		} else {
			p.ipv4MappedIPv6Enabled = true
		}
	}
}

//from https://github.com/golang/go
//Change: we check the first IP address in the list of candidate SCTP IP addresses
func (a *SCTPAddr) isWildcard() bool {
	if a == nil {
		return true
	}
	if 0 == len(a.IPAddrs) {
		return true
	}

	return a.IPAddrs[0].IP.IsUnspecified()
}

func (a *SCTPAddr) family() int {
	if a != nil {
		for _, ip := range a.IPAddrs {
			if ip.IP.To4() == nil {
				return syscall.AF_INET6
			}
		}
	}
	return syscall.AF_INET
}

//from https://github.com/golang/go
func favoriteAddrFamily(network string, laddr *SCTPAddr, raddr *SCTPAddr, mode string) (family int, ipv6only bool) {
	switch network[len(network)-1] {
	case '4':
		return syscall.AF_INET, false
	case '6':
		return syscall.AF_INET6, true
	}

	if mode == "listen" && (laddr == nil || laddr.isWildcard()) {
		if supportsIPv4map() || !supportsIPv4() {
			return syscall.AF_INET6, false
		}
		if laddr == nil {
			return syscall.AF_INET, false
		}
		return laddr.family(), false
	}

	if (laddr == nil || laddr.family() == syscall.AF_INET) &&
		(raddr == nil || raddr.family() == syscall.AF_INET) {
		return syscall.AF_INET, false
	}
	return syscall.AF_INET6, false
}

//from https://github.com/golang/go
//Changes: it is for SCTP only
func setDefaultSockopts(s int, family int, ipv6only bool) error {
	if family == syscall.AF_INET6 {
		// Allow both IP versions even if the OS default
		// is otherwise. Note that some operating systems
		// never admit this option.
		syscall.SetsockoptInt(s, syscall.IPPROTO_IPV6, syscall.IPV6_V6ONLY, boolint(ipv6only))
	}
	// Allow broadcast.
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_BROADCAST, 1))
}
