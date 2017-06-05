// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris windows

package icmp

import (
	"net"
	"os"
	"runtime"
	"syscall"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/ipv4"
	"golang.org/x/net/ipv6"
)

const sysIP_STRIPHDR = 0x17 // for now only darwin supports this option

// ListenPacket listens for incoming ICMP packets addressed to
// address. See net.Dial for the syntax of address.
//
// For non-privileged datagram-oriented ICMP endpoints, network must
// be "udp4" or "udp6". The endpoint allows to read, write a few
// limited ICMP messages such as echo request and echo reply.
// Currently only Darwin and Linux support this.
//
// Examples:
//	ListenPacket("udp4", "192.168.0.1")
//	ListenPacket("udp4", "0.0.0.0")
//	ListenPacket("udp6", "fe80::1%en0")
//	ListenPacket("udp6", "::")
//
// For privileged raw ICMP endpoints, network must be "ip4" or "ip6"
// followed by a colon and an ICMP protocol number or name.
//
// Examples:
//	ListenPacket("ip4:icmp", "192.168.0.1")
//	ListenPacket("ip4:1", "0.0.0.0")
//	ListenPacket("ip6:ipv6-icmp", "fe80::1%en0")
//	ListenPacket("ip6:58", "::")
func ListenPacket(network, address string) (*PacketConn, error) {
	var family, proto int
	switch network {
	case "udp4":
		family, proto = syscall.AF_INET, iana.ProtocolICMP
	case "udp6":
		family, proto = syscall.AF_INET6, iana.ProtocolIPv6ICMP
	default:
		i := last(network, ':')
		switch network[:i] {
		case "ip4":
			proto = iana.ProtocolICMP
		case "ip6":
			proto = iana.ProtocolIPv6ICMP
		}
	}
	var cerr error
	var c net.PacketConn
	switch family {
	case syscall.AF_INET, syscall.AF_INET6:
		s, err := syscall.Socket(family, syscall.SOCK_DGRAM, proto)
		if err != nil {
			return nil, os.NewSyscallError("socket", err)
		}
		defer syscall.Close(s)
		if runtime.GOOS == "darwin" && family == syscall.AF_INET {
			if err := syscall.SetsockoptInt(s, iana.ProtocolIP, sysIP_STRIPHDR, 1); err != nil {
				return nil, os.NewSyscallError("setsockopt", err)
			}
		}
		sa, err := sockaddr(family, address)
		if err != nil {
			return nil, err
		}
		if err := syscall.Bind(s, sa); err != nil {
			return nil, os.NewSyscallError("bind", err)
		}
		f := os.NewFile(uintptr(s), "datagram-oriented icmp")
		defer f.Close()
		c, cerr = net.FilePacketConn(f)
	default:
		c, cerr = net.ListenPacket(network, address)
	}
	if cerr != nil {
		return nil, cerr
	}
	switch proto {
	case iana.ProtocolICMP:
		return &PacketConn{c: c, p4: ipv4.NewPacketConn(c)}, nil
	case iana.ProtocolIPv6ICMP:
		return &PacketConn{c: c, p6: ipv6.NewPacketConn(c)}, nil
	default:
		return &PacketConn{c: c}, nil
	}
}
