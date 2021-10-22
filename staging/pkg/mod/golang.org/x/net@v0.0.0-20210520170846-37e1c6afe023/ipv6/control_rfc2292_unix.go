// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin
// +build darwin

package ipv6

import (
	"unsafe"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/internal/socket"

	"golang.org/x/sys/unix"
)

func marshal2292HopLimit(b []byte, cm *ControlMessage) []byte {
	m := socket.ControlMessage(b)
	m.MarshalHeader(iana.ProtocolIPv6, unix.IPV6_2292HOPLIMIT, 4)
	if cm != nil {
		socket.NativeEndian.PutUint32(m.Data(4), uint32(cm.HopLimit))
	}
	return m.Next(4)
}

func marshal2292PacketInfo(b []byte, cm *ControlMessage) []byte {
	m := socket.ControlMessage(b)
	m.MarshalHeader(iana.ProtocolIPv6, unix.IPV6_2292PKTINFO, sizeofInet6Pktinfo)
	if cm != nil {
		pi := (*inet6Pktinfo)(unsafe.Pointer(&m.Data(sizeofInet6Pktinfo)[0]))
		if ip := cm.Src.To16(); ip != nil && ip.To4() == nil {
			copy(pi.Addr[:], ip)
		}
		if cm.IfIndex > 0 {
			pi.setIfindex(cm.IfIndex)
		}
	}
	return m.Next(sizeofInet6Pktinfo)
}

func marshal2292NextHop(b []byte, cm *ControlMessage) []byte {
	m := socket.ControlMessage(b)
	m.MarshalHeader(iana.ProtocolIPv6, unix.IPV6_2292NEXTHOP, sizeofSockaddrInet6)
	if cm != nil {
		sa := (*sockaddrInet6)(unsafe.Pointer(&m.Data(sizeofSockaddrInet6)[0]))
		sa.setSockaddr(cm.NextHop, cm.IfIndex)
	}
	return m.Next(sizeofSockaddrInet6)
}
