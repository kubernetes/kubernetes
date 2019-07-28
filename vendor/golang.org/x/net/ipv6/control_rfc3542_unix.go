// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package ipv6

import (
	"net"
	"unsafe"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/internal/socket"
)

func marshalTrafficClass(b []byte, cm *ControlMessage) []byte {
	m := socket.ControlMessage(b)
	m.MarshalHeader(iana.ProtocolIPv6, sysIPV6_TCLASS, 4)
	if cm != nil {
		socket.NativeEndian.PutUint32(m.Data(4), uint32(cm.TrafficClass))
	}
	return m.Next(4)
}

func parseTrafficClass(cm *ControlMessage, b []byte) {
	cm.TrafficClass = int(socket.NativeEndian.Uint32(b[:4]))
}

func marshalHopLimit(b []byte, cm *ControlMessage) []byte {
	m := socket.ControlMessage(b)
	m.MarshalHeader(iana.ProtocolIPv6, sysIPV6_HOPLIMIT, 4)
	if cm != nil {
		socket.NativeEndian.PutUint32(m.Data(4), uint32(cm.HopLimit))
	}
	return m.Next(4)
}

func parseHopLimit(cm *ControlMessage, b []byte) {
	cm.HopLimit = int(socket.NativeEndian.Uint32(b[:4]))
}

func marshalPacketInfo(b []byte, cm *ControlMessage) []byte {
	m := socket.ControlMessage(b)
	m.MarshalHeader(iana.ProtocolIPv6, sysIPV6_PKTINFO, sizeofInet6Pktinfo)
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

func parsePacketInfo(cm *ControlMessage, b []byte) {
	pi := (*inet6Pktinfo)(unsafe.Pointer(&b[0]))
	if len(cm.Dst) < net.IPv6len {
		cm.Dst = make(net.IP, net.IPv6len)
	}
	copy(cm.Dst, pi.Addr[:])
	cm.IfIndex = int(pi.Ifindex)
}

func marshalNextHop(b []byte, cm *ControlMessage) []byte {
	m := socket.ControlMessage(b)
	m.MarshalHeader(iana.ProtocolIPv6, sysIPV6_NEXTHOP, sizeofSockaddrInet6)
	if cm != nil {
		sa := (*sockaddrInet6)(unsafe.Pointer(&m.Data(sizeofSockaddrInet6)[0]))
		sa.setSockaddr(cm.NextHop, cm.IfIndex)
	}
	return m.Next(sizeofSockaddrInet6)
}

func parseNextHop(cm *ControlMessage, b []byte) {
}

func marshalPathMTU(b []byte, cm *ControlMessage) []byte {
	m := socket.ControlMessage(b)
	m.MarshalHeader(iana.ProtocolIPv6, sysIPV6_PATHMTU, sizeofIPv6Mtuinfo)
	return m.Next(sizeofIPv6Mtuinfo)
}

func parsePathMTU(cm *ControlMessage, b []byte) {
	mi := (*ipv6Mtuinfo)(unsafe.Pointer(&b[0]))
	if len(cm.Dst) < net.IPv6len {
		cm.Dst = make(net.IP, net.IPv6len)
	}
	copy(cm.Dst, mi.Addr.Addr[:])
	cm.IfIndex = int(mi.Addr.Scope_id)
	cm.MTU = int(mi.Mtu)
}
