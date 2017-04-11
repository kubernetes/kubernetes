// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

package ipv6

import (
	"syscall"
	"unsafe"

	"golang.org/x/net/internal/iana"
)

func marshalTrafficClass(b []byte, cm *ControlMessage) []byte {
	m := (*syscall.Cmsghdr)(unsafe.Pointer(&b[0]))
	m.Level = iana.ProtocolIPv6
	m.Type = sysIPV6_TCLASS
	m.SetLen(syscall.CmsgLen(4))
	if cm != nil {
		data := b[syscall.CmsgLen(0):]
		nativeEndian.PutUint32(data[:4], uint32(cm.TrafficClass))
	}
	return b[syscall.CmsgSpace(4):]
}

func parseTrafficClass(cm *ControlMessage, b []byte) {
	cm.TrafficClass = int(nativeEndian.Uint32(b[:4]))
}

func marshalHopLimit(b []byte, cm *ControlMessage) []byte {
	m := (*syscall.Cmsghdr)(unsafe.Pointer(&b[0]))
	m.Level = iana.ProtocolIPv6
	m.Type = sysIPV6_HOPLIMIT
	m.SetLen(syscall.CmsgLen(4))
	if cm != nil {
		data := b[syscall.CmsgLen(0):]
		nativeEndian.PutUint32(data[:4], uint32(cm.HopLimit))
	}
	return b[syscall.CmsgSpace(4):]
}

func parseHopLimit(cm *ControlMessage, b []byte) {
	cm.HopLimit = int(nativeEndian.Uint32(b[:4]))
}

func marshalPacketInfo(b []byte, cm *ControlMessage) []byte {
	m := (*syscall.Cmsghdr)(unsafe.Pointer(&b[0]))
	m.Level = iana.ProtocolIPv6
	m.Type = sysIPV6_PKTINFO
	m.SetLen(syscall.CmsgLen(sysSizeofInet6Pktinfo))
	if cm != nil {
		pi := (*sysInet6Pktinfo)(unsafe.Pointer(&b[syscall.CmsgLen(0)]))
		if ip := cm.Src.To16(); ip != nil && ip.To4() == nil {
			copy(pi.Addr[:], ip)
		}
		if cm.IfIndex > 0 {
			pi.setIfindex(cm.IfIndex)
		}
	}
	return b[syscall.CmsgSpace(sysSizeofInet6Pktinfo):]
}

func parsePacketInfo(cm *ControlMessage, b []byte) {
	pi := (*sysInet6Pktinfo)(unsafe.Pointer(&b[0]))
	cm.Dst = pi.Addr[:]
	cm.IfIndex = int(pi.Ifindex)
}

func marshalNextHop(b []byte, cm *ControlMessage) []byte {
	m := (*syscall.Cmsghdr)(unsafe.Pointer(&b[0]))
	m.Level = iana.ProtocolIPv6
	m.Type = sysIPV6_NEXTHOP
	m.SetLen(syscall.CmsgLen(sysSizeofSockaddrInet6))
	if cm != nil {
		sa := (*sysSockaddrInet6)(unsafe.Pointer(&b[syscall.CmsgLen(0)]))
		sa.setSockaddr(cm.NextHop, cm.IfIndex)
	}
	return b[syscall.CmsgSpace(sysSizeofSockaddrInet6):]
}

func parseNextHop(cm *ControlMessage, b []byte) {
}

func marshalPathMTU(b []byte, cm *ControlMessage) []byte {
	m := (*syscall.Cmsghdr)(unsafe.Pointer(&b[0]))
	m.Level = iana.ProtocolIPv6
	m.Type = sysIPV6_PATHMTU
	m.SetLen(syscall.CmsgLen(sysSizeofIPv6Mtuinfo))
	return b[syscall.CmsgSpace(sysSizeofIPv6Mtuinfo):]
}

func parsePathMTU(cm *ControlMessage, b []byte) {
	mi := (*sysIPv6Mtuinfo)(unsafe.Pointer(&b[0]))
	cm.Dst = mi.Addr.Addr[:]
	cm.IfIndex = int(mi.Addr.Scope_id)
	cm.MTU = int(mi.Mtu)
}
