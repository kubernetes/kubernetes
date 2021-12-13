// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Added for go1.11 compatibility
//go:build aix
// +build aix

package ipv6

import (
	"net"
	"syscall"
	"unsafe"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/internal/socket"

	"golang.org/x/sys/unix"
)

var (
	ctlOpts = [ctlMax]ctlOpt{
		ctlTrafficClass: {unix.IPV6_TCLASS, 4, marshalTrafficClass, parseTrafficClass},
		ctlHopLimit:     {unix.IPV6_HOPLIMIT, 4, marshalHopLimit, parseHopLimit},
		ctlPacketInfo:   {unix.IPV6_PKTINFO, sizeofInet6Pktinfo, marshalPacketInfo, parsePacketInfo},
		ctlNextHop:      {unix.IPV6_NEXTHOP, sizeofSockaddrInet6, marshalNextHop, parseNextHop},
		ctlPathMTU:      {unix.IPV6_PATHMTU, sizeofIPv6Mtuinfo, marshalPathMTU, parsePathMTU},
	}

	sockOpts = map[int]*sockOpt{
		ssoTrafficClass:        {Option: socket.Option{Level: iana.ProtocolIPv6, Name: unix.IPV6_TCLASS, Len: 4}},
		ssoHopLimit:            {Option: socket.Option{Level: iana.ProtocolIPv6, Name: unix.IPV6_UNICAST_HOPS, Len: 4}},
		ssoMulticastInterface:  {Option: socket.Option{Level: iana.ProtocolIPv6, Name: unix.IPV6_MULTICAST_IF, Len: 4}},
		ssoMulticastHopLimit:   {Option: socket.Option{Level: iana.ProtocolIPv6, Name: unix.IPV6_MULTICAST_HOPS, Len: 4}},
		ssoMulticastLoopback:   {Option: socket.Option{Level: iana.ProtocolIPv6, Name: unix.IPV6_MULTICAST_LOOP, Len: 4}},
		ssoReceiveTrafficClass: {Option: socket.Option{Level: iana.ProtocolIPv6, Name: unix.IPV6_RECVTCLASS, Len: 4}},
		ssoReceiveHopLimit:     {Option: socket.Option{Level: iana.ProtocolIPv6, Name: unix.IPV6_RECVHOPLIMIT, Len: 4}},
		ssoReceivePacketInfo:   {Option: socket.Option{Level: iana.ProtocolIPv6, Name: unix.IPV6_RECVPKTINFO, Len: 4}},
		ssoReceivePathMTU:      {Option: socket.Option{Level: iana.ProtocolIPv6, Name: unix.IPV6_RECVPATHMTU, Len: 4}},
		ssoPathMTU:             {Option: socket.Option{Level: iana.ProtocolIPv6, Name: unix.IPV6_PATHMTU, Len: sizeofIPv6Mtuinfo}},
		ssoChecksum:            {Option: socket.Option{Level: iana.ProtocolIPv6, Name: unix.IPV6_CHECKSUM, Len: 4}},
		ssoICMPFilter:          {Option: socket.Option{Level: iana.ProtocolIPv6ICMP, Name: unix.ICMP6_FILTER, Len: sizeofICMPv6Filter}},
		ssoJoinGroup:           {Option: socket.Option{Level: iana.ProtocolIPv6, Name: unix.IPV6_JOIN_GROUP, Len: sizeofIPv6Mreq}, typ: ssoTypeIPMreq},
		ssoLeaveGroup:          {Option: socket.Option{Level: iana.ProtocolIPv6, Name: unix.IPV6_LEAVE_GROUP, Len: sizeofIPv6Mreq}, typ: ssoTypeIPMreq},
	}
)

func (sa *sockaddrInet6) setSockaddr(ip net.IP, i int) {
	sa.Len = sizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], ip)
	sa.Scope_id = uint32(i)
}

func (pi *inet6Pktinfo) setIfindex(i int) {
	pi.Ifindex = int32(i)
}

func (mreq *ipv6Mreq) setIfindex(i int) {
	mreq.Interface = uint32(i)
}

func (gr *groupReq) setGroup(grp net.IP) {
	sa := (*sockaddrInet6)(unsafe.Pointer(uintptr(unsafe.Pointer(gr)) + 4))
	sa.Len = sizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], grp)
}

func (gsr *groupSourceReq) setSourceGroup(grp, src net.IP) {
	sa := (*sockaddrInet6)(unsafe.Pointer(uintptr(unsafe.Pointer(gsr)) + 4))
	sa.Len = sizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], grp)
	sa = (*sockaddrInet6)(unsafe.Pointer(uintptr(unsafe.Pointer(gsr)) + 132))
	sa.Len = sizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], src)
}
