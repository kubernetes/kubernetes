// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6

import (
	"net"
	"strconv"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/internal/socket"
)

var (
	ctlOpts = [ctlMax]ctlOpt{
		ctlHopLimit:   {sysIPV6_2292HOPLIMIT, 4, marshal2292HopLimit, parseHopLimit},
		ctlPacketInfo: {sysIPV6_2292PKTINFO, sizeofInet6Pktinfo, marshal2292PacketInfo, parsePacketInfo},
	}

	sockOpts = map[int]*sockOpt{
		ssoHopLimit:           {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_UNICAST_HOPS, Len: 4}},
		ssoMulticastInterface: {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_MULTICAST_IF, Len: 4}},
		ssoMulticastHopLimit:  {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_MULTICAST_HOPS, Len: 4}},
		ssoMulticastLoopback:  {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_MULTICAST_LOOP, Len: 4}},
		ssoReceiveHopLimit:    {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_2292HOPLIMIT, Len: 4}},
		ssoReceivePacketInfo:  {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_2292PKTINFO, Len: 4}},
		ssoChecksum:           {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_CHECKSUM, Len: 4}},
		ssoICMPFilter:         {Option: socket.Option{Level: iana.ProtocolIPv6ICMP, Name: sysICMP6_FILTER, Len: sizeofICMPv6Filter}},
		ssoJoinGroup:          {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_JOIN_GROUP, Len: sizeofIPv6Mreq}, typ: ssoTypeIPMreq},
		ssoLeaveGroup:         {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_LEAVE_GROUP, Len: sizeofIPv6Mreq}, typ: ssoTypeIPMreq},
	}
)

func init() {
	// Seems like kern.osreldate is veiled on latest OS X. We use
	// kern.osrelease instead.
	s, err := syscall.Sysctl("kern.osrelease")
	if err != nil {
		return
	}
	ss := strings.Split(s, ".")
	if len(ss) == 0 {
		return
	}
	// The IP_PKTINFO and protocol-independent multicast API were
	// introduced in OS X 10.7 (Darwin 11). But it looks like
	// those features require OS X 10.8 (Darwin 12) or above.
	// See http://support.apple.com/kb/HT1633.
	if mjver, err := strconv.Atoi(ss[0]); err != nil || mjver < 12 {
		return
	}
	ctlOpts[ctlTrafficClass] = ctlOpt{sysIPV6_TCLASS, 4, marshalTrafficClass, parseTrafficClass}
	ctlOpts[ctlHopLimit] = ctlOpt{sysIPV6_HOPLIMIT, 4, marshalHopLimit, parseHopLimit}
	ctlOpts[ctlPacketInfo] = ctlOpt{sysIPV6_PKTINFO, sizeofInet6Pktinfo, marshalPacketInfo, parsePacketInfo}
	ctlOpts[ctlNextHop] = ctlOpt{sysIPV6_NEXTHOP, sizeofSockaddrInet6, marshalNextHop, parseNextHop}
	ctlOpts[ctlPathMTU] = ctlOpt{sysIPV6_PATHMTU, sizeofIPv6Mtuinfo, marshalPathMTU, parsePathMTU}
	sockOpts[ssoTrafficClass] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_TCLASS, Len: 4}}
	sockOpts[ssoReceiveTrafficClass] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_RECVTCLASS, Len: 4}}
	sockOpts[ssoReceiveHopLimit] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_RECVHOPLIMIT, Len: 4}}
	sockOpts[ssoReceivePacketInfo] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_RECVPKTINFO, Len: 4}}
	sockOpts[ssoReceivePathMTU] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_RECVPATHMTU, Len: 4}}
	sockOpts[ssoPathMTU] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_PATHMTU, Len: sizeofIPv6Mtuinfo}}
	sockOpts[ssoJoinGroup] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysMCAST_JOIN_GROUP, Len: sizeofGroupReq}, typ: ssoTypeGroupReq}
	sockOpts[ssoLeaveGroup] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysMCAST_LEAVE_GROUP, Len: sizeofGroupReq}, typ: ssoTypeGroupReq}
	sockOpts[ssoJoinSourceGroup] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysMCAST_JOIN_SOURCE_GROUP, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq}
	sockOpts[ssoLeaveSourceGroup] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysMCAST_LEAVE_SOURCE_GROUP, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq}
	sockOpts[ssoBlockSourceGroup] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysMCAST_BLOCK_SOURCE, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq}
	sockOpts[ssoUnblockSourceGroup] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysMCAST_UNBLOCK_SOURCE, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq}
}

func (sa *sockaddrInet6) setSockaddr(ip net.IP, i int) {
	sa.Len = sizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], ip)
	sa.Scope_id = uint32(i)
}

func (pi *inet6Pktinfo) setIfindex(i int) {
	pi.Ifindex = uint32(i)
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
