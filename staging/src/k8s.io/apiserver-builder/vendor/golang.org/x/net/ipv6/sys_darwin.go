// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6

import (
	"net"
	"syscall"
	"unsafe"

	"golang.org/x/net/internal/iana"
)

var (
	ctlOpts = [ctlMax]ctlOpt{
		ctlHopLimit:   {sysIPV6_2292HOPLIMIT, 4, marshal2292HopLimit, parseHopLimit},
		ctlPacketInfo: {sysIPV6_2292PKTINFO, sysSizeofInet6Pktinfo, marshal2292PacketInfo, parsePacketInfo},
	}

	sockOpts = [ssoMax]sockOpt{
		ssoHopLimit:           {iana.ProtocolIPv6, sysIPV6_UNICAST_HOPS, ssoTypeInt},
		ssoMulticastInterface: {iana.ProtocolIPv6, sysIPV6_MULTICAST_IF, ssoTypeInterface},
		ssoMulticastHopLimit:  {iana.ProtocolIPv6, sysIPV6_MULTICAST_HOPS, ssoTypeInt},
		ssoMulticastLoopback:  {iana.ProtocolIPv6, sysIPV6_MULTICAST_LOOP, ssoTypeInt},
		ssoReceiveHopLimit:    {iana.ProtocolIPv6, sysIPV6_2292HOPLIMIT, ssoTypeInt},
		ssoReceivePacketInfo:  {iana.ProtocolIPv6, sysIPV6_2292PKTINFO, ssoTypeInt},
		ssoChecksum:           {iana.ProtocolIPv6, sysIPV6_CHECKSUM, ssoTypeInt},
		ssoICMPFilter:         {iana.ProtocolIPv6ICMP, sysICMP6_FILTER, ssoTypeICMPFilter},
		ssoJoinGroup:          {iana.ProtocolIPv6, sysIPV6_JOIN_GROUP, ssoTypeIPMreq},
		ssoLeaveGroup:         {iana.ProtocolIPv6, sysIPV6_LEAVE_GROUP, ssoTypeIPMreq},
	}
)

func init() {
	// Seems like kern.osreldate is veiled on latest OS X. We use
	// kern.osrelease instead.
	osver, err := syscall.Sysctl("kern.osrelease")
	if err != nil {
		return
	}
	var i int
	for i = range osver {
		if osver[i] == '.' {
			break
		}
	}
	// The IP_PKTINFO and protocol-independent multicast API were
	// introduced in OS X 10.7 (Darwin 11.0.0). But it looks like
	// those features require OS X 10.8 (Darwin 12.0.0) and above.
	// See http://support.apple.com/kb/HT1633.
	if i > 2 || i == 2 && osver[0] >= '1' && osver[1] >= '2' {
		ctlOpts[ctlTrafficClass].name = sysIPV6_TCLASS
		ctlOpts[ctlTrafficClass].length = 4
		ctlOpts[ctlTrafficClass].marshal = marshalTrafficClass
		ctlOpts[ctlTrafficClass].parse = parseTrafficClass
		ctlOpts[ctlHopLimit].name = sysIPV6_HOPLIMIT
		ctlOpts[ctlHopLimit].marshal = marshalHopLimit
		ctlOpts[ctlPacketInfo].name = sysIPV6_PKTINFO
		ctlOpts[ctlPacketInfo].marshal = marshalPacketInfo
		ctlOpts[ctlNextHop].name = sysIPV6_NEXTHOP
		ctlOpts[ctlNextHop].length = sysSizeofSockaddrInet6
		ctlOpts[ctlNextHop].marshal = marshalNextHop
		ctlOpts[ctlNextHop].parse = parseNextHop
		ctlOpts[ctlPathMTU].name = sysIPV6_PATHMTU
		ctlOpts[ctlPathMTU].length = sysSizeofIPv6Mtuinfo
		ctlOpts[ctlPathMTU].marshal = marshalPathMTU
		ctlOpts[ctlPathMTU].parse = parsePathMTU
		sockOpts[ssoTrafficClass].level = iana.ProtocolIPv6
		sockOpts[ssoTrafficClass].name = sysIPV6_TCLASS
		sockOpts[ssoTrafficClass].typ = ssoTypeInt
		sockOpts[ssoReceiveTrafficClass].level = iana.ProtocolIPv6
		sockOpts[ssoReceiveTrafficClass].name = sysIPV6_RECVTCLASS
		sockOpts[ssoReceiveTrafficClass].typ = ssoTypeInt
		sockOpts[ssoReceiveHopLimit].name = sysIPV6_RECVHOPLIMIT
		sockOpts[ssoReceivePacketInfo].name = sysIPV6_RECVPKTINFO
		sockOpts[ssoReceivePathMTU].level = iana.ProtocolIPv6
		sockOpts[ssoReceivePathMTU].name = sysIPV6_RECVPATHMTU
		sockOpts[ssoReceivePathMTU].typ = ssoTypeInt
		sockOpts[ssoPathMTU].level = iana.ProtocolIPv6
		sockOpts[ssoPathMTU].name = sysIPV6_PATHMTU
		sockOpts[ssoPathMTU].typ = ssoTypeMTUInfo
		sockOpts[ssoJoinGroup].name = sysMCAST_JOIN_GROUP
		sockOpts[ssoJoinGroup].typ = ssoTypeGroupReq
		sockOpts[ssoLeaveGroup].name = sysMCAST_LEAVE_GROUP
		sockOpts[ssoLeaveGroup].typ = ssoTypeGroupReq
		sockOpts[ssoJoinSourceGroup].level = iana.ProtocolIPv6
		sockOpts[ssoJoinSourceGroup].name = sysMCAST_JOIN_SOURCE_GROUP
		sockOpts[ssoJoinSourceGroup].typ = ssoTypeGroupSourceReq
		sockOpts[ssoLeaveSourceGroup].level = iana.ProtocolIPv6
		sockOpts[ssoLeaveSourceGroup].name = sysMCAST_LEAVE_SOURCE_GROUP
		sockOpts[ssoLeaveSourceGroup].typ = ssoTypeGroupSourceReq
		sockOpts[ssoBlockSourceGroup].level = iana.ProtocolIPv6
		sockOpts[ssoBlockSourceGroup].name = sysMCAST_BLOCK_SOURCE
		sockOpts[ssoBlockSourceGroup].typ = ssoTypeGroupSourceReq
		sockOpts[ssoUnblockSourceGroup].level = iana.ProtocolIPv6
		sockOpts[ssoUnblockSourceGroup].name = sysMCAST_UNBLOCK_SOURCE
		sockOpts[ssoUnblockSourceGroup].typ = ssoTypeGroupSourceReq
	}
}

func (sa *sysSockaddrInet6) setSockaddr(ip net.IP, i int) {
	sa.Len = sysSizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], ip)
	sa.Scope_id = uint32(i)
}

func (pi *sysInet6Pktinfo) setIfindex(i int) {
	pi.Ifindex = uint32(i)
}

func (mreq *sysIPv6Mreq) setIfindex(i int) {
	mreq.Interface = uint32(i)
}

func (gr *sysGroupReq) setGroup(grp net.IP) {
	sa := (*sysSockaddrInet6)(unsafe.Pointer(&gr.Pad_cgo_0[0]))
	sa.Len = sysSizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], grp)
}

func (gsr *sysGroupSourceReq) setSourceGroup(grp, src net.IP) {
	sa := (*sysSockaddrInet6)(unsafe.Pointer(&gsr.Pad_cgo_0[0]))
	sa.Len = sysSizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], grp)
	sa = (*sysSockaddrInet6)(unsafe.Pointer(&gsr.Pad_cgo_1[0]))
	sa.Len = sysSizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], src)
}
