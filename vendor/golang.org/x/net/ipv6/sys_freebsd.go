// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6

import (
	"net"
	"runtime"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/internal/socket"
)

var (
	ctlOpts = [ctlMax]ctlOpt{
		ctlTrafficClass: {sysIPV6_TCLASS, 4, marshalTrafficClass, parseTrafficClass},
		ctlHopLimit:     {sysIPV6_HOPLIMIT, 4, marshalHopLimit, parseHopLimit},
		ctlPacketInfo:   {sysIPV6_PKTINFO, sizeofInet6Pktinfo, marshalPacketInfo, parsePacketInfo},
		ctlNextHop:      {sysIPV6_NEXTHOP, sizeofSockaddrInet6, marshalNextHop, parseNextHop},
		ctlPathMTU:      {sysIPV6_PATHMTU, sizeofIPv6Mtuinfo, marshalPathMTU, parsePathMTU},
	}

	sockOpts = map[int]sockOpt{
		ssoTrafficClass:        {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_TCLASS, Len: 4}},
		ssoHopLimit:            {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_UNICAST_HOPS, Len: 4}},
		ssoMulticastInterface:  {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_MULTICAST_IF, Len: 4}},
		ssoMulticastHopLimit:   {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_MULTICAST_HOPS, Len: 4}},
		ssoMulticastLoopback:   {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_MULTICAST_LOOP, Len: 4}},
		ssoReceiveTrafficClass: {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_RECVTCLASS, Len: 4}},
		ssoReceiveHopLimit:     {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_RECVHOPLIMIT, Len: 4}},
		ssoReceivePacketInfo:   {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_RECVPKTINFO, Len: 4}},
		ssoReceivePathMTU:      {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_RECVPATHMTU, Len: 4}},
		ssoPathMTU:             {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_PATHMTU, Len: sizeofIPv6Mtuinfo}},
		ssoChecksum:            {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysIPV6_CHECKSUM, Len: 4}},
		ssoICMPFilter:          {Option: socket.Option{Level: iana.ProtocolIPv6ICMP, Name: sysICMP6_FILTER, Len: sizeofICMPv6Filter}},
		ssoJoinGroup:           {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysMCAST_JOIN_GROUP, Len: sizeofGroupReq}, typ: ssoTypeGroupReq},
		ssoLeaveGroup:          {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysMCAST_LEAVE_GROUP, Len: sizeofGroupReq}, typ: ssoTypeGroupReq},
		ssoJoinSourceGroup:     {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysMCAST_JOIN_SOURCE_GROUP, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
		ssoLeaveSourceGroup:    {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysMCAST_LEAVE_SOURCE_GROUP, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
		ssoBlockSourceGroup:    {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysMCAST_BLOCK_SOURCE, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
		ssoUnblockSourceGroup:  {Option: socket.Option{Level: iana.ProtocolIPv6, Name: sysMCAST_UNBLOCK_SOURCE, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
	}
)

func init() {
	if runtime.GOOS == "freebsd" && runtime.GOARCH == "386" {
		archs, _ := syscall.Sysctl("kern.supported_archs")
		for _, s := range strings.Fields(archs) {
			if s == "amd64" {
				compatFreeBSD32 = true
				break
			}
		}
	}
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
	sa := (*sockaddrInet6)(unsafe.Pointer(&gr.Group))
	sa.Len = sizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], grp)
}

func (gsr *groupSourceReq) setSourceGroup(grp, src net.IP) {
	sa := (*sockaddrInet6)(unsafe.Pointer(&gsr.Group))
	sa.Len = sizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], grp)
	sa = (*sockaddrInet6)(unsafe.Pointer(&gsr.Source))
	sa.Len = sizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], src)
}
