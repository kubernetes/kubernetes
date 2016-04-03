// Copyright 2013 The Go Authors.  All rights reserved.
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
)

type sysSockoptLen int32

var (
	ctlOpts = [ctlMax]ctlOpt{
		ctlTrafficClass: {sysIPV6_TCLASS, 4, marshalTrafficClass, parseTrafficClass},
		ctlHopLimit:     {sysIPV6_HOPLIMIT, 4, marshalHopLimit, parseHopLimit},
		ctlPacketInfo:   {sysIPV6_PKTINFO, sysSizeofInet6Pktinfo, marshalPacketInfo, parsePacketInfo},
		ctlNextHop:      {sysIPV6_NEXTHOP, sysSizeofSockaddrInet6, marshalNextHop, parseNextHop},
		ctlPathMTU:      {sysIPV6_PATHMTU, sysSizeofIPv6Mtuinfo, marshalPathMTU, parsePathMTU},
	}

	sockOpts = [ssoMax]sockOpt{
		ssoTrafficClass:        {iana.ProtocolIPv6, sysIPV6_TCLASS, ssoTypeInt},
		ssoHopLimit:            {iana.ProtocolIPv6, sysIPV6_UNICAST_HOPS, ssoTypeInt},
		ssoMulticastInterface:  {iana.ProtocolIPv6, sysIPV6_MULTICAST_IF, ssoTypeInterface},
		ssoMulticastHopLimit:   {iana.ProtocolIPv6, sysIPV6_MULTICAST_HOPS, ssoTypeInt},
		ssoMulticastLoopback:   {iana.ProtocolIPv6, sysIPV6_MULTICAST_LOOP, ssoTypeInt},
		ssoReceiveTrafficClass: {iana.ProtocolIPv6, sysIPV6_RECVTCLASS, ssoTypeInt},
		ssoReceiveHopLimit:     {iana.ProtocolIPv6, sysIPV6_RECVHOPLIMIT, ssoTypeInt},
		ssoReceivePacketInfo:   {iana.ProtocolIPv6, sysIPV6_RECVPKTINFO, ssoTypeInt},
		ssoReceivePathMTU:      {iana.ProtocolIPv6, sysIPV6_RECVPATHMTU, ssoTypeInt},
		ssoPathMTU:             {iana.ProtocolIPv6, sysIPV6_PATHMTU, ssoTypeMTUInfo},
		ssoChecksum:            {iana.ProtocolIPv6, sysIPV6_CHECKSUM, ssoTypeInt},
		ssoICMPFilter:          {iana.ProtocolIPv6ICMP, sysICMP6_FILTER, ssoTypeICMPFilter},
		ssoJoinGroup:           {iana.ProtocolIPv6, sysMCAST_JOIN_GROUP, ssoTypeGroupReq},
		ssoLeaveGroup:          {iana.ProtocolIPv6, sysMCAST_LEAVE_GROUP, ssoTypeGroupReq},
		ssoJoinSourceGroup:     {iana.ProtocolIPv6, sysMCAST_JOIN_SOURCE_GROUP, ssoTypeGroupSourceReq},
		ssoLeaveSourceGroup:    {iana.ProtocolIPv6, sysMCAST_LEAVE_SOURCE_GROUP, ssoTypeGroupSourceReq},
		ssoBlockSourceGroup:    {iana.ProtocolIPv6, sysMCAST_BLOCK_SOURCE, ssoTypeGroupSourceReq},
		ssoUnblockSourceGroup:  {iana.ProtocolIPv6, sysMCAST_UNBLOCK_SOURCE, ssoTypeGroupSourceReq},
	}
)

func init() {
	if runtime.GOOS == "freebsd" && runtime.GOARCH == "386" {
		archs, _ := syscall.Sysctl("kern.supported_archs")
		for _, s := range strings.Fields(archs) {
			if s == "amd64" {
				freebsd32o64 = true
				break
			}
		}
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
	sa := (*sysSockaddrInet6)(unsafe.Pointer(&gr.Group))
	sa.Len = sysSizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], grp)
}

func (gsr *sysGroupSourceReq) setSourceGroup(grp, src net.IP) {
	sa := (*sysSockaddrInet6)(unsafe.Pointer(&gsr.Group))
	sa.Len = sysSizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], grp)
	sa = (*sysSockaddrInet6)(unsafe.Pointer(&gsr.Source))
	sa.Len = sysSizeofSockaddrInet6
	sa.Family = syscall.AF_INET6
	copy(sa.Addr[:], src)
}
