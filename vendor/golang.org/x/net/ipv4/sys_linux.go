// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

import (
	"net"
	"syscall"
	"unsafe"
)

type sysSockoptLen int32

var (
	ctlOpts = [ctlMax]ctlOpt{
		ctlTTL:        {sysIP_TTL, 1, marshalTTL, parseTTL},
		ctlPacketInfo: {sysIP_PKTINFO, sysSizeofInetPktinfo, marshalPacketInfo, parsePacketInfo},
	}

	sockOpts = [ssoMax]sockOpt{
		ssoTOS:                {sysIP_TOS, ssoTypeInt},
		ssoTTL:                {sysIP_TTL, ssoTypeInt},
		ssoMulticastTTL:       {sysIP_MULTICAST_TTL, ssoTypeInt},
		ssoMulticastInterface: {sysIP_MULTICAST_IF, ssoTypeIPMreqn},
		ssoMulticastLoopback:  {sysIP_MULTICAST_LOOP, ssoTypeInt},
		ssoReceiveTTL:         {sysIP_RECVTTL, ssoTypeInt},
		ssoPacketInfo:         {sysIP_PKTINFO, ssoTypeInt},
		ssoHeaderPrepend:      {sysIP_HDRINCL, ssoTypeInt},
		ssoICMPFilter:         {sysICMP_FILTER, ssoTypeICMPFilter},
		ssoJoinGroup:          {sysMCAST_JOIN_GROUP, ssoTypeGroupReq},
		ssoLeaveGroup:         {sysMCAST_LEAVE_GROUP, ssoTypeGroupReq},
		ssoJoinSourceGroup:    {sysMCAST_JOIN_SOURCE_GROUP, ssoTypeGroupSourceReq},
		ssoLeaveSourceGroup:   {sysMCAST_LEAVE_SOURCE_GROUP, ssoTypeGroupSourceReq},
		ssoBlockSourceGroup:   {sysMCAST_BLOCK_SOURCE, ssoTypeGroupSourceReq},
		ssoUnblockSourceGroup: {sysMCAST_UNBLOCK_SOURCE, ssoTypeGroupSourceReq},
	}
)

func (pi *sysInetPktinfo) setIfindex(i int) {
	pi.Ifindex = int32(i)
}

func (gr *sysGroupReq) setGroup(grp net.IP) {
	sa := (*sysSockaddrInet)(unsafe.Pointer(&gr.Group))
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], grp)
}

func (gsr *sysGroupSourceReq) setSourceGroup(grp, src net.IP) {
	sa := (*sysSockaddrInet)(unsafe.Pointer(&gsr.Group))
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], grp)
	sa = (*sysSockaddrInet)(unsafe.Pointer(&gsr.Source))
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], src)
}
