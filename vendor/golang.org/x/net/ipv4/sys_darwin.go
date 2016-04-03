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
		ctlTTL:       {sysIP_RECVTTL, 1, marshalTTL, parseTTL},
		ctlDst:       {sysIP_RECVDSTADDR, net.IPv4len, marshalDst, parseDst},
		ctlInterface: {sysIP_RECVIF, syscall.SizeofSockaddrDatalink, marshalInterface, parseInterface},
	}

	sockOpts = [ssoMax]sockOpt{
		ssoTOS:                {sysIP_TOS, ssoTypeInt},
		ssoTTL:                {sysIP_TTL, ssoTypeInt},
		ssoMulticastTTL:       {sysIP_MULTICAST_TTL, ssoTypeByte},
		ssoMulticastInterface: {sysIP_MULTICAST_IF, ssoTypeInterface},
		ssoMulticastLoopback:  {sysIP_MULTICAST_LOOP, ssoTypeInt},
		ssoReceiveTTL:         {sysIP_RECVTTL, ssoTypeInt},
		ssoReceiveDst:         {sysIP_RECVDSTADDR, ssoTypeInt},
		ssoReceiveInterface:   {sysIP_RECVIF, ssoTypeInt},
		ssoHeaderPrepend:      {sysIP_HDRINCL, ssoTypeInt},
		ssoStripHeader:        {sysIP_STRIPHDR, ssoTypeInt},
		ssoJoinGroup:          {sysIP_ADD_MEMBERSHIP, ssoTypeIPMreq},
		ssoLeaveGroup:         {sysIP_DROP_MEMBERSHIP, ssoTypeIPMreq},
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
		ctlOpts[ctlPacketInfo].name = sysIP_PKTINFO
		ctlOpts[ctlPacketInfo].length = sysSizeofInetPktinfo
		ctlOpts[ctlPacketInfo].marshal = marshalPacketInfo
		ctlOpts[ctlPacketInfo].parse = parsePacketInfo
		sockOpts[ssoPacketInfo].name = sysIP_RECVPKTINFO
		sockOpts[ssoPacketInfo].typ = ssoTypeInt
		sockOpts[ssoMulticastInterface].typ = ssoTypeIPMreqn
		sockOpts[ssoJoinGroup].name = sysMCAST_JOIN_GROUP
		sockOpts[ssoJoinGroup].typ = ssoTypeGroupReq
		sockOpts[ssoLeaveGroup].name = sysMCAST_LEAVE_GROUP
		sockOpts[ssoLeaveGroup].typ = ssoTypeGroupReq
		sockOpts[ssoJoinSourceGroup].name = sysMCAST_JOIN_SOURCE_GROUP
		sockOpts[ssoJoinSourceGroup].typ = ssoTypeGroupSourceReq
		sockOpts[ssoLeaveSourceGroup].name = sysMCAST_LEAVE_SOURCE_GROUP
		sockOpts[ssoLeaveSourceGroup].typ = ssoTypeGroupSourceReq
		sockOpts[ssoBlockSourceGroup].name = sysMCAST_BLOCK_SOURCE
		sockOpts[ssoBlockSourceGroup].typ = ssoTypeGroupSourceReq
		sockOpts[ssoUnblockSourceGroup].name = sysMCAST_UNBLOCK_SOURCE
		sockOpts[ssoUnblockSourceGroup].typ = ssoTypeGroupSourceReq
	}
}

func (pi *sysInetPktinfo) setIfindex(i int) {
	pi.Ifindex = uint32(i)
}

func (gr *sysGroupReq) setGroup(grp net.IP) {
	sa := (*sysSockaddrInet)(unsafe.Pointer(&gr.Pad_cgo_0[0]))
	sa.Len = sysSizeofSockaddrInet
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], grp)
}

func (gsr *sysGroupSourceReq) setSourceGroup(grp, src net.IP) {
	sa := (*sysSockaddrInet)(unsafe.Pointer(&gsr.Pad_cgo_0[0]))
	sa.Len = sysSizeofSockaddrInet
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], grp)
	sa = (*sysSockaddrInet)(unsafe.Pointer(&gsr.Pad_cgo_1[0]))
	sa.Len = sysSizeofSockaddrInet
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], src)
}
