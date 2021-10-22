// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

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
		ctlTTL:        {unix.IP_TTL, 1, marshalTTL, parseTTL},
		ctlPacketInfo: {unix.IP_PKTINFO, sizeofInetPktinfo, marshalPacketInfo, parsePacketInfo},
	}

	sockOpts = map[int]*sockOpt{
		ssoTOS:                {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_TOS, Len: 4}},
		ssoTTL:                {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_TTL, Len: 4}},
		ssoMulticastTTL:       {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_MULTICAST_TTL, Len: 4}},
		ssoMulticastInterface: {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_MULTICAST_IF, Len: unix.SizeofIPMreqn}, typ: ssoTypeIPMreqn},
		ssoMulticastLoopback:  {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_MULTICAST_LOOP, Len: 4}},
		ssoReceiveTTL:         {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_RECVTTL, Len: 4}},
		ssoPacketInfo:         {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_PKTINFO, Len: 4}},
		ssoHeaderPrepend:      {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_HDRINCL, Len: 4}},
		ssoICMPFilter:         {Option: socket.Option{Level: iana.ProtocolReserved, Name: unix.ICMP_FILTER, Len: sizeofICMPFilter}},
		ssoJoinGroup:          {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.MCAST_JOIN_GROUP, Len: sizeofGroupReq}, typ: ssoTypeGroupReq},
		ssoLeaveGroup:         {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.MCAST_LEAVE_GROUP, Len: sizeofGroupReq}, typ: ssoTypeGroupReq},
		ssoJoinSourceGroup:    {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.MCAST_JOIN_SOURCE_GROUP, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
		ssoLeaveSourceGroup:   {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.MCAST_LEAVE_SOURCE_GROUP, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
		ssoBlockSourceGroup:   {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.MCAST_BLOCK_SOURCE, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
		ssoUnblockSourceGroup: {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.MCAST_UNBLOCK_SOURCE, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
		ssoAttachFilter:       {Option: socket.Option{Level: unix.SOL_SOCKET, Name: unix.SO_ATTACH_FILTER, Len: unix.SizeofSockFprog}},
	}
)

func (pi *inetPktinfo) setIfindex(i int) {
	pi.Ifindex = int32(i)
}

func (gr *groupReq) setGroup(grp net.IP) {
	sa := (*sockaddrInet)(unsafe.Pointer(&gr.Group))
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], grp)
}

func (gsr *groupSourceReq) setSourceGroup(grp, src net.IP) {
	sa := (*sockaddrInet)(unsafe.Pointer(&gsr.Group))
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], grp)
	sa = (*sockaddrInet)(unsafe.Pointer(&gsr.Source))
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], src)
}
