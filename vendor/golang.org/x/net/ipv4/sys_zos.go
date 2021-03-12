// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

import (
	"net"
	"syscall"
	"unsafe"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/internal/socket"
)

var (
	ctlOpts = [ctlMax]ctlOpt{
		ctlPacketInfo: {sysIP_PKTINFO, sizeofInetPktinfo, marshalPacketInfo, parsePacketInfo},
	}

	sockOpts = map[int]*sockOpt{
		ssoMulticastTTL:       {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_MULTICAST_TTL, Len: 1}},
		ssoMulticastInterface: {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_MULTICAST_IF, Len: 4}},
		ssoMulticastLoopback:  {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_MULTICAST_LOOP, Len: 1}},
		ssoPacketInfo:         {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_RECVPKTINFO, Len: 4}},
		ssoJoinGroup:          {Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_JOIN_GROUP, Len: sizeofGroupReq}, typ: ssoTypeGroupReq},
		ssoLeaveGroup:         {Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_LEAVE_GROUP, Len: sizeofGroupReq}, typ: ssoTypeGroupReq},
		ssoJoinSourceGroup:    {Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_JOIN_SOURCE_GROUP, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
		ssoLeaveSourceGroup:   {Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_LEAVE_SOURCE_GROUP, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
		ssoBlockSourceGroup:   {Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_BLOCK_SOURCE, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
		ssoUnblockSourceGroup: {Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_UNBLOCK_SOURCE, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
	}
)

func (pi *inetPktinfo) setIfindex(i int) {
	pi.Ifindex = uint32(i)
}

func (gr *groupReq) setGroup(grp net.IP) {
	sa := (*sockaddrInet4)(unsafe.Pointer(&gr.Group))
	sa.Family = syscall.AF_INET
	sa.Len = sizeofSockaddrInet4
	copy(sa.Addr[:], grp)
}

func (gsr *groupSourceReq) setSourceGroup(grp, src net.IP) {
	sa := (*sockaddrInet4)(unsafe.Pointer(&gsr.Group))
	sa.Family = syscall.AF_INET
	sa.Len = sizeofSockaddrInet4
	copy(sa.Addr[:], grp)
	sa = (*sockaddrInet4)(unsafe.Pointer(&gsr.Source))
	sa.Family = syscall.AF_INET
	sa.Len = sizeofSockaddrInet4
	copy(sa.Addr[:], src)
}
