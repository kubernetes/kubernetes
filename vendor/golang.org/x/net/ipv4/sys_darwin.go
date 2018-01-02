// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

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
		ctlTTL:       {sysIP_RECVTTL, 1, marshalTTL, parseTTL},
		ctlDst:       {sysIP_RECVDSTADDR, net.IPv4len, marshalDst, parseDst},
		ctlInterface: {sysIP_RECVIF, syscall.SizeofSockaddrDatalink, marshalInterface, parseInterface},
	}

	sockOpts = map[int]*sockOpt{
		ssoTOS:                {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_TOS, Len: 4}},
		ssoTTL:                {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_TTL, Len: 4}},
		ssoMulticastTTL:       {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_MULTICAST_TTL, Len: 1}},
		ssoMulticastInterface: {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_MULTICAST_IF, Len: 4}},
		ssoMulticastLoopback:  {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_MULTICAST_LOOP, Len: 4}},
		ssoReceiveTTL:         {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_RECVTTL, Len: 4}},
		ssoReceiveDst:         {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_RECVDSTADDR, Len: 4}},
		ssoReceiveInterface:   {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_RECVIF, Len: 4}},
		ssoHeaderPrepend:      {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_HDRINCL, Len: 4}},
		ssoStripHeader:        {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_STRIPHDR, Len: 4}},
		ssoJoinGroup:          {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_ADD_MEMBERSHIP, Len: sizeofIPMreq}, typ: ssoTypeIPMreq},
		ssoLeaveGroup:         {Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_DROP_MEMBERSHIP, Len: sizeofIPMreq}, typ: ssoTypeIPMreq},
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
	ctlOpts[ctlPacketInfo].name = sysIP_PKTINFO
	ctlOpts[ctlPacketInfo].length = sizeofInetPktinfo
	ctlOpts[ctlPacketInfo].marshal = marshalPacketInfo
	ctlOpts[ctlPacketInfo].parse = parsePacketInfo
	sockOpts[ssoPacketInfo] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_RECVPKTINFO, Len: 4}}
	sockOpts[ssoMulticastInterface] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_MULTICAST_IF, Len: sizeofIPMreqn}, typ: ssoTypeIPMreqn}
	sockOpts[ssoJoinGroup] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_JOIN_GROUP, Len: sizeofGroupReq}, typ: ssoTypeGroupReq}
	sockOpts[ssoLeaveGroup] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_LEAVE_GROUP, Len: sizeofGroupReq}, typ: ssoTypeGroupReq}
	sockOpts[ssoJoinSourceGroup] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_JOIN_SOURCE_GROUP, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq}
	sockOpts[ssoLeaveSourceGroup] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_LEAVE_SOURCE_GROUP, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq}
	sockOpts[ssoBlockSourceGroup] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_BLOCK_SOURCE, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq}
	sockOpts[ssoUnblockSourceGroup] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_UNBLOCK_SOURCE, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq}
}

func (pi *inetPktinfo) setIfindex(i int) {
	pi.Ifindex = uint32(i)
}

func (gr *groupReq) setGroup(grp net.IP) {
	sa := (*sockaddrInet)(unsafe.Pointer(uintptr(unsafe.Pointer(gr)) + 4))
	sa.Len = sizeofSockaddrInet
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], grp)
}

func (gsr *groupSourceReq) setSourceGroup(grp, src net.IP) {
	sa := (*sockaddrInet)(unsafe.Pointer(uintptr(unsafe.Pointer(gsr)) + 4))
	sa.Len = sizeofSockaddrInet
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], grp)
	sa = (*sockaddrInet)(unsafe.Pointer(uintptr(unsafe.Pointer(gsr)) + 132))
	sa.Len = sizeofSockaddrInet
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], src)
}
