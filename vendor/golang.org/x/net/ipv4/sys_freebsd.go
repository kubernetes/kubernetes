// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

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
		ssoJoinGroup:          {Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_JOIN_GROUP, Len: sizeofGroupReq}, typ: ssoTypeGroupReq},
		ssoLeaveGroup:         {Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_LEAVE_GROUP, Len: sizeofGroupReq}, typ: ssoTypeGroupReq},
		ssoJoinSourceGroup:    {Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_JOIN_SOURCE_GROUP, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
		ssoLeaveSourceGroup:   {Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_LEAVE_SOURCE_GROUP, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
		ssoBlockSourceGroup:   {Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_BLOCK_SOURCE, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
		ssoUnblockSourceGroup: {Option: socket.Option{Level: iana.ProtocolIP, Name: sysMCAST_UNBLOCK_SOURCE, Len: sizeofGroupSourceReq}, typ: ssoTypeGroupSourceReq},
	}
)

func init() {
	freebsdVersion, _ = syscall.SysctlUint32("kern.osreldate")
	if freebsdVersion >= 1000000 {
		sockOpts[ssoMulticastInterface] = &sockOpt{Option: socket.Option{Level: iana.ProtocolIP, Name: sysIP_MULTICAST_IF, Len: sizeofIPMreqn}, typ: ssoTypeIPMreqn}
	}
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

func (gr *groupReq) setGroup(grp net.IP) {
	sa := (*sockaddrInet)(unsafe.Pointer(&gr.Group))
	sa.Len = sizeofSockaddrInet
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], grp)
}

func (gsr *groupSourceReq) setSourceGroup(grp, src net.IP) {
	sa := (*sockaddrInet)(unsafe.Pointer(&gsr.Group))
	sa.Len = sizeofSockaddrInet
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], grp)
	sa = (*sockaddrInet)(unsafe.Pointer(&gsr.Source))
	sa.Len = sizeofSockaddrInet
	sa.Family = syscall.AF_INET
	copy(sa.Addr[:], src)
}
