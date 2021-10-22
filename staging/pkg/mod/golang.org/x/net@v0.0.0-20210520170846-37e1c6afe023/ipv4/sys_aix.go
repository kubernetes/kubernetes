// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Added for go1.11 compatibility
//go:build aix
// +build aix

package ipv4

import (
	"net"
	"syscall"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/internal/socket"

	"golang.org/x/sys/unix"
)

// IP_RECVIF is defined on AIX but doesn't work. IP_RECVINTERFACE must be used instead.
const sockoptReceiveInterface = unix.IP_RECVINTERFACE

var (
	ctlOpts = [ctlMax]ctlOpt{
		ctlTTL:       {unix.IP_RECVTTL, 1, marshalTTL, parseTTL},
		ctlDst:       {unix.IP_RECVDSTADDR, net.IPv4len, marshalDst, parseDst},
		ctlInterface: {unix.IP_RECVINTERFACE, syscall.SizeofSockaddrDatalink, marshalInterface, parseInterface},
	}

	sockOpts = map[int]*sockOpt{
		ssoTOS:                {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_TOS, Len: 4}},
		ssoTTL:                {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_TTL, Len: 4}},
		ssoMulticastTTL:       {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_MULTICAST_TTL, Len: 1}},
		ssoMulticastInterface: {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_MULTICAST_IF, Len: 4}},
		ssoMulticastLoopback:  {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_MULTICAST_LOOP, Len: 1}},
		ssoReceiveTTL:         {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_RECVTTL, Len: 4}},
		ssoReceiveDst:         {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_RECVDSTADDR, Len: 4}},
		ssoReceiveInterface:   {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_RECVINTERFACE, Len: 4}},
		ssoHeaderPrepend:      {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_HDRINCL, Len: 4}},
		ssoJoinGroup:          {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_ADD_MEMBERSHIP, Len: sizeofIPMreq}, typ: ssoTypeIPMreq},
		ssoLeaveGroup:         {Option: socket.Option{Level: iana.ProtocolIP, Name: unix.IP_DROP_MEMBERSHIP, Len: sizeofIPMreq}, typ: ssoTypeIPMreq},
	}
)
