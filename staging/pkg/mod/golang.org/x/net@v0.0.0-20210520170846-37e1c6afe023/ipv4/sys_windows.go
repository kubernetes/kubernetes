// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

import (
	"golang.org/x/net/internal/iana"
	"golang.org/x/net/internal/socket"

	"golang.org/x/sys/windows"
)

const (
	sizeofIPMreq       = 0x8
	sizeofIPMreqSource = 0xc
)

type ipMreq struct {
	Multiaddr [4]byte
	Interface [4]byte
}

type ipMreqSource struct {
	Multiaddr  [4]byte
	Sourceaddr [4]byte
	Interface  [4]byte
}

// See http://msdn.microsoft.com/en-us/library/windows/desktop/ms738586(v=vs.85).aspx
var (
	ctlOpts = [ctlMax]ctlOpt{}

	sockOpts = map[int]*sockOpt{
		ssoTOS:                {Option: socket.Option{Level: iana.ProtocolIP, Name: windows.IP_TOS, Len: 4}},
		ssoTTL:                {Option: socket.Option{Level: iana.ProtocolIP, Name: windows.IP_TTL, Len: 4}},
		ssoMulticastTTL:       {Option: socket.Option{Level: iana.ProtocolIP, Name: windows.IP_MULTICAST_TTL, Len: 4}},
		ssoMulticastInterface: {Option: socket.Option{Level: iana.ProtocolIP, Name: windows.IP_MULTICAST_IF, Len: 4}},
		ssoMulticastLoopback:  {Option: socket.Option{Level: iana.ProtocolIP, Name: windows.IP_MULTICAST_LOOP, Len: 4}},
		ssoHeaderPrepend:      {Option: socket.Option{Level: iana.ProtocolIP, Name: windows.IP_HDRINCL, Len: 4}},
		ssoJoinGroup:          {Option: socket.Option{Level: iana.ProtocolIP, Name: windows.IP_ADD_MEMBERSHIP, Len: sizeofIPMreq}, typ: ssoTypeIPMreq},
		ssoLeaveGroup:         {Option: socket.Option{Level: iana.ProtocolIP, Name: windows.IP_DROP_MEMBERSHIP, Len: sizeofIPMreq}, typ: ssoTypeIPMreq},
	}
)
