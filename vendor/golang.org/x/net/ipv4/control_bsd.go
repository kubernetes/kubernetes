// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || netbsd || openbsd
// +build aix darwin dragonfly freebsd netbsd openbsd

package ipv4

import (
	"net"
	"syscall"
	"unsafe"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/internal/socket"
)

func marshalDst(b []byte, cm *ControlMessage) []byte {
	m := socket.ControlMessage(b)
	m.MarshalHeader(iana.ProtocolIP, sysIP_RECVDSTADDR, net.IPv4len)
	return m.Next(net.IPv4len)
}

func parseDst(cm *ControlMessage, b []byte) {
	if len(cm.Dst) < net.IPv4len {
		cm.Dst = make(net.IP, net.IPv4len)
	}
	copy(cm.Dst, b[:net.IPv4len])
}

func marshalInterface(b []byte, cm *ControlMessage) []byte {
	m := socket.ControlMessage(b)
	m.MarshalHeader(iana.ProtocolIP, sysIP_RECVIF, syscall.SizeofSockaddrDatalink)
	return m.Next(syscall.SizeofSockaddrDatalink)
}

func parseInterface(cm *ControlMessage, b []byte) {
	var sadl syscall.SockaddrDatalink
	copy((*[unsafe.Sizeof(sadl)]byte)(unsafe.Pointer(&sadl))[:], b)
	cm.IfIndex = int(sadl.Index)
}
