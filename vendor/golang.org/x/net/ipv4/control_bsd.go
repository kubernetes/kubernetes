// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

package ipv4

import (
	"net"
	"syscall"
	"unsafe"

	"golang.org/x/net/internal/iana"
)

func marshalDst(b []byte, cm *ControlMessage) []byte {
	m := (*syscall.Cmsghdr)(unsafe.Pointer(&b[0]))
	m.Level = iana.ProtocolIP
	m.Type = sysIP_RECVDSTADDR
	m.SetLen(syscall.CmsgLen(net.IPv4len))
	return b[syscall.CmsgSpace(net.IPv4len):]
}

func parseDst(cm *ControlMessage, b []byte) {
	cm.Dst = b[:net.IPv4len]
}

func marshalInterface(b []byte, cm *ControlMessage) []byte {
	m := (*syscall.Cmsghdr)(unsafe.Pointer(&b[0]))
	m.Level = iana.ProtocolIP
	m.Type = sysIP_RECVIF
	m.SetLen(syscall.CmsgLen(syscall.SizeofSockaddrDatalink))
	return b[syscall.CmsgSpace(syscall.SizeofSockaddrDatalink):]
}

func parseInterface(cm *ControlMessage, b []byte) {
	sadl := (*syscall.SockaddrDatalink)(unsafe.Pointer(&b[0]))
	cm.IfIndex = int(sadl.Index)
}
