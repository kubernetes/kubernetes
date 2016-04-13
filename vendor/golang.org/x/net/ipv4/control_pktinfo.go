// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin linux

package ipv4

import (
	"syscall"
	"unsafe"

	"golang.org/x/net/internal/iana"
)

func marshalPacketInfo(b []byte, cm *ControlMessage) []byte {
	m := (*syscall.Cmsghdr)(unsafe.Pointer(&b[0]))
	m.Level = iana.ProtocolIP
	m.Type = sysIP_PKTINFO
	m.SetLen(syscall.CmsgLen(sysSizeofInetPktinfo))
	if cm != nil {
		pi := (*sysInetPktinfo)(unsafe.Pointer(&b[syscall.CmsgLen(0)]))
		if ip := cm.Src.To4(); ip != nil {
			copy(pi.Spec_dst[:], ip)
		}
		if cm.IfIndex > 0 {
			pi.setIfindex(cm.IfIndex)
		}
	}
	return b[syscall.CmsgSpace(sysSizeofInetPktinfo):]
}

func parsePacketInfo(cm *ControlMessage, b []byte) {
	pi := (*sysInetPktinfo)(unsafe.Pointer(&b[0]))
	cm.IfIndex = int(pi.Ifindex)
	cm.Dst = pi.Addr[:]
}
