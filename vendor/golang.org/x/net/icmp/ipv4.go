// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp

import (
	"net"
	"runtime"
	"unsafe"

	"golang.org/x/net/ipv4"
)

// See http://www.freebsd.org/doc/en/books/porters-handbook/freebsd-versions.html.
var freebsdVersion uint32

// ParseIPv4Header parses b as an IPv4 header of ICMP error message
// invoking packet, which is contained in ICMP error message.
func ParseIPv4Header(b []byte) (*ipv4.Header, error) {
	if len(b) < ipv4.HeaderLen {
		return nil, errHeaderTooShort
	}
	hdrlen := int(b[0]&0x0f) << 2
	if hdrlen > len(b) {
		return nil, errBufferTooShort
	}
	h := &ipv4.Header{
		Version:  int(b[0] >> 4),
		Len:      hdrlen,
		TOS:      int(b[1]),
		ID:       int(b[4])<<8 | int(b[5]),
		FragOff:  int(b[6])<<8 | int(b[7]),
		TTL:      int(b[8]),
		Protocol: int(b[9]),
		Checksum: int(b[10])<<8 | int(b[11]),
		Src:      net.IPv4(b[12], b[13], b[14], b[15]),
		Dst:      net.IPv4(b[16], b[17], b[18], b[19]),
	}
	switch runtime.GOOS {
	case "darwin":
		// TODO(mikio): fix potential misaligned memory access
		h.TotalLen = int(*(*uint16)(unsafe.Pointer(&b[2:3][0])))
	case "freebsd":
		if freebsdVersion >= 1000000 {
			h.TotalLen = int(b[2])<<8 | int(b[3])
		} else {
			// TODO(mikio): fix potential misaligned memory access
			h.TotalLen = int(*(*uint16)(unsafe.Pointer(&b[2:3][0])))
		}
	default:
		h.TotalLen = int(b[2])<<8 | int(b[3])
	}
	h.Flags = ipv4.HeaderFlags(h.FragOff&0xe000) >> 13
	h.FragOff = h.FragOff & 0x1fff
	if hdrlen-ipv4.HeaderLen > 0 {
		h.Options = make([]byte, hdrlen-ipv4.HeaderLen)
		copy(h.Options, b[ipv4.HeaderLen:])
	}
	return h, nil
}
