// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

import (
	"encoding/binary"
	"fmt"
	"net"
	"runtime"
	"syscall"
)

const (
	Version      = 4  // protocol version
	HeaderLen    = 20 // header length without extension headers
	maxHeaderLen = 60 // sensible default, revisit if later RFCs define new usage of version and header length fields
)

type HeaderFlags int

const (
	MoreFragments HeaderFlags = 1 << iota // more fragments flag
	DontFragment                          // don't fragment flag
)

// A Header represents an IPv4 header.
type Header struct {
	Version  int         // protocol version
	Len      int         // header length
	TOS      int         // type-of-service
	TotalLen int         // packet total length
	ID       int         // identification
	Flags    HeaderFlags // flags
	FragOff  int         // fragment offset
	TTL      int         // time-to-live
	Protocol int         // next protocol
	Checksum int         // checksum
	Src      net.IP      // source address
	Dst      net.IP      // destination address
	Options  []byte      // options, extension headers
}

func (h *Header) String() string {
	if h == nil {
		return "<nil>"
	}
	return fmt.Sprintf("ver=%d hdrlen=%d tos=%#x totallen=%d id=%#x flags=%#x fragoff=%#x ttl=%d proto=%d cksum=%#x src=%v dst=%v", h.Version, h.Len, h.TOS, h.TotalLen, h.ID, h.Flags, h.FragOff, h.TTL, h.Protocol, h.Checksum, h.Src, h.Dst)
}

// Marshal returns the binary encoding of the IPv4 header h.
func (h *Header) Marshal() ([]byte, error) {
	if h == nil {
		return nil, syscall.EINVAL
	}
	if h.Len < HeaderLen {
		return nil, errHeaderTooShort
	}
	hdrlen := HeaderLen + len(h.Options)
	b := make([]byte, hdrlen)
	b[0] = byte(Version<<4 | (hdrlen >> 2 & 0x0f))
	b[1] = byte(h.TOS)
	flagsAndFragOff := (h.FragOff & 0x1fff) | int(h.Flags<<13)
	switch runtime.GOOS {
	case "darwin", "dragonfly", "freebsd", "netbsd":
		nativeEndian.PutUint16(b[2:4], uint16(h.TotalLen))
		nativeEndian.PutUint16(b[6:8], uint16(flagsAndFragOff))
	default:
		binary.BigEndian.PutUint16(b[2:4], uint16(h.TotalLen))
		binary.BigEndian.PutUint16(b[6:8], uint16(flagsAndFragOff))
	}
	binary.BigEndian.PutUint16(b[4:6], uint16(h.ID))
	b[8] = byte(h.TTL)
	b[9] = byte(h.Protocol)
	binary.BigEndian.PutUint16(b[10:12], uint16(h.Checksum))
	if ip := h.Src.To4(); ip != nil {
		copy(b[12:16], ip[:net.IPv4len])
	}
	if ip := h.Dst.To4(); ip != nil {
		copy(b[16:20], ip[:net.IPv4len])
	} else {
		return nil, errMissingAddress
	}
	if len(h.Options) > 0 {
		copy(b[HeaderLen:], h.Options)
	}
	return b, nil
}

// ParseHeader parses b as an IPv4 header.
func ParseHeader(b []byte) (*Header, error) {
	if len(b) < HeaderLen {
		return nil, errHeaderTooShort
	}
	hdrlen := int(b[0]&0x0f) << 2
	if hdrlen > len(b) {
		return nil, errBufferTooShort
	}
	h := &Header{
		Version:  int(b[0] >> 4),
		Len:      hdrlen,
		TOS:      int(b[1]),
		ID:       int(binary.BigEndian.Uint16(b[4:6])),
		TTL:      int(b[8]),
		Protocol: int(b[9]),
		Checksum: int(binary.BigEndian.Uint16(b[10:12])),
		Src:      net.IPv4(b[12], b[13], b[14], b[15]),
		Dst:      net.IPv4(b[16], b[17], b[18], b[19]),
	}
	switch runtime.GOOS {
	case "darwin", "dragonfly", "netbsd":
		h.TotalLen = int(nativeEndian.Uint16(b[2:4])) + hdrlen
		h.FragOff = int(nativeEndian.Uint16(b[6:8]))
	case "freebsd":
		h.TotalLen = int(nativeEndian.Uint16(b[2:4]))
		if freebsdVersion < 1000000 {
			h.TotalLen += hdrlen
		}
		h.FragOff = int(nativeEndian.Uint16(b[6:8]))
	default:
		h.TotalLen = int(binary.BigEndian.Uint16(b[2:4]))
		h.FragOff = int(binary.BigEndian.Uint16(b[6:8]))
	}
	h.Flags = HeaderFlags(h.FragOff&0xe000) >> 13
	h.FragOff = h.FragOff & 0x1fff
	if hdrlen-HeaderLen > 0 {
		h.Options = make([]byte, hdrlen-HeaderLen)
		copy(h.Options, b[HeaderLen:])
	}
	return h, nil
}
