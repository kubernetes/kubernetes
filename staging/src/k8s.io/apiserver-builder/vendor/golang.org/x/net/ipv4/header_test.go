// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

import (
	"bytes"
	"encoding/binary"
	"net"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

type headerTest struct {
	wireHeaderFromKernel          [HeaderLen]byte
	wireHeaderToKernel            [HeaderLen]byte
	wireHeaderFromTradBSDKernel   [HeaderLen]byte
	wireHeaderFromFreeBSD10Kernel [HeaderLen]byte
	wireHeaderToTradBSDKernel     [HeaderLen]byte
	*Header
}

var headerLittleEndianTest = headerTest{
	// TODO(mikio): Add platform dependent wire header formats when
	// we support new platforms.
	wireHeaderFromKernel: [HeaderLen]byte{
		0x45, 0x01, 0xbe, 0xef,
		0xca, 0xfe, 0x45, 0xdc,
		0xff, 0x01, 0xde, 0xad,
		172, 16, 254, 254,
		192, 168, 0, 1,
	},
	wireHeaderToKernel: [HeaderLen]byte{
		0x45, 0x01, 0xbe, 0xef,
		0xca, 0xfe, 0x45, 0xdc,
		0xff, 0x01, 0xde, 0xad,
		172, 16, 254, 254,
		192, 168, 0, 1,
	},
	wireHeaderFromTradBSDKernel: [HeaderLen]byte{
		0x45, 0x01, 0xdb, 0xbe,
		0xca, 0xfe, 0xdc, 0x45,
		0xff, 0x01, 0xde, 0xad,
		172, 16, 254, 254,
		192, 168, 0, 1,
	},
	wireHeaderFromFreeBSD10Kernel: [HeaderLen]byte{
		0x45, 0x01, 0xef, 0xbe,
		0xca, 0xfe, 0xdc, 0x45,
		0xff, 0x01, 0xde, 0xad,
		172, 16, 254, 254,
		192, 168, 0, 1,
	},
	wireHeaderToTradBSDKernel: [HeaderLen]byte{
		0x45, 0x01, 0xef, 0xbe,
		0xca, 0xfe, 0xdc, 0x45,
		0xff, 0x01, 0xde, 0xad,
		172, 16, 254, 254,
		192, 168, 0, 1,
	},
	Header: &Header{
		Version:  Version,
		Len:      HeaderLen,
		TOS:      1,
		TotalLen: 0xbeef,
		ID:       0xcafe,
		Flags:    DontFragment,
		FragOff:  1500,
		TTL:      255,
		Protocol: 1,
		Checksum: 0xdead,
		Src:      net.IPv4(172, 16, 254, 254),
		Dst:      net.IPv4(192, 168, 0, 1),
	},
}

func TestMarshalHeader(t *testing.T) {
	tt := &headerLittleEndianTest
	if nativeEndian != binary.LittleEndian {
		t.Skip("no test for non-little endian machine yet")
	}

	b, err := tt.Header.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	var wh []byte
	switch runtime.GOOS {
	case "darwin", "dragonfly", "netbsd":
		wh = tt.wireHeaderToTradBSDKernel[:]
	case "freebsd":
		if freebsdVersion < 1000000 {
			wh = tt.wireHeaderToTradBSDKernel[:]
		} else {
			wh = tt.wireHeaderFromFreeBSD10Kernel[:]
		}
	default:
		wh = tt.wireHeaderToKernel[:]
	}
	if !bytes.Equal(b, wh) {
		t.Fatalf("got %#v; want %#v", b, wh)
	}
}

func TestParseHeader(t *testing.T) {
	tt := &headerLittleEndianTest
	if nativeEndian != binary.LittleEndian {
		t.Skip("no test for big endian machine yet")
	}

	var wh []byte
	switch runtime.GOOS {
	case "darwin", "dragonfly", "netbsd":
		wh = tt.wireHeaderFromTradBSDKernel[:]
	case "freebsd":
		if freebsdVersion < 1000000 {
			wh = tt.wireHeaderFromTradBSDKernel[:]
		} else {
			wh = tt.wireHeaderFromFreeBSD10Kernel[:]
		}
	default:
		wh = tt.wireHeaderFromKernel[:]
	}
	h, err := ParseHeader(wh)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(h, tt.Header) {
		t.Fatalf("got %#v; want %#v", h, tt.Header)
	}
	s := h.String()
	if strings.Contains(s, ",") {
		t.Fatalf("should be space-separated values: %s", s)
	}
}
