// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

import (
	"bytes"
	"net"
	"reflect"
	"runtime"
	"testing"
)

var (
	wireHeaderFromKernel = [HeaderLen]byte{
		0x45, 0x01, 0xbe, 0xef,
		0xca, 0xfe, 0x45, 0xdc,
		0xff, 0x01, 0xde, 0xad,
		172, 16, 254, 254,
		192, 168, 0, 1,
	}
	wireHeaderToKernel = [HeaderLen]byte{
		0x45, 0x01, 0xbe, 0xef,
		0xca, 0xfe, 0x45, 0xdc,
		0xff, 0x01, 0xde, 0xad,
		172, 16, 254, 254,
		192, 168, 0, 1,
	}
	wireHeaderFromTradBSDKernel = [HeaderLen]byte{
		0x45, 0x01, 0xdb, 0xbe,
		0xca, 0xfe, 0xdc, 0x45,
		0xff, 0x01, 0xde, 0xad,
		172, 16, 254, 254,
		192, 168, 0, 1,
	}
	wireHeaderFromFreeBSD10Kernel = [HeaderLen]byte{
		0x45, 0x01, 0xef, 0xbe,
		0xca, 0xfe, 0xdc, 0x45,
		0xff, 0x01, 0xde, 0xad,
		172, 16, 254, 254,
		192, 168, 0, 1,
	}
	wireHeaderToTradBSDKernel = [HeaderLen]byte{
		0x45, 0x01, 0xef, 0xbe,
		0xca, 0xfe, 0xdc, 0x45,
		0xff, 0x01, 0xde, 0xad,
		172, 16, 254, 254,
		192, 168, 0, 1,
	}
	// TODO(mikio): Add platform dependent wire header formats when
	// we support new platforms.

	testHeader = &Header{
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
	}
)

func TestMarshalHeader(t *testing.T) {
	b, err := testHeader.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	var wh []byte
	switch runtime.GOOS {
	case "darwin", "dragonfly", "netbsd":
		wh = wireHeaderToTradBSDKernel[:]
	case "freebsd":
		if freebsdVersion < 1000000 {
			wh = wireHeaderToTradBSDKernel[:]
		} else {
			wh = wireHeaderFromFreeBSD10Kernel[:]
		}
	default:
		wh = wireHeaderToKernel[:]
	}
	if !bytes.Equal(b, wh) {
		t.Fatalf("got %#v; want %#v", b, wh)
	}
}

func TestParseHeader(t *testing.T) {
	var wh []byte
	switch runtime.GOOS {
	case "darwin", "dragonfly", "netbsd":
		wh = wireHeaderFromTradBSDKernel[:]
	case "freebsd":
		if freebsdVersion < 1000000 {
			wh = wireHeaderFromTradBSDKernel[:]
		} else {
			wh = wireHeaderFromFreeBSD10Kernel[:]
		}
	default:
		wh = wireHeaderFromKernel[:]
	}
	h, err := ParseHeader(wh)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(h, testHeader) {
		t.Fatalf("got %#v; want %#v", h, testHeader)
	}
}
