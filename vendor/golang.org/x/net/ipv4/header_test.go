// Copyright 2012 The Go Authors. All rights reserved.
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

	"golang.org/x/net/internal/socket"
)

type headerTest struct {
	wireHeaderFromKernel          []byte
	wireHeaderToKernel            []byte
	wireHeaderFromTradBSDKernel   []byte
	wireHeaderToTradBSDKernel     []byte
	wireHeaderFromFreeBSD10Kernel []byte
	wireHeaderToFreeBSD10Kernel   []byte
	*Header
}

var headerLittleEndianTests = []headerTest{
	// TODO(mikio): Add platform dependent wire header formats when
	// we support new platforms.
	{
		wireHeaderFromKernel: []byte{
			0x45, 0x01, 0xbe, 0xef,
			0xca, 0xfe, 0x45, 0xdc,
			0xff, 0x01, 0xde, 0xad,
			172, 16, 254, 254,
			192, 168, 0, 1,
		},
		wireHeaderToKernel: []byte{
			0x45, 0x01, 0xbe, 0xef,
			0xca, 0xfe, 0x45, 0xdc,
			0xff, 0x01, 0xde, 0xad,
			172, 16, 254, 254,
			192, 168, 0, 1,
		},
		wireHeaderFromTradBSDKernel: []byte{
			0x45, 0x01, 0xdb, 0xbe,
			0xca, 0xfe, 0xdc, 0x45,
			0xff, 0x01, 0xde, 0xad,
			172, 16, 254, 254,
			192, 168, 0, 1,
		},
		wireHeaderToTradBSDKernel: []byte{
			0x45, 0x01, 0xef, 0xbe,
			0xca, 0xfe, 0xdc, 0x45,
			0xff, 0x01, 0xde, 0xad,
			172, 16, 254, 254,
			192, 168, 0, 1,
		},
		wireHeaderFromFreeBSD10Kernel: []byte{
			0x45, 0x01, 0xef, 0xbe,
			0xca, 0xfe, 0xdc, 0x45,
			0xff, 0x01, 0xde, 0xad,
			172, 16, 254, 254,
			192, 168, 0, 1,
		},
		wireHeaderToFreeBSD10Kernel: []byte{
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
	},

	// with option headers
	{
		wireHeaderFromKernel: []byte{
			0x46, 0x01, 0xbe, 0xf3,
			0xca, 0xfe, 0x45, 0xdc,
			0xff, 0x01, 0xde, 0xad,
			172, 16, 254, 254,
			192, 168, 0, 1,
			0xff, 0xfe, 0xfe, 0xff,
		},
		wireHeaderToKernel: []byte{
			0x46, 0x01, 0xbe, 0xf3,
			0xca, 0xfe, 0x45, 0xdc,
			0xff, 0x01, 0xde, 0xad,
			172, 16, 254, 254,
			192, 168, 0, 1,
			0xff, 0xfe, 0xfe, 0xff,
		},
		wireHeaderFromTradBSDKernel: []byte{
			0x46, 0x01, 0xdb, 0xbe,
			0xca, 0xfe, 0xdc, 0x45,
			0xff, 0x01, 0xde, 0xad,
			172, 16, 254, 254,
			192, 168, 0, 1,
			0xff, 0xfe, 0xfe, 0xff,
		},
		wireHeaderToTradBSDKernel: []byte{
			0x46, 0x01, 0xf3, 0xbe,
			0xca, 0xfe, 0xdc, 0x45,
			0xff, 0x01, 0xde, 0xad,
			172, 16, 254, 254,
			192, 168, 0, 1,
			0xff, 0xfe, 0xfe, 0xff,
		},
		wireHeaderFromFreeBSD10Kernel: []byte{
			0x46, 0x01, 0xf3, 0xbe,
			0xca, 0xfe, 0xdc, 0x45,
			0xff, 0x01, 0xde, 0xad,
			172, 16, 254, 254,
			192, 168, 0, 1,
			0xff, 0xfe, 0xfe, 0xff,
		},
		wireHeaderToFreeBSD10Kernel: []byte{
			0x46, 0x01, 0xf3, 0xbe,
			0xca, 0xfe, 0xdc, 0x45,
			0xff, 0x01, 0xde, 0xad,
			172, 16, 254, 254,
			192, 168, 0, 1,
			0xff, 0xfe, 0xfe, 0xff,
		},
		Header: &Header{
			Version:  Version,
			Len:      HeaderLen + 4,
			TOS:      1,
			TotalLen: 0xbef3,
			ID:       0xcafe,
			Flags:    DontFragment,
			FragOff:  1500,
			TTL:      255,
			Protocol: 1,
			Checksum: 0xdead,
			Src:      net.IPv4(172, 16, 254, 254),
			Dst:      net.IPv4(192, 168, 0, 1),
			Options:  []byte{0xff, 0xfe, 0xfe, 0xff},
		},
	},
}

func TestMarshalHeader(t *testing.T) {
	if socket.NativeEndian != binary.LittleEndian {
		t.Skip("no test for non-little endian machine yet")
	}

	for _, tt := range headerLittleEndianTests {
		b, err := tt.Header.Marshal()
		if err != nil {
			t.Fatal(err)
		}
		var wh []byte
		switch runtime.GOOS {
		case "darwin", "dragonfly", "netbsd":
			wh = tt.wireHeaderToTradBSDKernel
		case "freebsd":
			switch {
			case freebsdVersion < 1000000:
				wh = tt.wireHeaderToTradBSDKernel
			case 1000000 <= freebsdVersion && freebsdVersion < 1100000:
				wh = tt.wireHeaderToFreeBSD10Kernel
			default:
				wh = tt.wireHeaderToKernel
			}
		default:
			wh = tt.wireHeaderToKernel
		}
		if !bytes.Equal(b, wh) {
			t.Fatalf("got %#v; want %#v", b, wh)
		}
	}
}

func TestParseHeader(t *testing.T) {
	if socket.NativeEndian != binary.LittleEndian {
		t.Skip("no test for big endian machine yet")
	}

	for _, tt := range headerLittleEndianTests {
		var wh []byte
		switch runtime.GOOS {
		case "darwin", "dragonfly", "netbsd":
			wh = tt.wireHeaderFromTradBSDKernel
		case "freebsd":
			switch {
			case freebsdVersion < 1000000:
				wh = tt.wireHeaderFromTradBSDKernel
			case 1000000 <= freebsdVersion && freebsdVersion < 1100000:
				wh = tt.wireHeaderFromFreeBSD10Kernel
			default:
				wh = tt.wireHeaderFromKernel
			}
		default:
			wh = tt.wireHeaderFromKernel
		}
		h, err := ParseHeader(wh)
		if err != nil {
			t.Fatal(err)
		}
		if err := h.Parse(wh); err != nil {
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
}
