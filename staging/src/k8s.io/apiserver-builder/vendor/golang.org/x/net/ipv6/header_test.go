// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6_test

import (
	"net"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/ipv6"
)

var (
	wireHeaderFromKernel = [ipv6.HeaderLen]byte{
		0x69, 0x8b, 0xee, 0xf1,
		0xca, 0xfe, 0x2c, 0x01,
		0x20, 0x01, 0x0d, 0xb8,
		0x00, 0x01, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x01,
		0x20, 0x01, 0x0d, 0xb8,
		0x00, 0x02, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x01,
	}

	testHeader = &ipv6.Header{
		Version:      ipv6.Version,
		TrafficClass: iana.DiffServAF43,
		FlowLabel:    0xbeef1,
		PayloadLen:   0xcafe,
		NextHeader:   iana.ProtocolIPv6Frag,
		HopLimit:     1,
		Src:          net.ParseIP("2001:db8:1::1"),
		Dst:          net.ParseIP("2001:db8:2::1"),
	}
)

func TestParseHeader(t *testing.T) {
	h, err := ipv6.ParseHeader(wireHeaderFromKernel[:])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(h, testHeader) {
		t.Fatalf("got %#v; want %#v", h, testHeader)
	}
	s := h.String()
	if strings.Contains(s, ",") {
		t.Fatalf("should be space-separated values: %s", s)
	}
}
