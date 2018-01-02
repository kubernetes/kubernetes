// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6_test

import (
	"net"
	"runtime"
	"testing"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/internal/nettest"
	"golang.org/x/net/ipv6"
)

func TestConnUnicastSocketOptions(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !supportsIPv6 {
		t.Skip("ipv6 is not supported")
	}

	ln, err := net.Listen("tcp6", "[::1]:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	errc := make(chan error, 1)
	go func() {
		c, err := ln.Accept()
		if err != nil {
			errc <- err
			return
		}
		errc <- c.Close()
	}()

	c, err := net.Dial("tcp6", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	testUnicastSocketOptions(t, ipv6.NewConn(c))

	if err := <-errc; err != nil {
		t.Errorf("server: %v", err)
	}
}

var packetConnUnicastSocketOptionTests = []struct {
	net, proto, addr string
}{
	{"udp6", "", "[::1]:0"},
	{"ip6", ":ipv6-icmp", "::1"},
}

func TestPacketConnUnicastSocketOptions(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !supportsIPv6 {
		t.Skip("ipv6 is not supported")
	}

	m, ok := nettest.SupportsRawIPSocket()
	for _, tt := range packetConnUnicastSocketOptionTests {
		if tt.net == "ip6" && !ok {
			t.Log(m)
			continue
		}
		c, err := net.ListenPacket(tt.net+tt.proto, tt.addr)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()

		testUnicastSocketOptions(t, ipv6.NewPacketConn(c))
	}
}

type testIPv6UnicastConn interface {
	TrafficClass() (int, error)
	SetTrafficClass(int) error
	HopLimit() (int, error)
	SetHopLimit(int) error
}

func testUnicastSocketOptions(t *testing.T, c testIPv6UnicastConn) {
	tclass := iana.DiffServCS0 | iana.NotECNTransport
	if err := c.SetTrafficClass(tclass); err != nil {
		switch runtime.GOOS {
		case "darwin": // older darwin kernels don't support IPV6_TCLASS option
			t.Logf("not supported on %s", runtime.GOOS)
			goto next
		}
		t.Fatal(err)
	}
	if v, err := c.TrafficClass(); err != nil {
		t.Fatal(err)
	} else if v != tclass {
		t.Fatalf("got %v; want %v", v, tclass)
	}

next:
	hoplim := 255
	if err := c.SetHopLimit(hoplim); err != nil {
		t.Fatal(err)
	}
	if v, err := c.HopLimit(); err != nil {
		t.Fatal(err)
	} else if v != hoplim {
		t.Fatalf("got %v; want %v", v, hoplim)
	}
}
