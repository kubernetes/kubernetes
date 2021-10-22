// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4_test

import (
	"net"
	"runtime"
	"testing"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/ipv4"
	"golang.org/x/net/nettest"
)

func TestConnUnicastSocketOptions(t *testing.T) {
	switch runtime.GOOS {
	case "fuchsia", "hurd", "js", "nacl", "plan9", "windows", "zos":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if _, err := nettest.RoutedInterface("ip4", net.FlagUp|net.FlagLoopback); err != nil {
		t.Skipf("not available on %s", runtime.GOOS)
	}

	ln, err := net.Listen("tcp4", "127.0.0.1:0")
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

	c, err := net.Dial("tcp4", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	testUnicastSocketOptions(t, ipv4.NewConn(c))

	if err := <-errc; err != nil {
		t.Errorf("server: %v", err)
	}
}

var packetConnUnicastSocketOptionTests = []struct {
	net, proto, addr string
}{
	{"udp4", "", "127.0.0.1:0"},
	{"ip4", ":icmp", "127.0.0.1"},
}

func TestPacketConnUnicastSocketOptions(t *testing.T) {
	switch runtime.GOOS {
	case "fuchsia", "hurd", "js", "nacl", "plan9", "windows", "zos":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if _, err := nettest.RoutedInterface("ip4", net.FlagUp|net.FlagLoopback); err != nil {
		t.Skipf("not available on %s", runtime.GOOS)
	}

	ok := nettest.SupportsRawSocket()
	for _, tt := range packetConnUnicastSocketOptionTests {
		if tt.net == "ip4" && !ok {
			t.Logf("not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
			continue
		}
		c, err := net.ListenPacket(tt.net+tt.proto, tt.addr)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()

		testUnicastSocketOptions(t, ipv4.NewPacketConn(c))
	}
}

func TestRawConnUnicastSocketOptions(t *testing.T) {
	switch runtime.GOOS {
	case "fuchsia", "hurd", "js", "nacl", "plan9", "windows", "zos":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !nettest.SupportsRawSocket() {
		t.Skipf("not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	if _, err := nettest.RoutedInterface("ip4", net.FlagUp|net.FlagLoopback); err != nil {
		t.Skipf("not available on %s", runtime.GOOS)
	}

	c, err := net.ListenPacket("ip4:icmp", "127.0.0.1")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	r, err := ipv4.NewRawConn(c)
	if err != nil {
		t.Fatal(err)
	}

	testUnicastSocketOptions(t, r)
}

type testIPv4UnicastConn interface {
	TOS() (int, error)
	SetTOS(int) error
	TTL() (int, error)
	SetTTL(int) error
}

func testUnicastSocketOptions(t *testing.T, c testIPv4UnicastConn) {
	t.Helper()

	tos := iana.DiffServCS0 | iana.NotECNTransport
	switch runtime.GOOS {
	case "windows":
		// IP_TOS option is supported on Windows 8 and beyond.
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	if err := c.SetTOS(tos); err != nil {
		t.Fatal(err)
	}
	if v, err := c.TOS(); err != nil {
		t.Fatal(err)
	} else if v != tos {
		t.Fatalf("got %v; want %v", v, tos)
	}
	const ttl = 255
	if err := c.SetTTL(ttl); err != nil {
		t.Fatal(err)
	}
	if v, err := c.TTL(); err != nil {
		t.Fatal(err)
	} else if v != ttl {
		t.Fatalf("got %v; want %v", v, ttl)
	}
}
