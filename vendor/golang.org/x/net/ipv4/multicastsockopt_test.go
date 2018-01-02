// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4_test

import (
	"net"
	"runtime"
	"testing"

	"golang.org/x/net/internal/nettest"
	"golang.org/x/net/ipv4"
)

var packetConnMulticastSocketOptionTests = []struct {
	net, proto, addr string
	grp, src         net.Addr
}{
	{"udp4", "", "224.0.0.0:0", &net.UDPAddr{IP: net.IPv4(224, 0, 0, 249)}, nil}, // see RFC 4727
	{"ip4", ":icmp", "0.0.0.0", &net.IPAddr{IP: net.IPv4(224, 0, 0, 250)}, nil},  // see RFC 4727

	{"udp4", "", "232.0.0.0:0", &net.UDPAddr{IP: net.IPv4(232, 0, 1, 249)}, &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)}}, // see RFC 5771
	{"ip4", ":icmp", "0.0.0.0", &net.IPAddr{IP: net.IPv4(232, 0, 1, 250)}, &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)}},  // see RFC 5771
}

func TestPacketConnMulticastSocketOptions(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	ifi := nettest.RoutedInterface("ip4", net.FlagUp|net.FlagMulticast|net.FlagLoopback)
	if ifi == nil {
		t.Skipf("not available on %s", runtime.GOOS)
	}

	m, ok := nettest.SupportsRawIPSocket()
	for _, tt := range packetConnMulticastSocketOptionTests {
		if tt.net == "ip4" && !ok {
			t.Log(m)
			continue
		}
		c, err := net.ListenPacket(tt.net+tt.proto, tt.addr)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		p := ipv4.NewPacketConn(c)
		defer p.Close()

		if tt.src == nil {
			testMulticastSocketOptions(t, p, ifi, tt.grp)
		} else {
			testSourceSpecificMulticastSocketOptions(t, p, ifi, tt.grp, tt.src)
		}
	}
}

var rawConnMulticastSocketOptionTests = []struct {
	grp, src net.Addr
}{
	{&net.IPAddr{IP: net.IPv4(224, 0, 0, 250)}, nil}, // see RFC 4727

	{&net.IPAddr{IP: net.IPv4(232, 0, 1, 250)}, &net.IPAddr{IP: net.IPv4(127, 0, 0, 1)}}, // see RFC 5771
}

func TestRawConnMulticastSocketOptions(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if m, ok := nettest.SupportsRawIPSocket(); !ok {
		t.Skip(m)
	}
	ifi := nettest.RoutedInterface("ip4", net.FlagUp|net.FlagMulticast|net.FlagLoopback)
	if ifi == nil {
		t.Skipf("not available on %s", runtime.GOOS)
	}

	for _, tt := range rawConnMulticastSocketOptionTests {
		c, err := net.ListenPacket("ip4:icmp", "0.0.0.0")
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		r, err := ipv4.NewRawConn(c)
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()

		if tt.src == nil {
			testMulticastSocketOptions(t, r, ifi, tt.grp)
		} else {
			testSourceSpecificMulticastSocketOptions(t, r, ifi, tt.grp, tt.src)
		}
	}
}

type testIPv4MulticastConn interface {
	MulticastTTL() (int, error)
	SetMulticastTTL(ttl int) error
	MulticastLoopback() (bool, error)
	SetMulticastLoopback(bool) error
	JoinGroup(*net.Interface, net.Addr) error
	LeaveGroup(*net.Interface, net.Addr) error
	JoinSourceSpecificGroup(*net.Interface, net.Addr, net.Addr) error
	LeaveSourceSpecificGroup(*net.Interface, net.Addr, net.Addr) error
	ExcludeSourceSpecificGroup(*net.Interface, net.Addr, net.Addr) error
	IncludeSourceSpecificGroup(*net.Interface, net.Addr, net.Addr) error
}

func testMulticastSocketOptions(t *testing.T, c testIPv4MulticastConn, ifi *net.Interface, grp net.Addr) {
	const ttl = 255
	if err := c.SetMulticastTTL(ttl); err != nil {
		t.Error(err)
		return
	}
	if v, err := c.MulticastTTL(); err != nil {
		t.Error(err)
		return
	} else if v != ttl {
		t.Errorf("got %v; want %v", v, ttl)
		return
	}

	for _, toggle := range []bool{true, false} {
		if err := c.SetMulticastLoopback(toggle); err != nil {
			t.Error(err)
			return
		}
		if v, err := c.MulticastLoopback(); err != nil {
			t.Error(err)
			return
		} else if v != toggle {
			t.Errorf("got %v; want %v", v, toggle)
			return
		}
	}

	if err := c.JoinGroup(ifi, grp); err != nil {
		t.Error(err)
		return
	}
	if err := c.LeaveGroup(ifi, grp); err != nil {
		t.Error(err)
		return
	}
}

func testSourceSpecificMulticastSocketOptions(t *testing.T, c testIPv4MulticastConn, ifi *net.Interface, grp, src net.Addr) {
	// MCAST_JOIN_GROUP -> MCAST_BLOCK_SOURCE -> MCAST_UNBLOCK_SOURCE -> MCAST_LEAVE_GROUP
	if err := c.JoinGroup(ifi, grp); err != nil {
		t.Error(err)
		return
	}
	if err := c.ExcludeSourceSpecificGroup(ifi, grp, src); err != nil {
		switch runtime.GOOS {
		case "freebsd", "linux":
		default: // platforms that don't support IGMPv2/3 fail here
			t.Logf("not supported on %s", runtime.GOOS)
			return
		}
		t.Error(err)
		return
	}
	if err := c.IncludeSourceSpecificGroup(ifi, grp, src); err != nil {
		t.Error(err)
		return
	}
	if err := c.LeaveGroup(ifi, grp); err != nil {
		t.Error(err)
		return
	}

	// MCAST_JOIN_SOURCE_GROUP -> MCAST_LEAVE_SOURCE_GROUP
	if err := c.JoinSourceSpecificGroup(ifi, grp, src); err != nil {
		t.Error(err)
		return
	}
	if err := c.LeaveSourceSpecificGroup(ifi, grp, src); err != nil {
		t.Error(err)
		return
	}

	// MCAST_JOIN_SOURCE_GROUP -> MCAST_LEAVE_GROUP
	if err := c.JoinSourceSpecificGroup(ifi, grp, src); err != nil {
		t.Error(err)
		return
	}
	if err := c.LeaveGroup(ifi, grp); err != nil {
		t.Error(err)
		return
	}
}
