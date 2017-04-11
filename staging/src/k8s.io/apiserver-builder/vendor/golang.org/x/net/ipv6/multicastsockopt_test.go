// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6_test

import (
	"net"
	"runtime"
	"testing"

	"golang.org/x/net/internal/nettest"
	"golang.org/x/net/ipv6"
)

var packetConnMulticastSocketOptionTests = []struct {
	net, proto, addr string
	grp, src         net.Addr
}{
	{"udp6", "", "[ff02::]:0", &net.UDPAddr{IP: net.ParseIP("ff02::114")}, nil}, // see RFC 4727
	{"ip6", ":ipv6-icmp", "::", &net.IPAddr{IP: net.ParseIP("ff02::115")}, nil}, // see RFC 4727

	{"udp6", "", "[ff30::8000:0]:0", &net.UDPAddr{IP: net.ParseIP("ff30::8000:1")}, &net.UDPAddr{IP: net.IPv6loopback}}, // see RFC 5771
	{"ip6", ":ipv6-icmp", "::", &net.IPAddr{IP: net.ParseIP("ff30::8000:2")}, &net.IPAddr{IP: net.IPv6loopback}},        // see RFC 5771
}

func TestPacketConnMulticastSocketOptions(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9", "solaris", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !supportsIPv6 {
		t.Skip("ipv6 is not supported")
	}
	ifi := nettest.RoutedInterface("ip6", net.FlagUp|net.FlagMulticast|net.FlagLoopback)
	if ifi == nil {
		t.Skipf("not available on %s", runtime.GOOS)
	}

	m, ok := nettest.SupportsRawIPSocket()
	for _, tt := range packetConnMulticastSocketOptionTests {
		if tt.net == "ip6" && !ok {
			t.Log(m)
			continue
		}
		c, err := net.ListenPacket(tt.net+tt.proto, tt.addr)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		p := ipv6.NewPacketConn(c)
		defer p.Close()

		if tt.src == nil {
			testMulticastSocketOptions(t, p, ifi, tt.grp)
		} else {
			testSourceSpecificMulticastSocketOptions(t, p, ifi, tt.grp, tt.src)
		}
	}
}

type testIPv6MulticastConn interface {
	MulticastHopLimit() (int, error)
	SetMulticastHopLimit(ttl int) error
	MulticastLoopback() (bool, error)
	SetMulticastLoopback(bool) error
	JoinGroup(*net.Interface, net.Addr) error
	LeaveGroup(*net.Interface, net.Addr) error
	JoinSourceSpecificGroup(*net.Interface, net.Addr, net.Addr) error
	LeaveSourceSpecificGroup(*net.Interface, net.Addr, net.Addr) error
	ExcludeSourceSpecificGroup(*net.Interface, net.Addr, net.Addr) error
	IncludeSourceSpecificGroup(*net.Interface, net.Addr, net.Addr) error
}

func testMulticastSocketOptions(t *testing.T, c testIPv6MulticastConn, ifi *net.Interface, grp net.Addr) {
	const hoplim = 255
	if err := c.SetMulticastHopLimit(hoplim); err != nil {
		t.Error(err)
		return
	}
	if v, err := c.MulticastHopLimit(); err != nil {
		t.Error(err)
		return
	} else if v != hoplim {
		t.Errorf("got %v; want %v", v, hoplim)
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

func testSourceSpecificMulticastSocketOptions(t *testing.T, c testIPv6MulticastConn, ifi *net.Interface, grp, src net.Addr) {
	// MCAST_JOIN_GROUP -> MCAST_BLOCK_SOURCE -> MCAST_UNBLOCK_SOURCE -> MCAST_LEAVE_GROUP
	if err := c.JoinGroup(ifi, grp); err != nil {
		t.Error(err)
		return
	}
	if err := c.ExcludeSourceSpecificGroup(ifi, grp, src); err != nil {
		switch runtime.GOOS {
		case "freebsd", "linux":
		default: // platforms that don't support MLDv2 fail here
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
