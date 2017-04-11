// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6_test

import (
	"fmt"
	"net"
	"runtime"
	"testing"

	"golang.org/x/net/internal/nettest"
	"golang.org/x/net/ipv6"
)

var udpMultipleGroupListenerTests = []net.Addr{
	&net.UDPAddr{IP: net.ParseIP("ff02::114")}, // see RFC 4727
	&net.UDPAddr{IP: net.ParseIP("ff02::1:114")},
	&net.UDPAddr{IP: net.ParseIP("ff02::2:114")},
}

func TestUDPSinglePacketConnWithMultipleGroupListeners(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9", "solaris", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !supportsIPv6 {
		t.Skip("ipv6 is not supported")
	}

	for _, gaddr := range udpMultipleGroupListenerTests {
		c, err := net.ListenPacket("udp6", "[::]:0") // wildcard address with non-reusable port
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()

		p := ipv6.NewPacketConn(c)
		var mift []*net.Interface

		ift, err := net.Interfaces()
		if err != nil {
			t.Fatal(err)
		}
		for i, ifi := range ift {
			if _, ok := nettest.IsMulticastCapable("ip6", &ifi); !ok {
				continue
			}
			if err := p.JoinGroup(&ifi, gaddr); err != nil {
				t.Fatal(err)
			}
			mift = append(mift, &ift[i])
		}
		for _, ifi := range mift {
			if err := p.LeaveGroup(ifi, gaddr); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestUDPMultiplePacketConnWithMultipleGroupListeners(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9", "solaris", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !supportsIPv6 {
		t.Skip("ipv6 is not supported")
	}

	for _, gaddr := range udpMultipleGroupListenerTests {
		c1, err := net.ListenPacket("udp6", "[ff02::]:1024") // wildcard address with reusable port
		if err != nil {
			t.Fatal(err)
		}
		defer c1.Close()

		c2, err := net.ListenPacket("udp6", "[ff02::]:1024") // wildcard address with reusable port
		if err != nil {
			t.Fatal(err)
		}
		defer c2.Close()

		var ps [2]*ipv6.PacketConn
		ps[0] = ipv6.NewPacketConn(c1)
		ps[1] = ipv6.NewPacketConn(c2)
		var mift []*net.Interface

		ift, err := net.Interfaces()
		if err != nil {
			t.Fatal(err)
		}
		for i, ifi := range ift {
			if _, ok := nettest.IsMulticastCapable("ip6", &ifi); !ok {
				continue
			}
			for _, p := range ps {
				if err := p.JoinGroup(&ifi, gaddr); err != nil {
					t.Fatal(err)
				}
			}
			mift = append(mift, &ift[i])
		}
		for _, ifi := range mift {
			for _, p := range ps {
				if err := p.LeaveGroup(ifi, gaddr); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
}

func TestUDPPerInterfaceSinglePacketConnWithSingleGroupListener(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9", "solaris", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !supportsIPv6 {
		t.Skip("ipv6 is not supported")
	}

	gaddr := net.IPAddr{IP: net.ParseIP("ff02::114")} // see RFC 4727
	type ml struct {
		c   *ipv6.PacketConn
		ifi *net.Interface
	}
	var mlt []*ml

	ift, err := net.Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	for i, ifi := range ift {
		ip, ok := nettest.IsMulticastCapable("ip6", &ifi)
		if !ok {
			continue
		}
		c, err := net.ListenPacket("udp6", fmt.Sprintf("[%s%%%s]:1024", ip.String(), ifi.Name)) // unicast address with non-reusable port
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		p := ipv6.NewPacketConn(c)
		if err := p.JoinGroup(&ifi, &gaddr); err != nil {
			t.Fatal(err)
		}
		mlt = append(mlt, &ml{p, &ift[i]})
	}
	for _, m := range mlt {
		if err := m.c.LeaveGroup(m.ifi, &gaddr); err != nil {
			t.Fatal(err)
		}
	}
}

func TestIPSinglePacketConnWithSingleGroupListener(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9", "solaris", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !supportsIPv6 {
		t.Skip("ipv6 is not supported")
	}
	if m, ok := nettest.SupportsRawIPSocket(); !ok {
		t.Skip(m)
	}

	c, err := net.ListenPacket("ip6:ipv6-icmp", "::") // wildcard address
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	p := ipv6.NewPacketConn(c)
	gaddr := net.IPAddr{IP: net.ParseIP("ff02::114")} // see RFC 4727
	var mift []*net.Interface

	ift, err := net.Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	for i, ifi := range ift {
		if _, ok := nettest.IsMulticastCapable("ip6", &ifi); !ok {
			continue
		}
		if err := p.JoinGroup(&ifi, &gaddr); err != nil {
			t.Fatal(err)
		}
		mift = append(mift, &ift[i])
	}
	for _, ifi := range mift {
		if err := p.LeaveGroup(ifi, &gaddr); err != nil {
			t.Fatal(err)
		}
	}
}

func TestIPPerInterfaceSinglePacketConnWithSingleGroupListener(t *testing.T) {
	switch runtime.GOOS {
	case "darwin", "dragonfly", "openbsd": // platforms that return fe80::1%lo0: bind: can't assign requested address
		t.Skipf("not supported on %s", runtime.GOOS)
	case "nacl", "plan9", "solaris", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !supportsIPv6 {
		t.Skip("ipv6 is not supported")
	}
	if m, ok := nettest.SupportsRawIPSocket(); !ok {
		t.Skip(m)
	}

	gaddr := net.IPAddr{IP: net.ParseIP("ff02::114")} // see RFC 4727
	type ml struct {
		c   *ipv6.PacketConn
		ifi *net.Interface
	}
	var mlt []*ml

	ift, err := net.Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	for i, ifi := range ift {
		ip, ok := nettest.IsMulticastCapable("ip6", &ifi)
		if !ok {
			continue
		}
		c, err := net.ListenPacket("ip6:ipv6-icmp", fmt.Sprintf("%s%%%s", ip.String(), ifi.Name)) // unicast address
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		p := ipv6.NewPacketConn(c)
		if err := p.JoinGroup(&ifi, &gaddr); err != nil {
			t.Fatal(err)
		}
		mlt = append(mlt, &ml{p, &ift[i]})
	}
	for _, m := range mlt {
		if err := m.c.LeaveGroup(m.ifi, &gaddr); err != nil {
			t.Fatal(err)
		}
	}
}
