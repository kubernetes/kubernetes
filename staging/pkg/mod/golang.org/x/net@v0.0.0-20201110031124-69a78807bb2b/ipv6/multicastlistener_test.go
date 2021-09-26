// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6_test

import (
	"net"
	"runtime"
	"testing"

	"golang.org/x/net/ipv6"
	"golang.org/x/net/nettest"
)

var udpMultipleGroupListenerTests = []net.Addr{
	&net.UDPAddr{IP: net.ParseIP("ff02::114")}, // see RFC 4727
	&net.UDPAddr{IP: net.ParseIP("ff02::1:114")},
	&net.UDPAddr{IP: net.ParseIP("ff02::2:114")},
}

func TestUDPSinglePacketConnWithMultipleGroupListeners(t *testing.T) {
	switch runtime.GOOS {
	case "fuchsia", "hurd", "js", "nacl", "plan9", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !nettest.SupportsIPv6() {
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
			if _, err := nettest.MulticastSource("ip6", &ifi); err != nil {
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
	case "fuchsia", "hurd", "js", "nacl", "plan9", "windows", "zos":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !nettest.SupportsIPv6() {
		t.Skip("ipv6 is not supported")
	}

	for _, gaddr := range udpMultipleGroupListenerTests {
		c1, err := net.ListenPacket("udp6", "[ff02::]:0") // wildcard address with reusable port
		if err != nil {
			t.Fatal(err)
		}
		defer c1.Close()
		_, port, err := net.SplitHostPort(c1.LocalAddr().String())
		if err != nil {
			t.Fatal(err)
		}
		c2, err := net.ListenPacket("udp6", net.JoinHostPort("ff02::", port)) // wildcard address with reusable port
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
			if _, err := nettest.MulticastSource("ip6", &ifi); err != nil {
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
	case "fuchsia", "hurd", "js", "nacl", "plan9", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !nettest.SupportsIPv6() {
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
	port := "0"
	for i, ifi := range ift {
		ip, err := nettest.MulticastSource("ip6", &ifi)
		if err != nil {
			continue
		}
		c, err := net.ListenPacket("udp6", net.JoinHostPort(ip.String()+"%"+ifi.Name, port)) // unicast address with non-reusable port
		if err != nil {
			// The listen may fail when the serivce is
			// already in use, but it's fine because the
			// purpose of this is not to test the
			// bookkeeping of IP control block inside the
			// kernel.
			t.Log(err)
			continue
		}
		defer c.Close()
		if port == "0" {
			_, port, err = net.SplitHostPort(c.LocalAddr().String())
			if err != nil {
				t.Fatal(err)
			}
		}
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
	case "fuchsia", "hurd", "js", "nacl", "plan9", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !nettest.SupportsIPv6() {
		t.Skip("ipv6 is not supported")
	}
	if !nettest.SupportsRawSocket() {
		t.Skipf("not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
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
		if _, err := nettest.MulticastSource("ip6", &ifi); err != nil {
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
	case "darwin", "ios", "dragonfly", "openbsd": // platforms that return fe80::1%lo0: bind: can't assign requested address
		t.Skipf("not supported on %s", runtime.GOOS)
	case "fuchsia", "hurd", "js", "nacl", "plan9", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !nettest.SupportsIPv6() {
		t.Skip("ipv6 is not supported")
	}
	if !nettest.SupportsRawSocket() {
		t.Skipf("not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
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
		ip, err := nettest.MulticastSource("ip6", &ifi)
		if err != nil {
			continue
		}
		c, err := net.ListenPacket("ip6:ipv6-icmp", ip.String()+"%"+ifi.Name) // unicast address
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
