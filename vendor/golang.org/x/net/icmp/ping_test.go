// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp_test

import (
	"errors"
	"fmt"
	"net"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/icmp"
	"golang.org/x/net/internal/iana"
	"golang.org/x/net/internal/nettest"
	"golang.org/x/net/ipv4"
	"golang.org/x/net/ipv6"
)

func googleAddr(c *icmp.PacketConn, protocol int) (net.Addr, error) {
	const host = "www.google.com"
	ips, err := net.LookupIP(host)
	if err != nil {
		return nil, err
	}
	netaddr := func(ip net.IP) (net.Addr, error) {
		switch c.LocalAddr().(type) {
		case *net.UDPAddr:
			return &net.UDPAddr{IP: ip}, nil
		case *net.IPAddr:
			return &net.IPAddr{IP: ip}, nil
		default:
			return nil, errors.New("neither UDPAddr nor IPAddr")
		}
	}
	for _, ip := range ips {
		switch protocol {
		case iana.ProtocolICMP:
			if ip.To4() != nil {
				return netaddr(ip)
			}
		case iana.ProtocolIPv6ICMP:
			if ip.To16() != nil && ip.To4() == nil {
				return netaddr(ip)
			}
		}
	}
	return nil, errors.New("no A or AAAA record")
}

type pingTest struct {
	network, address string
	protocol         int
	mtype            icmp.Type
}

var nonPrivilegedPingTests = []pingTest{
	{"udp4", "0.0.0.0", iana.ProtocolICMP, ipv4.ICMPTypeEcho},

	{"udp6", "::", iana.ProtocolIPv6ICMP, ipv6.ICMPTypeEchoRequest},
}

func TestNonPrivilegedPing(t *testing.T) {
	if testing.Short() {
		t.Skip("avoid external network")
	}
	switch runtime.GOOS {
	case "darwin":
	case "linux":
		t.Log("you may need to adjust the net.ipv4.ping_group_range kernel state")
	default:
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	for i, tt := range nonPrivilegedPingTests {
		if err := doPing(tt, i); err != nil {
			t.Error(err)
		}
	}
}

var privilegedPingTests = []pingTest{
	{"ip4:icmp", "0.0.0.0", iana.ProtocolICMP, ipv4.ICMPTypeEcho},

	{"ip6:ipv6-icmp", "::", iana.ProtocolIPv6ICMP, ipv6.ICMPTypeEchoRequest},
}

func TestPrivilegedPing(t *testing.T) {
	if testing.Short() {
		t.Skip("avoid external network")
	}
	if m, ok := nettest.SupportsRawIPSocket(); !ok {
		t.Skip(m)
	}

	for i, tt := range privilegedPingTests {
		if err := doPing(tt, i); err != nil {
			t.Error(err)
		}
	}
}

func doPing(tt pingTest, seq int) error {
	c, err := icmp.ListenPacket(tt.network, tt.address)
	if err != nil {
		return err
	}
	defer c.Close()

	dst, err := googleAddr(c, tt.protocol)
	if err != nil {
		return err
	}

	if tt.network != "udp6" && tt.protocol == iana.ProtocolIPv6ICMP {
		var f ipv6.ICMPFilter
		f.SetAll(true)
		f.Accept(ipv6.ICMPTypeDestinationUnreachable)
		f.Accept(ipv6.ICMPTypePacketTooBig)
		f.Accept(ipv6.ICMPTypeTimeExceeded)
		f.Accept(ipv6.ICMPTypeParameterProblem)
		f.Accept(ipv6.ICMPTypeEchoReply)
		if err := c.IPv6PacketConn().SetICMPFilter(&f); err != nil {
			return err
		}
	}

	wm := icmp.Message{
		Type: tt.mtype, Code: 0,
		Body: &icmp.Echo{
			ID: os.Getpid() & 0xffff, Seq: 1 << uint(seq),
			Data: []byte("HELLO-R-U-THERE"),
		},
	}
	wb, err := wm.Marshal(nil)
	if err != nil {
		return err
	}
	if n, err := c.WriteTo(wb, dst); err != nil {
		return err
	} else if n != len(wb) {
		return fmt.Errorf("got %v; want %v", n, len(wb))
	}

	rb := make([]byte, 1500)
	if err := c.SetReadDeadline(time.Now().Add(3 * time.Second)); err != nil {
		return err
	}
	n, peer, err := c.ReadFrom(rb)
	if err != nil {
		return err
	}
	rm, err := icmp.ParseMessage(tt.protocol, rb[:n])
	if err != nil {
		return err
	}
	switch rm.Type {
	case ipv4.ICMPTypeEchoReply, ipv6.ICMPTypeEchoReply:
		return nil
	default:
		return fmt.Errorf("got %+v from %v; want echo reply", rm, peer)
	}
}

func TestConcurrentNonPrivilegedListenPacket(t *testing.T) {
	if testing.Short() {
		t.Skip("avoid external network")
	}
	switch runtime.GOOS {
	case "darwin":
	case "linux":
		t.Log("you may need to adjust the net.ipv4.ping_group_range kernel state")
	default:
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	network, address := "udp4", "127.0.0.1"
	if !nettest.SupportsIPv4() {
		network, address = "udp6", "::1"
	}
	const N = 1000
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			c, err := icmp.ListenPacket(network, address)
			if err != nil {
				t.Error(err)
				return
			}
			c.Close()
		}()
	}
	wg.Wait()
}
