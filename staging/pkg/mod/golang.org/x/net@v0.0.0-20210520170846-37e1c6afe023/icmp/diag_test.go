// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package icmp_test

import (
	"errors"
	"flag"
	"fmt"
	"net"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/icmp"
	"golang.org/x/net/internal/iana"
	"golang.org/x/net/ipv4"
	"golang.org/x/net/ipv6"
	"golang.org/x/net/nettest"
)

var testDiag = flag.Bool("diag", false, "whether to test ICMP message exchange with external network")

type diagTest struct {
	network, address string
	protocol         int
	m                icmp.Message
}

func TestDiag(t *testing.T) {
	if !*testDiag {
		t.Skip("avoid external network")
	}

	t.Run("Ping/NonPrivileged", func(t *testing.T) {
		if m, ok := supportsNonPrivilegedICMP(); !ok {
			t.Skip(m)
		}
		for i, dt := range []diagTest{
			{
				"udp4", "0.0.0.0", iana.ProtocolICMP,
				icmp.Message{
					Type: ipv4.ICMPTypeEcho, Code: 0,
					Body: &icmp.Echo{
						ID:   os.Getpid() & 0xffff,
						Data: []byte("HELLO-R-U-THERE"),
					},
				},
			},

			{
				"udp6", "::", iana.ProtocolIPv6ICMP,
				icmp.Message{
					Type: ipv6.ICMPTypeEchoRequest, Code: 0,
					Body: &icmp.Echo{
						ID:   os.Getpid() & 0xffff,
						Data: []byte("HELLO-R-U-THERE"),
					},
				},
			},
		} {
			if err := doDiag(dt, i); err != nil {
				t.Error(err)
			}
		}
	})
	t.Run("Ping/Privileged", func(t *testing.T) {
		if !nettest.SupportsRawSocket() {
			t.Skipf("not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
		}
		for i, dt := range []diagTest{
			{
				"ip4:icmp", "0.0.0.0", iana.ProtocolICMP,
				icmp.Message{
					Type: ipv4.ICMPTypeEcho, Code: 0,
					Body: &icmp.Echo{
						ID:   os.Getpid() & 0xffff,
						Data: []byte("HELLO-R-U-THERE"),
					},
				},
			},

			{
				"ip6:ipv6-icmp", "::", iana.ProtocolIPv6ICMP,
				icmp.Message{
					Type: ipv6.ICMPTypeEchoRequest, Code: 0,
					Body: &icmp.Echo{
						ID:   os.Getpid() & 0xffff,
						Data: []byte("HELLO-R-U-THERE"),
					},
				},
			},
		} {
			if err := doDiag(dt, i); err != nil {
				t.Error(err)
			}
		}
	})
	t.Run("Probe/Privileged", func(t *testing.T) {
		if !nettest.SupportsRawSocket() {
			t.Skipf("not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
		}
		for i, dt := range []diagTest{
			{
				"ip4:icmp", "0.0.0.0", iana.ProtocolICMP,
				icmp.Message{
					Type: ipv4.ICMPTypeExtendedEchoRequest, Code: 0,
					Body: &icmp.ExtendedEchoRequest{
						ID:    os.Getpid() & 0xffff,
						Local: true,
						Extensions: []icmp.Extension{
							&icmp.InterfaceIdent{
								Class: 3, Type: 1,
								Name: "doesnotexist",
							},
						},
					},
				},
			},

			{
				"ip6:ipv6-icmp", "::", iana.ProtocolIPv6ICMP,
				icmp.Message{
					Type: ipv6.ICMPTypeExtendedEchoRequest, Code: 0,
					Body: &icmp.ExtendedEchoRequest{
						ID:    os.Getpid() & 0xffff,
						Local: true,
						Extensions: []icmp.Extension{
							&icmp.InterfaceIdent{
								Class: 3, Type: 1,
								Name: "doesnotexist",
							},
						},
					},
				},
			},
		} {
			if err := doDiag(dt, i); err != nil {
				t.Error(err)
			}
		}
	})
}

func doDiag(dt diagTest, seq int) error {
	c, err := icmp.ListenPacket(dt.network, dt.address)
	if err != nil {
		return err
	}
	defer c.Close()

	dst, err := googleAddr(c, dt.protocol)
	if err != nil {
		return err
	}

	if dt.network != "udp6" && dt.protocol == iana.ProtocolIPv6ICMP {
		var f ipv6.ICMPFilter
		f.SetAll(true)
		f.Accept(ipv6.ICMPTypeDestinationUnreachable)
		f.Accept(ipv6.ICMPTypePacketTooBig)
		f.Accept(ipv6.ICMPTypeTimeExceeded)
		f.Accept(ipv6.ICMPTypeParameterProblem)
		f.Accept(ipv6.ICMPTypeEchoReply)
		f.Accept(ipv6.ICMPTypeExtendedEchoReply)
		if err := c.IPv6PacketConn().SetICMPFilter(&f); err != nil {
			return err
		}
	}

	switch m := dt.m.Body.(type) {
	case *icmp.Echo:
		m.Seq = 1 << uint(seq)
	case *icmp.ExtendedEchoRequest:
		m.Seq = 1 << uint(seq)
	}
	wb, err := dt.m.Marshal(nil)
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
	rm, err := icmp.ParseMessage(dt.protocol, rb[:n])
	if err != nil {
		return err
	}
	switch {
	case dt.m.Type == ipv4.ICMPTypeEcho && rm.Type == ipv4.ICMPTypeEchoReply:
		fallthrough
	case dt.m.Type == ipv6.ICMPTypeEchoRequest && rm.Type == ipv6.ICMPTypeEchoReply:
		fallthrough
	case dt.m.Type == ipv4.ICMPTypeExtendedEchoRequest && rm.Type == ipv4.ICMPTypeExtendedEchoReply:
		fallthrough
	case dt.m.Type == ipv6.ICMPTypeExtendedEchoRequest && rm.Type == ipv6.ICMPTypeExtendedEchoReply:
		return nil
	default:
		return fmt.Errorf("got %+v from %v; want echo reply or extended echo reply", rm, peer)
	}
}

func googleAddr(c *icmp.PacketConn, protocol int) (net.Addr, error) {
	host := "ipv4.google.com"
	if protocol == iana.ProtocolIPv6ICMP {
		host = "ipv6.google.com"
	}
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
	if len(ips) > 0 {
		return netaddr(ips[0])
	}
	return nil, errors.New("no A or AAAA record")
}

func TestConcurrentNonPrivilegedListenPacket(t *testing.T) {
	if testing.Short() {
		t.Skip("avoid external network")
	}
	if m, ok := supportsNonPrivilegedICMP(); !ok {
		t.Skip(m)
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

var (
	nonPrivOnce sync.Once
	nonPrivMsg  string
	nonPrivICMP bool
)

func supportsNonPrivilegedICMP() (string, bool) {
	nonPrivOnce.Do(func() {
		switch runtime.GOOS {
		case "darwin", "ios":
			nonPrivICMP = true
		case "linux":
			for _, t := range []struct{ network, address string }{
				{"udp4", "127.0.0.1"},
				{"udp6", "::1"},
			} {
				c, err := icmp.ListenPacket(t.network, t.address)
				if err != nil {
					nonPrivMsg = "you may need to adjust the net.ipv4.ping_group_range kernel state"
					return
				}
				c.Close()
			}
			nonPrivICMP = true
		default:
			nonPrivMsg = "not supported on " + runtime.GOOS
		}
	})
	return nonPrivMsg, nonPrivICMP
}
