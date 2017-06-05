// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4_test

import (
	"bytes"
	"net"
	"os"
	"runtime"
	"testing"
	"time"

	"golang.org/x/net/icmp"
	"golang.org/x/net/internal/iana"
	"golang.org/x/net/internal/nettest"
	"golang.org/x/net/ipv4"
)

var packetConnReadWriteMulticastUDPTests = []struct {
	addr     string
	grp, src *net.UDPAddr
}{
	{"224.0.0.0:0", &net.UDPAddr{IP: net.IPv4(224, 0, 0, 254)}, nil}, // see RFC 4727

	{"232.0.1.0:0", &net.UDPAddr{IP: net.IPv4(232, 0, 1, 254)}, &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)}}, // see RFC 5771
}

func TestPacketConnReadWriteMulticastUDP(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9", "solaris", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	ifi := nettest.RoutedInterface("ip4", net.FlagUp|net.FlagMulticast|net.FlagLoopback)
	if ifi == nil {
		t.Skipf("not available on %s", runtime.GOOS)
	}

	for _, tt := range packetConnReadWriteMulticastUDPTests {
		c, err := net.ListenPacket("udp4", tt.addr)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()

		grp := *tt.grp
		grp.Port = c.LocalAddr().(*net.UDPAddr).Port
		p := ipv4.NewPacketConn(c)
		defer p.Close()
		if tt.src == nil {
			if err := p.JoinGroup(ifi, &grp); err != nil {
				t.Fatal(err)
			}
			defer p.LeaveGroup(ifi, &grp)
		} else {
			if err := p.JoinSourceSpecificGroup(ifi, &grp, tt.src); err != nil {
				switch runtime.GOOS {
				case "freebsd", "linux":
				default: // platforms that don't support IGMPv2/3 fail here
					t.Logf("not supported on %s", runtime.GOOS)
					continue
				}
				t.Fatal(err)
			}
			defer p.LeaveSourceSpecificGroup(ifi, &grp, tt.src)
		}
		if err := p.SetMulticastInterface(ifi); err != nil {
			t.Fatal(err)
		}
		if _, err := p.MulticastInterface(); err != nil {
			t.Fatal(err)
		}
		if err := p.SetMulticastLoopback(true); err != nil {
			t.Fatal(err)
		}
		if _, err := p.MulticastLoopback(); err != nil {
			t.Fatal(err)
		}
		cf := ipv4.FlagTTL | ipv4.FlagDst | ipv4.FlagInterface
		wb := []byte("HELLO-R-U-THERE")

		for i, toggle := range []bool{true, false, true} {
			if err := p.SetControlMessage(cf, toggle); err != nil {
				if nettest.ProtocolNotSupported(err) {
					t.Logf("not supported on %s", runtime.GOOS)
					continue
				}
				t.Fatal(err)
			}
			if err := p.SetDeadline(time.Now().Add(200 * time.Millisecond)); err != nil {
				t.Fatal(err)
			}
			p.SetMulticastTTL(i + 1)
			if n, err := p.WriteTo(wb, nil, &grp); err != nil {
				t.Fatal(err)
			} else if n != len(wb) {
				t.Fatalf("got %v; want %v", n, len(wb))
			}
			rb := make([]byte, 128)
			if n, _, _, err := p.ReadFrom(rb); err != nil {
				t.Fatal(err)
			} else if !bytes.Equal(rb[:n], wb) {
				t.Fatalf("got %v; want %v", rb[:n], wb)
			}
		}
	}
}

var packetConnReadWriteMulticastICMPTests = []struct {
	grp, src *net.IPAddr
}{
	{&net.IPAddr{IP: net.IPv4(224, 0, 0, 254)}, nil}, // see RFC 4727

	{&net.IPAddr{IP: net.IPv4(232, 0, 1, 254)}, &net.IPAddr{IP: net.IPv4(127, 0, 0, 1)}}, // see RFC 5771
}

func TestPacketConnReadWriteMulticastICMP(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9", "solaris", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if m, ok := nettest.SupportsRawIPSocket(); !ok {
		t.Skip(m)
	}
	ifi := nettest.RoutedInterface("ip4", net.FlagUp|net.FlagMulticast|net.FlagLoopback)
	if ifi == nil {
		t.Skipf("not available on %s", runtime.GOOS)
	}

	for _, tt := range packetConnReadWriteMulticastICMPTests {
		c, err := net.ListenPacket("ip4:icmp", "0.0.0.0")
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()

		p := ipv4.NewPacketConn(c)
		defer p.Close()
		if tt.src == nil {
			if err := p.JoinGroup(ifi, tt.grp); err != nil {
				t.Fatal(err)
			}
			defer p.LeaveGroup(ifi, tt.grp)
		} else {
			if err := p.JoinSourceSpecificGroup(ifi, tt.grp, tt.src); err != nil {
				switch runtime.GOOS {
				case "freebsd", "linux":
				default: // platforms that don't support IGMPv2/3 fail here
					t.Logf("not supported on %s", runtime.GOOS)
					continue
				}
				t.Fatal(err)
			}
			defer p.LeaveSourceSpecificGroup(ifi, tt.grp, tt.src)
		}
		if err := p.SetMulticastInterface(ifi); err != nil {
			t.Fatal(err)
		}
		if _, err := p.MulticastInterface(); err != nil {
			t.Fatal(err)
		}
		if err := p.SetMulticastLoopback(true); err != nil {
			t.Fatal(err)
		}
		if _, err := p.MulticastLoopback(); err != nil {
			t.Fatal(err)
		}
		cf := ipv4.FlagTTL | ipv4.FlagDst | ipv4.FlagInterface

		for i, toggle := range []bool{true, false, true} {
			wb, err := (&icmp.Message{
				Type: ipv4.ICMPTypeEcho, Code: 0,
				Body: &icmp.Echo{
					ID: os.Getpid() & 0xffff, Seq: i + 1,
					Data: []byte("HELLO-R-U-THERE"),
				},
			}).Marshal(nil)
			if err != nil {
				t.Fatal(err)
			}
			if err := p.SetControlMessage(cf, toggle); err != nil {
				if nettest.ProtocolNotSupported(err) {
					t.Logf("not supported on %s", runtime.GOOS)
					continue
				}
				t.Fatal(err)
			}
			if err := p.SetDeadline(time.Now().Add(200 * time.Millisecond)); err != nil {
				t.Fatal(err)
			}
			p.SetMulticastTTL(i + 1)
			if n, err := p.WriteTo(wb, nil, tt.grp); err != nil {
				t.Fatal(err)
			} else if n != len(wb) {
				t.Fatalf("got %v; want %v", n, len(wb))
			}
			rb := make([]byte, 128)
			if n, _, _, err := p.ReadFrom(rb); err != nil {
				t.Fatal(err)
			} else {
				m, err := icmp.ParseMessage(iana.ProtocolICMP, rb[:n])
				if err != nil {
					t.Fatal(err)
				}
				switch {
				case m.Type == ipv4.ICMPTypeEchoReply && m.Code == 0: // net.inet.icmp.bmcastecho=1
				case m.Type == ipv4.ICMPTypeEcho && m.Code == 0: // net.inet.icmp.bmcastecho=0
				default:
					t.Fatalf("got type=%v, code=%v; want type=%v, code=%v", m.Type, m.Code, ipv4.ICMPTypeEchoReply, 0)
				}
			}
		}
	}
}

var rawConnReadWriteMulticastICMPTests = []struct {
	grp, src *net.IPAddr
}{
	{&net.IPAddr{IP: net.IPv4(224, 0, 0, 254)}, nil}, // see RFC 4727

	{&net.IPAddr{IP: net.IPv4(232, 0, 1, 254)}, &net.IPAddr{IP: net.IPv4(127, 0, 0, 1)}}, // see RFC 5771
}

func TestRawConnReadWriteMulticastICMP(t *testing.T) {
	if testing.Short() {
		t.Skip("to avoid external network")
	}
	if m, ok := nettest.SupportsRawIPSocket(); !ok {
		t.Skip(m)
	}
	ifi := nettest.RoutedInterface("ip4", net.FlagUp|net.FlagMulticast|net.FlagLoopback)
	if ifi == nil {
		t.Skipf("not available on %s", runtime.GOOS)
	}

	for _, tt := range rawConnReadWriteMulticastICMPTests {
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
			if err := r.JoinGroup(ifi, tt.grp); err != nil {
				t.Fatal(err)
			}
			defer r.LeaveGroup(ifi, tt.grp)
		} else {
			if err := r.JoinSourceSpecificGroup(ifi, tt.grp, tt.src); err != nil {
				switch runtime.GOOS {
				case "freebsd", "linux":
				default: // platforms that don't support IGMPv2/3 fail here
					t.Logf("not supported on %s", runtime.GOOS)
					continue
				}
				t.Fatal(err)
			}
			defer r.LeaveSourceSpecificGroup(ifi, tt.grp, tt.src)
		}
		if err := r.SetMulticastInterface(ifi); err != nil {
			t.Fatal(err)
		}
		if _, err := r.MulticastInterface(); err != nil {
			t.Fatal(err)
		}
		if err := r.SetMulticastLoopback(true); err != nil {
			t.Fatal(err)
		}
		if _, err := r.MulticastLoopback(); err != nil {
			t.Fatal(err)
		}
		cf := ipv4.FlagTTL | ipv4.FlagDst | ipv4.FlagInterface

		for i, toggle := range []bool{true, false, true} {
			wb, err := (&icmp.Message{
				Type: ipv4.ICMPTypeEcho, Code: 0,
				Body: &icmp.Echo{
					ID: os.Getpid() & 0xffff, Seq: i + 1,
					Data: []byte("HELLO-R-U-THERE"),
				},
			}).Marshal(nil)
			if err != nil {
				t.Fatal(err)
			}
			wh := &ipv4.Header{
				Version:  ipv4.Version,
				Len:      ipv4.HeaderLen,
				TOS:      i + 1,
				TotalLen: ipv4.HeaderLen + len(wb),
				Protocol: 1,
				Dst:      tt.grp.IP,
			}
			if err := r.SetControlMessage(cf, toggle); err != nil {
				if nettest.ProtocolNotSupported(err) {
					t.Logf("not supported on %s", runtime.GOOS)
					continue
				}
				t.Fatal(err)
			}
			if err := r.SetDeadline(time.Now().Add(200 * time.Millisecond)); err != nil {
				t.Fatal(err)
			}
			r.SetMulticastTTL(i + 1)
			if err := r.WriteTo(wh, wb, nil); err != nil {
				t.Fatal(err)
			}
			rb := make([]byte, ipv4.HeaderLen+128)
			if rh, b, _, err := r.ReadFrom(rb); err != nil {
				t.Fatal(err)
			} else {
				m, err := icmp.ParseMessage(iana.ProtocolICMP, b)
				if err != nil {
					t.Fatal(err)
				}
				switch {
				case (rh.Dst.IsLoopback() || rh.Dst.IsLinkLocalUnicast() || rh.Dst.IsGlobalUnicast()) && m.Type == ipv4.ICMPTypeEchoReply && m.Code == 0: // net.inet.icmp.bmcastecho=1
				case rh.Dst.IsMulticast() && m.Type == ipv4.ICMPTypeEcho && m.Code == 0: // net.inet.icmp.bmcastecho=0
				default:
					t.Fatalf("got type=%v, code=%v; want type=%v, code=%v", m.Type, m.Code, ipv4.ICMPTypeEchoReply, 0)
				}
			}
		}
	}
}
