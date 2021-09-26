// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6_test

import (
	"bytes"
	"net"
	"os"
	"runtime"
	"testing"
	"time"

	"golang.org/x/net/icmp"
	"golang.org/x/net/internal/iana"
	"golang.org/x/net/ipv6"
	"golang.org/x/net/nettest"
)

func TestPacketConnReadWriteUnicastUDP(t *testing.T) {
	switch runtime.GOOS {
	case "fuchsia", "hurd", "js", "nacl", "plan9", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !nettest.SupportsIPv6() {
		t.Skip("ipv6 is not supported")
	}

	c, err := nettest.NewLocalPacketListener("udp6")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	p := ipv6.NewPacketConn(c)
	defer p.Close()

	dst := c.LocalAddr()
	cm := ipv6.ControlMessage{
		TrafficClass: iana.DiffServAF11 | iana.CongestionExperienced,
		Src:          net.IPv6loopback,
	}
	cf := ipv6.FlagTrafficClass | ipv6.FlagHopLimit | ipv6.FlagSrc | ipv6.FlagDst | ipv6.FlagInterface | ipv6.FlagPathMTU
	ifi, _ := nettest.RoutedInterface("ip6", net.FlagUp|net.FlagLoopback)
	if ifi != nil {
		cm.IfIndex = ifi.Index
	}
	wb := []byte("HELLO-R-U-THERE")

	for i, toggle := range []bool{true, false, true} {
		if err := p.SetControlMessage(cf, toggle); err != nil {
			if protocolNotSupported(err) {
				t.Logf("not supported on %s", runtime.GOOS)
				continue
			}
			t.Fatal(err)
		}
		cm.HopLimit = i + 1
		if err := p.SetWriteDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
			t.Fatal(err)
		}
		if n, err := p.WriteTo(wb, &cm, dst); err != nil {
			t.Fatal(err)
		} else if n != len(wb) {
			t.Fatalf("got %v; want %v", n, len(wb))
		}
		rb := make([]byte, 128)
		if err := p.SetReadDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
			t.Fatal(err)
		}
		if n, _, _, err := p.ReadFrom(rb); err != nil {
			t.Fatal(err)
		} else if !bytes.Equal(rb[:n], wb) {
			t.Fatalf("got %v; want %v", rb[:n], wb)
		}
	}
}

func TestPacketConnReadWriteUnicastICMP(t *testing.T) {
	switch runtime.GOOS {
	case "fuchsia", "hurd", "js", "nacl", "plan9", "windows", "zos":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !nettest.SupportsIPv6() {
		t.Skip("ipv6 is not supported")
	}
	if !nettest.SupportsRawSocket() {
		t.Skipf("not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	c, err := net.ListenPacket("ip6:ipv6-icmp", "::1")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	p := ipv6.NewPacketConn(c)
	defer p.Close()

	dst, err := net.ResolveIPAddr("ip6", "::1")
	if err != nil {
		t.Fatal(err)
	}

	pshicmp := icmp.IPv6PseudoHeader(c.LocalAddr().(*net.IPAddr).IP, dst.IP)
	cm := ipv6.ControlMessage{
		TrafficClass: iana.DiffServAF11 | iana.CongestionExperienced,
		Src:          net.IPv6loopback,
	}
	cf := ipv6.FlagTrafficClass | ipv6.FlagHopLimit | ipv6.FlagSrc | ipv6.FlagDst | ipv6.FlagInterface | ipv6.FlagPathMTU
	ifi, _ := nettest.RoutedInterface("ip6", net.FlagUp|net.FlagLoopback)
	if ifi != nil {
		cm.IfIndex = ifi.Index
	}

	var f ipv6.ICMPFilter
	f.SetAll(true)
	f.Accept(ipv6.ICMPTypeEchoReply)
	if err := p.SetICMPFilter(&f); err != nil {
		t.Fatal(err)
	}

	var psh []byte
	for i, toggle := range []bool{true, false, true} {
		if toggle {
			psh = nil
			if err := p.SetChecksum(true, 2); err != nil {
				// AIX, Illumos and Solaris never allow
				// modification of ICMP properties.
				switch runtime.GOOS {
				case "aix", "illumos", "solaris":
				default:
					t.Fatal(err)
				}
			}
		} else {
			psh = pshicmp
			// Some platforms never allow to disable the
			// kernel checksum processing.
			p.SetChecksum(false, -1)
		}
		wb, err := (&icmp.Message{
			Type: ipv6.ICMPTypeEchoRequest, Code: 0,
			Body: &icmp.Echo{
				ID: os.Getpid() & 0xffff, Seq: i + 1,
				Data: []byte("HELLO-R-U-THERE"),
			},
		}).Marshal(psh)
		if err != nil {
			t.Fatal(err)
		}
		if err := p.SetControlMessage(cf, toggle); err != nil {
			if protocolNotSupported(err) {
				t.Logf("not supported on %s", runtime.GOOS)
				continue
			}
			t.Fatal(err)
		}
		cm.HopLimit = i + 1
		if err := p.SetWriteDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
			t.Fatal(err)
		}
		if n, err := p.WriteTo(wb, &cm, dst); err != nil {
			t.Fatal(err)
		} else if n != len(wb) {
			t.Fatalf("got %v; want %v", n, len(wb))
		}
		rb := make([]byte, 128)
		if err := p.SetReadDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
			t.Fatal(err)
		}
		if n, _, _, err := p.ReadFrom(rb); err != nil {
			switch runtime.GOOS {
			case "darwin", "ios": // older darwin kernels have some limitation on receiving icmp packet through raw socket
				t.Logf("not supported on %s", runtime.GOOS)
				continue
			}
			t.Fatal(err)
		} else {
			if m, err := icmp.ParseMessage(iana.ProtocolIPv6ICMP, rb[:n]); err != nil {
				t.Fatal(err)
			} else if m.Type != ipv6.ICMPTypeEchoReply || m.Code != 0 {
				t.Fatalf("got type=%v, code=%v; want type=%v, code=%v", m.Type, m.Code, ipv6.ICMPTypeEchoReply, 0)
			}
		}
	}
}
