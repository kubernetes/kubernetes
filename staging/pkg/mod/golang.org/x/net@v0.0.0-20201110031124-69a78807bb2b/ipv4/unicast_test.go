// Copyright 2012 The Go Authors. All rights reserved.
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
	"golang.org/x/net/ipv4"
	"golang.org/x/net/nettest"
)

func TestPacketConnReadWriteUnicastUDP(t *testing.T) {
	switch runtime.GOOS {
	case "fuchsia", "hurd", "js", "nacl", "plan9", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	// Skip this check on z/OS since net.Interfaces() does not return loopback, however
	// this does not affect the test and it will still pass.
	if _, err := nettest.RoutedInterface("ip4", net.FlagUp|net.FlagLoopback); err != nil && runtime.GOOS != "zos" {
		t.Skipf("not available on %s", runtime.GOOS)
	}

	c, err := nettest.NewLocalPacketListener("udp4")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	p := ipv4.NewPacketConn(c)
	defer p.Close()

	dst := c.LocalAddr()
	cf := ipv4.FlagTTL | ipv4.FlagDst | ipv4.FlagInterface
	wb := []byte("HELLO-R-U-THERE")

	for i, toggle := range []bool{true, false, true} {
		if err := p.SetControlMessage(cf, toggle); err != nil {
			if protocolNotSupported(err) {
				t.Logf("not supported on %s", runtime.GOOS)
				continue
			}
			t.Fatal(err)
		}
		p.SetTTL(i + 1)
		if err := p.SetWriteDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
			t.Fatal(err)
		}
		if n, err := p.WriteTo(wb, nil, dst); err != nil {
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
	case "fuchsia", "hurd", "js", "nacl", "plan9", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !nettest.SupportsRawSocket() {
		t.Skipf("not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	// Skip this check on z/OS since net.Interfaces() does not return loopback, however
	// this does not affect the test and it will still pass.
	if _, err := nettest.RoutedInterface("ip4", net.FlagUp|net.FlagLoopback); err != nil && runtime.GOOS != "zos" {
		t.Skipf("not available on %s", runtime.GOOS)
	}

	c, err := net.ListenPacket("ip4:icmp", "0.0.0.0")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	dst, err := net.ResolveIPAddr("ip4", "127.0.0.1")
	if err != nil {
		t.Fatal(err)
	}
	p := ipv4.NewPacketConn(c)
	defer p.Close()
	cf := ipv4.FlagDst | ipv4.FlagInterface
	if runtime.GOOS != "illumos" && runtime.GOOS != "solaris" {
		// Illumos and Solaris never allow modification of ICMP properties.
		cf |= ipv4.FlagTTL
	}

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
			if protocolNotSupported(err) {
				t.Logf("not supported on %s", runtime.GOOS)
				continue
			}
			t.Fatal(err)
		}
		p.SetTTL(i + 1)
		if err := p.SetWriteDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
			t.Fatal(err)
		}
		if n, err := p.WriteTo(wb, nil, dst); err != nil {
			t.Fatal(err)
		} else if n != len(wb) {
			t.Fatalf("got %v; want %v", n, len(wb))
		}
		rb := make([]byte, 128)
	loop:
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
			m, err := icmp.ParseMessage(iana.ProtocolICMP, rb[:n])
			if err != nil {
				t.Fatal(err)
			}
			if runtime.GOOS == "linux" && m.Type == ipv4.ICMPTypeEcho {
				// On Linux we must handle own sent packets.
				goto loop
			}
			if m.Type != ipv4.ICMPTypeEchoReply || m.Code != 0 {
				t.Fatalf("got type=%v, code=%v; want type=%v, code=%v", m.Type, m.Code, ipv4.ICMPTypeEchoReply, 0)
			}
		}
	}
}

func TestRawConnReadWriteUnicastICMP(t *testing.T) {
	switch runtime.GOOS {
	case "fuchsia", "hurd", "js", "nacl", "plan9", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !nettest.SupportsRawSocket() {
		t.Skipf("not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	if _, err := nettest.RoutedInterface("ip4", net.FlagUp|net.FlagLoopback); err != nil {
		t.Skipf("not available on %s", runtime.GOOS)
	}

	c, err := net.ListenPacket("ip4:icmp", "0.0.0.0")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	dst, err := net.ResolveIPAddr("ip4", "127.0.0.1")
	if err != nil {
		t.Fatal(err)
	}
	r, err := ipv4.NewRawConn(c)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
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
			TTL:      i + 1,
			Protocol: 1,
			Dst:      dst.IP,
		}
		if err := r.SetControlMessage(cf, toggle); err != nil {
			if protocolNotSupported(err) {
				t.Logf("not supported on %s", runtime.GOOS)
				continue
			}
			t.Fatal(err)
		}
		if err := r.SetWriteDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
			t.Fatal(err)
		}
		if err := r.WriteTo(wh, wb, nil); err != nil {
			t.Fatal(err)
		}
		rb := make([]byte, ipv4.HeaderLen+128)
	loop:
		if err := r.SetReadDeadline(time.Now().Add(100 * time.Millisecond)); err != nil {
			t.Fatal(err)
		}
		if _, b, _, err := r.ReadFrom(rb); err != nil {
			switch runtime.GOOS {
			case "darwin", "ios": // older darwin kernels have some limitation on receiving icmp packet through raw socket
				t.Logf("not supported on %s", runtime.GOOS)
				continue
			}
			t.Fatal(err)
		} else {
			m, err := icmp.ParseMessage(iana.ProtocolICMP, b)
			if err != nil {
				t.Fatal(err)
			}
			if runtime.GOOS == "linux" && m.Type == ipv4.ICMPTypeEcho {
				// On Linux we must handle own sent packets.
				goto loop
			}
			if m.Type != ipv4.ICMPTypeEchoReply || m.Code != 0 {
				t.Fatalf("got type=%v, code=%v; want type=%v, code=%v", m.Type, m.Code, ipv4.ICMPTypeEchoReply, 0)
			}
		}
	}
}
