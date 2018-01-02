// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6_test

import (
	"bytes"
	"net"
	"runtime"
	"strings"
	"sync"
	"testing"

	"golang.org/x/net/internal/iana"
	"golang.org/x/net/internal/nettest"
	"golang.org/x/net/ipv6"
)

func BenchmarkReadWriteUnicast(b *testing.B) {
	c, err := nettest.NewLocalPacketListener("udp6")
	if err != nil {
		b.Skipf("not supported on %s/%s: %v", runtime.GOOS, runtime.GOARCH, err)
	}
	defer c.Close()

	dst := c.LocalAddr()
	wb, rb := []byte("HELLO-R-U-THERE"), make([]byte, 128)

	b.Run("NetUDP", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if _, err := c.WriteTo(wb, dst); err != nil {
				b.Fatal(err)
			}
			if _, _, err := c.ReadFrom(rb); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("IPv6UDP", func(b *testing.B) {
		p := ipv6.NewPacketConn(c)
		cf := ipv6.FlagTrafficClass | ipv6.FlagHopLimit | ipv6.FlagSrc | ipv6.FlagDst | ipv6.FlagInterface | ipv6.FlagPathMTU
		if err := p.SetControlMessage(cf, true); err != nil {
			b.Fatal(err)
		}
		cm := ipv6.ControlMessage{
			TrafficClass: iana.DiffServAF11 | iana.CongestionExperienced,
			HopLimit:     1,
		}
		ifi := nettest.RoutedInterface("ip6", net.FlagUp|net.FlagLoopback)
		if ifi != nil {
			cm.IfIndex = ifi.Index
		}

		for i := 0; i < b.N; i++ {
			if _, err := p.WriteTo(wb, &cm, dst); err != nil {
				b.Fatal(err)
			}
			if _, _, _, err := p.ReadFrom(rb); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func TestPacketConnConcurrentReadWriteUnicastUDP(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !supportsIPv6 {
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
	ifi := nettest.RoutedInterface("ip6", net.FlagUp|net.FlagLoopback)
	cf := ipv6.FlagTrafficClass | ipv6.FlagHopLimit | ipv6.FlagSrc | ipv6.FlagDst | ipv6.FlagInterface | ipv6.FlagPathMTU
	wb := []byte("HELLO-R-U-THERE")

	if err := p.SetControlMessage(cf, true); err != nil { // probe before test
		if nettest.ProtocolNotSupported(err) {
			t.Skipf("not supported on %s", runtime.GOOS)
		}
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	reader := func() {
		defer wg.Done()
		rb := make([]byte, 128)
		if n, cm, _, err := p.ReadFrom(rb); err != nil {
			t.Error(err)
			return
		} else if !bytes.Equal(rb[:n], wb) {
			t.Errorf("got %v; want %v", rb[:n], wb)
			return
		} else {
			s := cm.String()
			if strings.Contains(s, ",") {
				t.Errorf("should be space-separated values: %s", s)
			}
		}
	}
	writer := func(toggle bool) {
		defer wg.Done()
		cm := ipv6.ControlMessage{
			TrafficClass: iana.DiffServAF11 | iana.CongestionExperienced,
			Src:          net.IPv6loopback,
		}
		if ifi != nil {
			cm.IfIndex = ifi.Index
		}
		if err := p.SetControlMessage(cf, toggle); err != nil {
			t.Error(err)
			return
		}
		if n, err := p.WriteTo(wb, &cm, dst); err != nil {
			t.Error(err)
			return
		} else if n != len(wb) {
			t.Errorf("got %d; want %d", n, len(wb))
			return
		}
	}

	const N = 10
	wg.Add(N)
	for i := 0; i < N; i++ {
		go reader()
	}
	wg.Add(2 * N)
	for i := 0; i < 2*N; i++ {
		go writer(i%2 != 0)
	}
	wg.Add(N)
	for i := 0; i < N; i++ {
		go reader()
	}
	wg.Wait()
}
