// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4_test

import (
	"bytes"
	"net"
	"runtime"
	"strings"
	"sync"
	"testing"

	"golang.org/x/net/internal/nettest"
	"golang.org/x/net/ipv4"
)

func BenchmarkReadWriteUnicast(b *testing.B) {
	c, err := nettest.NewLocalPacketListener("udp4")
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
	b.Run("IPv4UDP", func(b *testing.B) {
		p := ipv4.NewPacketConn(c)
		cf := ipv4.FlagTTL | ipv4.FlagInterface
		if err := p.SetControlMessage(cf, true); err != nil {
			b.Fatal(err)
		}
		cm := ipv4.ControlMessage{TTL: 1}
		ifi := nettest.RoutedInterface("ip4", net.FlagUp|net.FlagLoopback)
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

	c, err := nettest.NewLocalPacketListener("udp4")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	p := ipv4.NewPacketConn(c)
	defer p.Close()

	dst := c.LocalAddr()
	ifi := nettest.RoutedInterface("ip4", net.FlagUp|net.FlagLoopback)
	cf := ipv4.FlagTTL | ipv4.FlagSrc | ipv4.FlagDst | ipv4.FlagInterface
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
		cm := ipv4.ControlMessage{
			Src: net.IPv4(127, 0, 0, 1),
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
