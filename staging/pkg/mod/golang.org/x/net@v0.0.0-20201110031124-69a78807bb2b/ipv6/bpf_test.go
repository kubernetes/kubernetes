// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6_test

import (
	"net"
	"runtime"
	"testing"
	"time"

	"golang.org/x/net/bpf"
	"golang.org/x/net/ipv6"
	"golang.org/x/net/nettest"
)

func TestBPF(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !nettest.SupportsIPv6() {
		t.Skip("ipv6 is not supported")
	}

	l, err := net.ListenPacket("udp6", "[::1]:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()

	p := ipv6.NewPacketConn(l)

	// This filter accepts UDP packets whose first payload byte is
	// even.
	prog, err := bpf.Assemble([]bpf.Instruction{
		// Load the first byte of the payload (skipping UDP header).
		bpf.LoadAbsolute{Off: 8, Size: 1},
		// Select LSB of the byte.
		bpf.ALUOpConstant{Op: bpf.ALUOpAnd, Val: 1},
		// Byte is even?
		bpf.JumpIf{Cond: bpf.JumpEqual, Val: 0, SkipFalse: 1},
		// Accept.
		bpf.RetConstant{Val: 4096},
		// Ignore.
		bpf.RetConstant{Val: 0},
	})
	if err != nil {
		t.Fatalf("compiling BPF: %s", err)
	}

	if err = p.SetBPF(prog); err != nil {
		t.Fatalf("attaching filter to Conn: %s", err)
	}

	s, err := net.Dial("udp6", l.LocalAddr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	go func() {
		for i := byte(0); i < 10; i++ {
			s.Write([]byte{i})
		}
	}()

	l.SetDeadline(time.Now().Add(2 * time.Second))
	seen := make([]bool, 5)
	for {
		var b [512]byte
		n, _, err := l.ReadFrom(b[:])
		if err != nil {
			t.Fatalf("reading from listener: %s", err)
		}
		if n != 1 {
			t.Fatalf("unexpected packet length, want 1, got %d", n)
		}
		if b[0] >= 10 {
			t.Fatalf("unexpected byte, want 0-9, got %d", b[0])
		}
		if b[0]%2 != 0 {
			t.Fatalf("got odd byte %d, wanted only even bytes", b[0])
		}
		seen[b[0]/2] = true

		seenAll := true
		for _, v := range seen {
			if !v {
				seenAll = false
				break
			}
		}
		if seenAll {
			break
		}
	}
}
