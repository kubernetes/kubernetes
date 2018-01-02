// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.9
// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package socket_test

import (
	"bytes"
	"fmt"
	"net"
	"runtime"
	"testing"

	"golang.org/x/net/internal/nettest"
	"golang.org/x/net/internal/socket"
)

type mockControl struct {
	Level int
	Type  int
	Data  []byte
}

func TestControlMessage(t *testing.T) {
	for _, tt := range []struct {
		cs []mockControl
	}{
		{
			[]mockControl{
				{Level: 1, Type: 1},
			},
		},
		{
			[]mockControl{
				{Level: 2, Type: 2, Data: []byte{0xfe}},
			},
		},
		{
			[]mockControl{
				{Level: 3, Type: 3, Data: []byte{0xfe, 0xff, 0xff, 0xfe}},
			},
		},
		{
			[]mockControl{
				{Level: 4, Type: 4, Data: []byte{0xfe, 0xff, 0xff, 0xfe, 0xfe, 0xff, 0xff, 0xfe}},
			},
		},
		{
			[]mockControl{
				{Level: 4, Type: 4, Data: []byte{0xfe, 0xff, 0xff, 0xfe, 0xfe, 0xff, 0xff, 0xfe}},
				{Level: 2, Type: 2, Data: []byte{0xfe}},
			},
		},
	} {
		var w []byte
		var tailPadLen int
		mm := socket.NewControlMessage([]int{0})
		for i, c := range tt.cs {
			m := socket.NewControlMessage([]int{len(c.Data)})
			l := len(m) - len(mm)
			if i == len(tt.cs)-1 && l > len(c.Data) {
				tailPadLen = l - len(c.Data)
			}
			w = append(w, m...)
		}

		var err error
		ww := make([]byte, len(w))
		copy(ww, w)
		m := socket.ControlMessage(ww)
		for _, c := range tt.cs {
			if err = m.MarshalHeader(c.Level, c.Type, len(c.Data)); err != nil {
				t.Fatalf("(%v).MarshalHeader() = %v", tt.cs, err)
			}
			copy(m.Data(len(c.Data)), c.Data)
			m = m.Next(len(c.Data))
		}
		m = socket.ControlMessage(w)
		for _, c := range tt.cs {
			m, err = m.Marshal(c.Level, c.Type, c.Data)
			if err != nil {
				t.Fatalf("(%v).Marshal() = %v", tt.cs, err)
			}
		}
		if !bytes.Equal(ww, w) {
			t.Fatalf("got %#v; want %#v", ww, w)
		}

		ws := [][]byte{w}
		if tailPadLen > 0 {
			// Test a message with no tail padding.
			nopad := w[:len(w)-tailPadLen]
			ws = append(ws, [][]byte{nopad}...)
		}
		for _, w := range ws {
			ms, err := socket.ControlMessage(w).Parse()
			if err != nil {
				t.Fatalf("(%v).Parse() = %v", tt.cs, err)
			}
			for i, m := range ms {
				lvl, typ, dataLen, err := m.ParseHeader()
				if err != nil {
					t.Fatalf("(%v).ParseHeader() = %v", tt.cs, err)
				}
				if lvl != tt.cs[i].Level || typ != tt.cs[i].Type || dataLen != len(tt.cs[i].Data) {
					t.Fatalf("%v: got %d, %d, %d; want %d, %d, %d", tt.cs[i], lvl, typ, dataLen, tt.cs[i].Level, tt.cs[i].Type, len(tt.cs[i].Data))
				}
			}
		}
	}
}

func TestUDP(t *testing.T) {
	c, err := nettest.NewLocalPacketListener("udp")
	if err != nil {
		t.Skipf("not supported on %s/%s: %v", runtime.GOOS, runtime.GOARCH, err)
	}
	defer c.Close()

	t.Run("Message", func(t *testing.T) {
		testUDPMessage(t, c.(net.Conn))
	})
	switch runtime.GOOS {
	case "linux":
		t.Run("Messages", func(t *testing.T) {
			testUDPMessages(t, c.(net.Conn))
		})
	}
}

func testUDPMessage(t *testing.T, c net.Conn) {
	cc, err := socket.NewConn(c)
	if err != nil {
		t.Fatal(err)
	}
	data := []byte("HELLO-R-U-THERE")
	wm := socket.Message{
		Buffers: bytes.SplitAfter(data, []byte("-")),
		Addr:    c.LocalAddr(),
	}
	if err := cc.SendMsg(&wm, 0); err != nil {
		t.Fatal(err)
	}
	b := make([]byte, 32)
	rm := socket.Message{
		Buffers: [][]byte{b[:1], b[1:3], b[3:7], b[7:11], b[11:]},
	}
	if err := cc.RecvMsg(&rm, 0); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(b[:rm.N], data) {
		t.Fatalf("got %#v; want %#v", b[:rm.N], data)
	}
}

func testUDPMessages(t *testing.T, c net.Conn) {
	cc, err := socket.NewConn(c)
	if err != nil {
		t.Fatal(err)
	}
	data := []byte("HELLO-R-U-THERE")
	wmbs := bytes.SplitAfter(data, []byte("-"))
	wms := []socket.Message{
		{Buffers: wmbs[:1], Addr: c.LocalAddr()},
		{Buffers: wmbs[1:], Addr: c.LocalAddr()},
	}
	n, err := cc.SendMsgs(wms, 0)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(wms) {
		t.Fatalf("got %d; want %d", n, len(wms))
	}
	b := make([]byte, 32)
	rmbs := [][][]byte{{b[:len(wmbs[0])]}, {b[len(wmbs[0]):]}}
	rms := []socket.Message{
		{Buffers: rmbs[0]},
		{Buffers: rmbs[1]},
	}
	n, err = cc.RecvMsgs(rms, 0)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(rms) {
		t.Fatalf("got %d; want %d", n, len(rms))
	}
	nn := 0
	for i := 0; i < n; i++ {
		nn += rms[i].N
	}
	if !bytes.Equal(b[:nn], data) {
		t.Fatalf("got %#v; want %#v", b[:nn], data)
	}
}

func BenchmarkUDP(b *testing.B) {
	c, err := nettest.NewLocalPacketListener("udp")
	if err != nil {
		b.Skipf("not supported on %s/%s: %v", runtime.GOOS, runtime.GOARCH, err)
	}
	defer c.Close()
	cc, err := socket.NewConn(c.(net.Conn))
	if err != nil {
		b.Fatal(err)
	}
	data := []byte("HELLO-R-U-THERE")
	wm := socket.Message{
		Buffers: [][]byte{data},
		Addr:    c.LocalAddr(),
	}
	rm := socket.Message{
		Buffers: [][]byte{make([]byte, 128)},
		OOB:     make([]byte, 128),
	}

	for M := 1; M <= 1<<9; M = M << 1 {
		b.Run(fmt.Sprintf("Iter-%d", M), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for j := 0; j < M; j++ {
					if err := cc.SendMsg(&wm, 0); err != nil {
						b.Fatal(err)
					}
					if err := cc.RecvMsg(&rm, 0); err != nil {
						b.Fatal(err)
					}
				}
			}
		})
		switch runtime.GOOS {
		case "linux":
			wms := make([]socket.Message, M)
			for i := range wms {
				wms[i].Buffers = [][]byte{data}
				wms[i].Addr = c.LocalAddr()
			}
			rms := make([]socket.Message, M)
			for i := range rms {
				rms[i].Buffers = [][]byte{make([]byte, 128)}
				rms[i].OOB = make([]byte, 128)
			}
			b.Run(fmt.Sprintf("Batch-%d", M), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					if _, err := cc.SendMsgs(wms, 0); err != nil {
						b.Fatal(err)
					}
					if _, err := cc.RecvMsgs(rms, 0); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
