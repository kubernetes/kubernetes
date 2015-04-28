// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lldb

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"path"
	"runtime"
	"strings"
	"testing"
)

var dbg = func(s string, va ...interface{}) {
	_, fn, fl, _ := runtime.Caller(1)
	fmt.Printf("%s:%d: ", path.Base(fn), fl)
	fmt.Printf(s+"\n", va...)
}

func use(...interface{}) {}

func TestN2Atoms(t *testing.T) {
	tab := []struct{ n, a int }{
		{0, 1},
		{1, 1},
		{2, 1},
		{3, 1},
		{4, 1},
		{5, 1},
		{6, 1},
		{7, 1},
		{8, 1},
		{9, 1},
		{10, 1},
		{11, 1},
		{12, 1},
		{13, 1},
		{14, 1},

		{15, 2},
		{16, 2},
		{17, 2},
		{18, 2},
		{19, 2},
		{20, 2},
		{21, 2},
		{22, 2},
		{23, 2},
		{24, 2},
		{25, 2},
		{26, 2},
		{27, 2},
		{28, 2},
		{29, 2},
		{30, 2},

		{31, 3},

		{252, 16},
		{253, 17},
		{254, 17},
		{255, 17},
		{256, 17},
		{257, 17},
		{258, 17},
		{259, 17},
		{260, 17},
		{261, 17},
		{262, 17},
		{263, 17},
		{264, 17},
		{265, 17},
		{266, 17},
		{267, 17},
		{268, 17},
		{269, 18},
		{65532, 4096},
		{65533, 4097},
		{65787, 4112},
	}

	for i, test := range tab {
		if g, e := n2atoms(test.n), test.a; g != e {
			t.Errorf("(%d) %d %d %d", i, test.n, g, e)
		}
	}
}

func TestN2Padding(t *testing.T) {
	tab := []struct{ n, p int }{
		{0, 14},
		{1, 13},
		{2, 12},
		{3, 11},
		{4, 10},
		{5, 9},
		{6, 8},
		{7, 7},
		{8, 6},
		{9, 5},
		{10, 4},
		{11, 3},
		{12, 2},
		{13, 1},
		{14, 0},

		{15, 15},
		{16, 14},
		{17, 13},
		{18, 12},
		{19, 11},
		{20, 10},
		{21, 9},
		{22, 8},
		{23, 7},
		{24, 6},
		{25, 5},
		{26, 4},
		{27, 3},
		{28, 2},
		{29, 1},
		{30, 0},

		{31, 15},

		{252, 0},
		{253, 15},
		{254, 14},
		{255, 13},
		{256, 12},
		{257, 11},
		{258, 10},
		{259, 9},
		{260, 8},
		{261, 7},
		{262, 6},
		{263, 5},
		{264, 4},
		{265, 3},
		{266, 2},
		{267, 1},
		{268, 0},
		{269, 15},
	}

	for i, test := range tab {
		if g, e := n2padding(test.n), test.p; g != e {
			t.Errorf("(%d) %d %d %d", i, test.n, g, e)
		}
	}
}

func TestH2Off(t *testing.T) {
	tab := []struct{ h, off int64 }{
		{-1, fltSz - 32},
		{0, fltSz - 16},
		{1, fltSz + 0},
		{2, fltSz + 16},
		{3, fltSz + 32},
	}

	for i, test := range tab {
		if g, e := h2off(test.h), test.off; g != e {
			t.Error("h2off", i, g, e)
		}
		if g, e := off2h(test.off), test.h; g != e {
			t.Error("off2h", i, g, e)
		}
	}
}

func TestB2H(t *testing.T) {
	tab := []struct {
		b []byte
		h int64
	}{
		{[]byte{0, 0, 0, 0, 0, 0, 0}, 0},
		{[]byte{0, 0, 0, 0, 0, 0, 1}, 1},
		{[]byte{0, 0, 0, 0, 0, 0, 1, 2}, 1},
		{[]byte{0, 0, 0, 0, 0, 0x32, 0x10}, 0x3210},
		{[]byte{0, 0, 0, 0, 0x54, 0x32, 0x10}, 0x543210},
		{[]byte{0, 0, 0, 0x76, 0x54, 0x32, 0x10}, 0x76543210},
		{[]byte{0, 0, 0x98, 0x76, 0x54, 0x32, 0x10}, 0x9876543210},
		{[]byte{0, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10}, 0xba9876543210},
		{[]byte{0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10}, 0xdcba9876543210},
	}

	for i, test := range tab {
		if g, e := b2h(test.b), test.h; g != e {
			t.Errorf("b2h: %d %#8x %#8x", i, g, e)
		}
		var g [7]byte
		h2b(g[:], test.h)
		if e := test.b; !bytes.Equal(g[:], e[:7]) {
			t.Errorf("b2h: %d g: % 0x e: % 0x", i, g, e)
		}
	}
}

func s2b(s string) []byte {
	if s == "" {
		return nil
	}

	s = strings.Replace(s, " ", "", -1)
	if n := len(s) & 1; n != 0 {
		panic(n)
	}
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}
