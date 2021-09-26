// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package catmsg

import (
	"fmt"
	"testing"
)

func TestEncodeUint(t *testing.T) {
	testCases := []struct {
		x   uint64
		enc string
	}{
		{0, "\x00"},
		{1, "\x01"},
		{2, "\x02"},
		{0x7f, "\x7f"},
		{0x80, "\x80\x01"},
		{1 << 14, "\x80\x80\x01"},
		{0xffffffff, "\xff\xff\xff\xff\x0f"},
		{0xffffffffffffffff, "\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01"},
	}
	for _, tc := range testCases {
		buf := [maxVarintBytes]byte{}
		got := string(buf[:encodeUint(buf[:], tc.x)])
		if got != tc.enc {
			t.Errorf("EncodeUint(%#x) = %q; want %q", tc.x, got, tc.enc)
		}
	}
}

func TestDecodeUint(t *testing.T) {
	testCases := []struct {
		x    uint64
		size int
		enc  string
		err  error
	}{{
		x:    0,
		size: 0,
		enc:  "",
		err:  errIllegalVarint,
	}, {
		x:    0,
		size: 1,
		enc:  "\x80",
		err:  errIllegalVarint,
	}, {
		x:    0,
		size: 3,
		enc:  "\x80\x80\x80",
		err:  errIllegalVarint,
	}, {
		x:    0,
		size: 1,
		enc:  "\x00",
	}, {
		x:    1,
		size: 1,
		enc:  "\x01",
	}, {
		x:    2,
		size: 1,
		enc:  "\x02",
	}, {
		x:    0x7f,
		size: 1,
		enc:  "\x7f",
	}, {
		x:    0x80,
		size: 2,
		enc:  "\x80\x01",
	}, {
		x:    1 << 14,
		size: 3,
		enc:  "\x80\x80\x01",
	}, {
		x:    0xffffffff,
		size: 5,
		enc:  "\xff\xff\xff\xff\x0f",
	}, {
		x:    0xffffffffffffffff,
		size: 10,
		enc:  "\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01",
	}, {
		x:    0xffffffffffffffff,
		size: 10,
		enc:  "\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\x00",
	}, {
		x:    0,
		size: 10,
		enc:  "\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01",
		err:  errVarintTooLarge,
	}}
	forms := []struct {
		name   string
		decode func(s string) (x uint64, size int, err error)
	}{
		{"decode", func(s string) (x uint64, size int, err error) {
			return decodeUint([]byte(s))
		}},
		{"decodeString", decodeUintString},
	}
	for _, f := range forms {
		for _, tc := range testCases {
			t.Run(fmt.Sprintf("%s:%q", f.name, tc.enc), func(t *testing.T) {
				x, size, err := f.decode(tc.enc)
				if err != tc.err {
					t.Errorf("err = %q; want %q", err, tc.err)
				}
				if size != tc.size {
					t.Errorf("size = %d; want %d", size, tc.size)
				}
				if x != tc.x {
					t.Errorf("decode = %#x; want %#x", x, tc.x)
				}
			})
		}
	}
}
