// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bitfield

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"testing"
)

type myUint8 uint8

type test1 struct { // 28 bits
	foo  uint16 `bitfield:",fob"`
	Bar  int8   `bitfield:"5,baz"`
	Foo  uint64
	bar  myUint8 `bitfield:"3"`
	Bool bool    `bitfield:""`
	Baz  int8    `bitfield:"3"`
}

type test2 struct {
	larger1 uint16 `bitfield:"32"`
	larger2 uint16 `bitfield:"32"`
}

type tooManyBits struct {
	u1 uint16 `bitfield:"12"`
	u2 uint16 `bitfield:"12"`
	u3 uint16 `bitfield:"12"`
	u4 uint16 `bitfield:"12"`
	u5 uint16 `bitfield:"12"`
	u6 uint16 `bitfield:"12"`
}

type just64 struct {
	foo uint64 `bitfield:""`
}

type toUint8 struct {
	foo bool `bitfield:""`
}

type toUint16 struct {
	foo int `bitfield:"9"`
}

type faultySize struct {
	foo uint64 `bitfield:"a"`
}

type faultyType struct {
	foo *int `bitfield:"5"`
}

var (
	maxed = test1{
		foo:  0xffff,
		Bar:  0x1f,
		Foo:  0xffff,
		bar:  0x7,
		Bool: true,
		Baz:  0x7,
	}
	alternate1 = test1{
		foo: 0xffff,
		bar: 0x7,
		Baz: 0x7,
	}
	alternate2 = test1{
		Bar:  0x1f,
		Bool: true,
	}
	overflow = test1{
		Bar: 0x3f,
	}
	negative = test1{
		Bar: -1,
	}
)

func TestPack(t *testing.T) {
	testCases := []struct {
		desc  string
		x     interface{}
		nBits uint
		out   uint64
		ok    bool
	}{
		{"maxed out fields", maxed, 0, 0xfffffff0, true},
		{"maxed using less bits", maxed, 28, 0x0fffffff, true},

		{"alternate1", alternate1, 0, 0xffff0770, true},
		{"alternate2", alternate2, 0, 0x0000f880, true},

		{"just64", &just64{0x0f0f0f0f}, 00, 0xf0f0f0f, true},
		{"just64", &just64{0x0f0f0f0f}, 64, 0xf0f0f0f, true},
		{"just64", &just64{0xffffFFFF}, 64, 0xffffffff, true},
		{"to uint8", &toUint8{true}, 0, 0x80, true},
		{"to uint16", &toUint16{1}, 0, 0x0080, true},
		// errors
		{"overflow", overflow, 0, 0, false},
		{"too many bits", &tooManyBits{}, 0, 0, false},
		{"fault size", &faultySize{}, 0, 0, false},
		{"fault type", &faultyType{}, 0, 0, false},
		{"negative", negative, 0, 0, false},
		{"not enough bits", maxed, 27, 0, false},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%T/%s", tc.x, tc.desc), func(t *testing.T) {
			v, err := Pack(tc.x, &Config{NumBits: tc.nBits})
			if ok := err == nil; v != tc.out || ok != tc.ok {
				t.Errorf("got %#x, %v; want %#x, %v (%v)", v, ok, tc.out, tc.ok, err)
			}
		})
	}
}

func TestRoundtrip(t *testing.T) {
	testCases := []struct {
		x test1
	}{
		{maxed},
		{alternate1},
		{alternate2},
	}
	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			v, err := Pack(tc.x, nil)
			if err != nil {
				t.Fatal(err)
			}
			want := tc.x
			want.Foo = 0 // not stored
			x := myInt(v)
			got := test1{
				foo:  x.fob(),
				Bar:  x.baz(),
				bar:  x.bar(),
				Bool: x.Bool(),
				Baz:  x.Baz(),
			}
			if got != want {
				t.Errorf("\ngot  %#v\nwant %#v (%#x)", got, want, v)
			}
		})
	}
}

func TestGen(t *testing.T) {
	testCases := []struct {
		desc   string
		x      interface{}
		config *Config
		ok     bool
		out    string
	}{{
		desc: "test1",
		x:    &test1{},
		ok:   true,
		out:  test1Gen,
	}, {
		desc:   "test1 with options",
		x:      &test1{},
		config: &Config{Package: "bitfield", TypeName: "myInt"},
		ok:     true,
		out:    mustRead("gen1_test.go"),
	}, {
		desc:   "test1 with alternative bits",
		x:      &test1{},
		config: &Config{NumBits: 28, Package: "bitfield", TypeName: "myInt2"},
		ok:     true,
		out:    mustRead("gen2_test.go"),
	}, {
		desc:   "failure",
		x:      &test1{},
		config: &Config{NumBits: 27}, // Too few bits.
		ok:     false,
		out:    "",
	}}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			w := &bytes.Buffer{}
			err := Gen(w, tc.x, tc.config)
			if ok := err == nil; ok != tc.ok {
				t.Fatalf("got %v; want %v (%v)", ok, tc.ok, err)
			}
			got := w.String()
			if got != tc.out {
				t.Errorf("got:\n%s\nwant:\n%s", got, tc.out)
			}
		})
	}
}

const test1Gen = `type test1 uint32

func (t test1) fob() uint16 {
	return uint16((t >> 16) & 0xffff)
}

func (t test1) baz() int8 {
	return int8((t >> 11) & 0x1f)
}

func (t test1) bar() myUint8 {
	return myUint8((t >> 8) & 0x7)
}

func (t test1) Bool() bool {
	const bit = 1 << 7
	return t&bit == bit
}

func (t test1) Baz() int8 {
	return int8((t >> 4) & 0x7)
}
`

func mustRead(filename string) string {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(err)
	}
	return string(b)
}
