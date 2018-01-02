// Copyright 2016 Frank Schroeder. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package properties

import (
	"reflect"
	"testing"
	"time"
)

func TestDecodeValues(t *testing.T) {
	type S struct {
		S   string
		BT  bool
		BF  bool
		I   int
		I8  int8
		I16 int16
		I32 int32
		I64 int64
		U   uint
		U8  uint8
		U16 uint16
		U32 uint32
		U64 uint64
		F32 float32
		F64 float64
		D   time.Duration
		TM  time.Time
	}
	in := `
	S=abc
	BT=true
	BF=false
	I=-1
	I8=-8
	I16=-16
	I32=-32
	I64=-64
	U=1
	U8=8
	U16=16
	U32=32
	U64=64
	F32=3.2
	F64=6.4
	D=5s
	TM=2015-01-02T12:34:56Z
	`
	out := &S{
		S:   "abc",
		BT:  true,
		BF:  false,
		I:   -1,
		I8:  -8,
		I16: -16,
		I32: -32,
		I64: -64,
		U:   1,
		U8:  8,
		U16: 16,
		U32: 32,
		U64: 64,
		F32: 3.2,
		F64: 6.4,
		D:   5 * time.Second,
		TM:  tm(t, time.RFC3339, "2015-01-02T12:34:56Z"),
	}
	testDecode(t, in, &S{}, out)
}

func TestDecodeValueDefaults(t *testing.T) {
	type S struct {
		S   string        `properties:",default=abc"`
		BT  bool          `properties:",default=true"`
		BF  bool          `properties:",default=false"`
		I   int           `properties:",default=-1"`
		I8  int8          `properties:",default=-8"`
		I16 int16         `properties:",default=-16"`
		I32 int32         `properties:",default=-32"`
		I64 int64         `properties:",default=-64"`
		U   uint          `properties:",default=1"`
		U8  uint8         `properties:",default=8"`
		U16 uint16        `properties:",default=16"`
		U32 uint32        `properties:",default=32"`
		U64 uint64        `properties:",default=64"`
		F32 float32       `properties:",default=3.2"`
		F64 float64       `properties:",default=6.4"`
		D   time.Duration `properties:",default=5s"`
		TM  time.Time     `properties:",default=2015-01-02T12:34:56Z"`
	}
	out := &S{
		S:   "abc",
		BT:  true,
		BF:  false,
		I:   -1,
		I8:  -8,
		I16: -16,
		I32: -32,
		I64: -64,
		U:   1,
		U8:  8,
		U16: 16,
		U32: 32,
		U64: 64,
		F32: 3.2,
		F64: 6.4,
		D:   5 * time.Second,
		TM:  tm(t, time.RFC3339, "2015-01-02T12:34:56Z"),
	}
	testDecode(t, "", &S{}, out)
}

func TestDecodeArrays(t *testing.T) {
	type S struct {
		S   []string
		B   []bool
		I   []int
		I8  []int8
		I16 []int16
		I32 []int32
		I64 []int64
		U   []uint
		U8  []uint8
		U16 []uint16
		U32 []uint32
		U64 []uint64
		F32 []float32
		F64 []float64
		D   []time.Duration
		TM  []time.Time
	}
	in := `
	S=a;b
	B=true;false
	I=-1;-2
	I8=-8;-9
	I16=-16;-17
	I32=-32;-33
	I64=-64;-65
	U=1;2
	U8=8;9
	U16=16;17
	U32=32;33
	U64=64;65
	F32=3.2;3.3
	F64=6.4;6.5
	D=4s;5s
	TM=2015-01-01T00:00:00Z;2016-01-01T00:00:00Z
	`
	out := &S{
		S:   []string{"a", "b"},
		B:   []bool{true, false},
		I:   []int{-1, -2},
		I8:  []int8{-8, -9},
		I16: []int16{-16, -17},
		I32: []int32{-32, -33},
		I64: []int64{-64, -65},
		U:   []uint{1, 2},
		U8:  []uint8{8, 9},
		U16: []uint16{16, 17},
		U32: []uint32{32, 33},
		U64: []uint64{64, 65},
		F32: []float32{3.2, 3.3},
		F64: []float64{6.4, 6.5},
		D:   []time.Duration{4 * time.Second, 5 * time.Second},
		TM:  []time.Time{tm(t, time.RFC3339, "2015-01-01T00:00:00Z"), tm(t, time.RFC3339, "2016-01-01T00:00:00Z")},
	}
	testDecode(t, in, &S{}, out)
}

func TestDecodeArrayDefaults(t *testing.T) {
	type S struct {
		S   []string        `properties:",default=a;b"`
		B   []bool          `properties:",default=true;false"`
		I   []int           `properties:",default=-1;-2"`
		I8  []int8          `properties:",default=-8;-9"`
		I16 []int16         `properties:",default=-16;-17"`
		I32 []int32         `properties:",default=-32;-33"`
		I64 []int64         `properties:",default=-64;-65"`
		U   []uint          `properties:",default=1;2"`
		U8  []uint8         `properties:",default=8;9"`
		U16 []uint16        `properties:",default=16;17"`
		U32 []uint32        `properties:",default=32;33"`
		U64 []uint64        `properties:",default=64;65"`
		F32 []float32       `properties:",default=3.2;3.3"`
		F64 []float64       `properties:",default=6.4;6.5"`
		D   []time.Duration `properties:",default=4s;5s"`
		TM  []time.Time     `properties:",default=2015-01-01T00:00:00Z;2016-01-01T00:00:00Z"`
	}
	out := &S{
		S:   []string{"a", "b"},
		B:   []bool{true, false},
		I:   []int{-1, -2},
		I8:  []int8{-8, -9},
		I16: []int16{-16, -17},
		I32: []int32{-32, -33},
		I64: []int64{-64, -65},
		U:   []uint{1, 2},
		U8:  []uint8{8, 9},
		U16: []uint16{16, 17},
		U32: []uint32{32, 33},
		U64: []uint64{64, 65},
		F32: []float32{3.2, 3.3},
		F64: []float64{6.4, 6.5},
		D:   []time.Duration{4 * time.Second, 5 * time.Second},
		TM:  []time.Time{tm(t, time.RFC3339, "2015-01-01T00:00:00Z"), tm(t, time.RFC3339, "2016-01-01T00:00:00Z")},
	}
	testDecode(t, "", &S{}, out)
}

func TestDecodeSkipUndef(t *testing.T) {
	type S struct {
		X     string `properties:"-"`
		Undef string `properties:",default=some value"`
	}
	in := `X=ignore`
	out := &S{"", "some value"}
	testDecode(t, in, &S{}, out)
}

func TestDecodeStruct(t *testing.T) {
	type A struct {
		S string
		T string `properties:"t"`
		U string `properties:"u,default=uuu"`
	}
	type S struct {
		A A
		B A `properties:"b"`
	}
	in := `
	A.S=sss
	A.t=ttt
	b.S=SSS
	b.t=TTT
	`
	out := &S{
		A{S: "sss", T: "ttt", U: "uuu"},
		A{S: "SSS", T: "TTT", U: "uuu"},
	}
	testDecode(t, in, &S{}, out)
}

func TestDecodeMap(t *testing.T) {
	type S struct {
		A string `properties:"a"`
	}
	type X struct {
		A map[string]string
		B map[string][]string
		C map[string]map[string]string
		D map[string]S
		E map[string]int
		F map[string]int `properties:"-"`
	}
	in := `
	A.foo=bar
	A.bar=bang
	B.foo=a;b;c
	B.bar=1;2;3
	C.foo.one=1
	C.foo.two=2
	C.bar.three=3
	C.bar.four=4
	D.foo.a=bar
	`
	out := &X{
		A: map[string]string{"foo": "bar", "bar": "bang"},
		B: map[string][]string{"foo": []string{"a", "b", "c"}, "bar": []string{"1", "2", "3"}},
		C: map[string]map[string]string{"foo": map[string]string{"one": "1", "two": "2"}, "bar": map[string]string{"three": "3", "four": "4"}},
		D: map[string]S{"foo": S{"bar"}},
		E: map[string]int{},
	}
	testDecode(t, in, &X{}, out)
}

func testDecode(t *testing.T, in string, v, out interface{}) {
	p, err := parse(in)
	if err != nil {
		t.Fatalf("got %v want nil", err)
	}
	if err := p.Decode(v); err != nil {
		t.Fatalf("got %v want nil", err)
	}
	if got, want := v, out; !reflect.DeepEqual(got, want) {
		t.Fatalf("\ngot  %+v\nwant %+v", got, want)
	}
}

func tm(t *testing.T, layout, s string) time.Time {
	tm, err := time.Parse(layout, s)
	if err != nil {
		t.Fatalf("got %v want nil", err)
	}
	return tm
}
