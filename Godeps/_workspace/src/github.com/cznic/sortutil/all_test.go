// Copyright 2014 The sortutil Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sortutil

import (
	"fmt"
	"math"
	"path"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/cznic/mathutil"
)

func dbg(s string, va ...interface{}) {
	if s == "" {
		s = strings.Repeat("%v ", len(va))
	}
	_, fn, fl, _ := runtime.Caller(1)
	fmt.Printf("dbg %s:%d: ", path.Base(fn), fl)
	fmt.Printf(s, va...)
	fmt.Println()
}

func caller(s string, va ...interface{}) {
	_, fn, fl, _ := runtime.Caller(2)
	fmt.Printf("caller: %s:%d: ", path.Base(fn), fl)
	fmt.Printf(s, va...)
	fmt.Println()
	_, fn, fl, _ = runtime.Caller(1)
	fmt.Printf("\tcallee: %s:%d: ", path.Base(fn), fl)
	fmt.Println()
}

func use(...interface{}) {}

func TestByteSlice(t *testing.T) {
	const N = 1e4
	s := make(ByteSlice, N)
	for i := range s {
		s[i] = byte(i) ^ 0x55
	}
	s.Sort()
	if !sort.IsSorted(s) {
		t.Fatal(false)
	}
}

func TestSearchBytes(t *testing.T) {
	const N = 1e1
	s := make(ByteSlice, N)
	for i := range s {
		s[i] = byte(2 * i)
	}
	if g, e := SearchBytes(s, 12), 6; g != e {
		t.Fatal(g, e)
	}
}

func TestFloat32Slice(t *testing.T) {
	const N = 1e4
	s := make(Float32Slice, N)
	for i := range s {
		s[i] = float32(i ^ 0x55aa55aa)
	}
	s.Sort()
	if !sort.IsSorted(s) {
		t.Fatal(false)
	}
}

func TestSearchFloat32s(t *testing.T) {
	const N = 1e4
	s := make(Float32Slice, N)
	for i := range s {
		s[i] = float32(2 * i)
	}
	if g, e := SearchFloat32s(s, 12), 6; g != e {
		t.Fatal(g, e)
	}
}

func TestInt8Slice(t *testing.T) {
	const N = 1e4
	s := make(Int8Slice, N)
	for i := range s {
		s[i] = int8(i) ^ 0x55
	}
	s.Sort()
	if !sort.IsSorted(s) {
		t.Fatal(false)
	}
}

func TestSearchInt8s(t *testing.T) {
	const N = 1e1
	s := make(Int8Slice, N)
	for i := range s {
		s[i] = int8(2 * i)
	}
	if g, e := SearchInt8s(s, 12), 6; g != e {
		t.Fatal(g, e)
	}
}

func TestInt16Slice(t *testing.T) {
	const N = 1e4
	s := make(Int16Slice, N)
	for i := range s {
		s[i] = int16(i) ^ 0x55aa
	}
	s.Sort()
	if !sort.IsSorted(s) {
		t.Fatal(false)
	}
}

func TestSearchInt16s(t *testing.T) {
	const N = 1e4
	s := make(Int16Slice, N)
	for i := range s {
		s[i] = int16(2 * i)
	}
	if g, e := SearchInt16s(s, 12), 6; g != e {
		t.Fatal(g, e)
	}
}

func TestInt32Slice(t *testing.T) {
	const N = 1e4
	s := make(Int32Slice, N)
	for i := range s {
		s[i] = int32(i) ^ 0x55aa55aa
	}
	s.Sort()
	if !sort.IsSorted(s) {
		t.Fatal(false)
	}
}

func TestSearchInt32s(t *testing.T) {
	const N = 1e4
	s := make(Int32Slice, N)
	for i := range s {
		s[i] = int32(2 * i)
	}
	if g, e := SearchInt32s(s, 12), 6; g != e {
		t.Fatal(g, e)
	}
}

func TestInt64Slice(t *testing.T) {
	const N = 1e4
	s := make(Int64Slice, N)
	for i := range s {
		s[i] = int64(i) ^ 0x55aa55aa55aa55aa
	}
	s.Sort()
	if !sort.IsSorted(s) {
		t.Fatal(false)
	}
}

func TestSearchInt64s(t *testing.T) {
	const N = 1e4
	s := make(Int64Slice, N)
	for i := range s {
		s[i] = int64(2 * i)
	}
	if g, e := SearchInt64s(s, 12), 6; g != e {
		t.Fatal(g, e)
	}
}

func TestUintSlice(t *testing.T) {
	const N = 1e4
	s := make(UintSlice, N)
	for i := range s {
		s[i] = uint(i) ^ 0x55aa55aa
	}
	s.Sort()
	if !sort.IsSorted(s) {
		t.Fatal(false)
	}
}

func TestSearchUints(t *testing.T) {
	const N = 1e4
	s := make(UintSlice, N)
	for i := range s {
		s[i] = uint(2 * i)
	}
	if g, e := SearchUints(s, 12), 6; g != e {
		t.Fatal(g, e)
	}
}

func TestUint16Slice(t *testing.T) {
	const N = 1e4
	s := make(Uint16Slice, N)
	for i := range s {
		s[i] = uint16(i) ^ 0x55aa
	}
	s.Sort()
	if !sort.IsSorted(s) {
		t.Fatal(false)
	}
}

func TestSearchUint16s(t *testing.T) {
	const N = 1e4
	s := make(Uint16Slice, N)
	for i := range s {
		s[i] = uint16(2 * i)
	}
	if g, e := SearchUint16s(s, 12), 6; g != e {
		t.Fatal(g, e)
	}
}

func TestUint32Slice(t *testing.T) {
	const N = 1e4
	s := make(Uint32Slice, N)
	for i := range s {
		s[i] = uint32(i) ^ 0x55aa55aa
	}
	s.Sort()
	if !sort.IsSorted(s) {
		t.Fatal(false)
	}
}

func TestSearchUint32s(t *testing.T) {
	const N = 1e4
	s := make(Uint32Slice, N)
	for i := range s {
		s[i] = uint32(2 * i)
	}
	if g, e := SearchUint32s(s, 12), 6; g != e {
		t.Fatal(g, e)
	}
}

func TestUint64Slice(t *testing.T) {
	const N = 1e4
	s := make(Uint64Slice, N)
	for i := range s {
		s[i] = uint64(i) ^ 0x55aa55aa55aa55aa
	}
	s.Sort()
	if !sort.IsSorted(s) {
		t.Fatal(false)
	}
}

func TestSearchUint64s(t *testing.T) {
	const N = 1e4
	s := make(Uint64Slice, N)
	for i := range s {
		s[i] = uint64(2 * i)
	}
	if g, e := SearchUint64s(s, 12), 6; g != e {
		t.Fatal(g, e)
	}
}

func TestRuneSlice(t *testing.T) {
	const N = 1e4
	s := make(RuneSlice, N)
	for i := range s {
		s[i] = rune(i ^ 0x55aa55aa)
	}
	s.Sort()
	if !sort.IsSorted(s) {
		t.Fatal(false)
	}
}

func TestSearchRunes(t *testing.T) {
	const N = 1e4
	s := make(RuneSlice, N)
	for i := range s {
		s[i] = rune(2 * i)
	}
	if g, e := SearchRunes(s, rune('\x0c')), 6; g != e {
		t.Fatal(g, e)
	}
}

func dedupe(a []int) (r []int) {
	a = append([]int(nil), a...)
	if len(a) < 2 {
		return a
	}

	sort.Ints(a)
	if a[0] < 0 {
		panic("internal error")
	}

	last := -1
	for _, v := range a {
		if v != last {
			r = append(r, v)
			last = v
		}
	}
	return r
}

func TestDedup(t *testing.T) {
	a := []int{}
	n := Dedupe(sort.IntSlice(a))
	if g, e := n, 0; g != e {
		t.Fatal(g, e)
	}

	if g, e := len(a), 0; g != e {
		t.Fatal(g, e)
	}

	for c := 1; c <= 7; c++ {
		in := make([]int, c)
		lim := int(mathutil.ModPowUint32(uint32(c), uint32(c), math.MaxUint32))
		for n := 0; n < lim; n++ {
			m := n
			for i := range in {
				in[i] = m % c
				m /= c
			}
			in0 := append([]int(nil), in...)
			out0 := dedupe(in)
			n := Dedupe(sort.IntSlice(in))
			if g, e := n, len(out0); g != e {
				t.Fatalf("n %d, exp %d, in0 %v, in %v, out0 %v", g, e, in0, in, out0)
			}

			for i, v := range out0 {
				if g, e := in[i], v; g != e {
					t.Fatalf("n %d, in0 %v, in %v, out0 %v", n, in0, in, out0)
				}
			}
		}
	}
}

func ExampleDedupe() {
	a := []int{4, 1, 2, 1, 3, 4, 2}
	fmt.Println(a[:Dedupe(sort.IntSlice(a))])

	b := []string{"foo", "bar", "baz", "bar", "foo", "qux", "qux"}
	fmt.Println(b[:Dedupe(sort.StringSlice(b))])
	// Output:
	// [1 2 3 4]
	// [bar baz foo qux]
}
