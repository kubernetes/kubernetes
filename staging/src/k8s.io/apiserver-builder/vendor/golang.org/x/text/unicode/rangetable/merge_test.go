// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rangetable

import (
	"testing"
	"unicode"
)

var (
	maxRuneTable = &unicode.RangeTable{
		R32: []unicode.Range32{
			{unicode.MaxRune, unicode.MaxRune, 1},
		},
	}

	overlap1 = &unicode.RangeTable{
		R16: []unicode.Range16{
			{0x100, 0xfffc, 4},
		},
		R32: []unicode.Range32{
			{0x100000, 0x10fffc, 4},
		},
	}

	overlap2 = &unicode.RangeTable{
		R16: []unicode.Range16{
			{0x101, 0xfffd, 4},
		},
		R32: []unicode.Range32{
			{0x100001, 0x10fffd, 3},
		},
	}

	// The following table should be compacted into two entries for R16 and R32.
	optimize = &unicode.RangeTable{
		R16: []unicode.Range16{
			{0x1, 0x1, 1},
			{0x2, 0x2, 1},
			{0x3, 0x3, 1},
			{0x5, 0x5, 1},
			{0x7, 0x7, 1},
			{0x9, 0x9, 1},
			{0xb, 0xf, 2},
		},
		R32: []unicode.Range32{
			{0x10001, 0x10001, 1},
			{0x10002, 0x10002, 1},
			{0x10003, 0x10003, 1},
			{0x10005, 0x10005, 1},
			{0x10007, 0x10007, 1},
			{0x10009, 0x10009, 1},
			{0x1000b, 0x1000f, 2},
		},
	}
)

func TestMerge(t *testing.T) {
	for i, tt := range [][]*unicode.RangeTable{
		{unicode.Cc, unicode.Cf},
		{unicode.L, unicode.Ll},
		{unicode.L, unicode.Ll, unicode.Lu},
		{unicode.Ll, unicode.Lu},
		{unicode.M},
		unicode.GraphicRanges,
		cased,

		// Merge R16 only and R32 only and vice versa.
		{unicode.Khmer, unicode.Khudawadi},
		{unicode.Imperial_Aramaic, unicode.Radical},

		// Merge with empty.
		{&unicode.RangeTable{}},
		{&unicode.RangeTable{}, &unicode.RangeTable{}},
		{&unicode.RangeTable{}, &unicode.RangeTable{}, &unicode.RangeTable{}},
		{&unicode.RangeTable{}, unicode.Hiragana},
		{unicode.Inherited, &unicode.RangeTable{}},
		{&unicode.RangeTable{}, unicode.Hanunoo, &unicode.RangeTable{}},

		// Hypothetical tables.
		{maxRuneTable},
		{overlap1, overlap2},

		// Optimization
		{optimize},
	} {
		rt := Merge(tt...)
		for r := rune(0); r <= unicode.MaxRune; r++ {
			if got, want := unicode.Is(rt, r), unicode.In(r, tt...); got != want {
				t.Fatalf("%d:%U: got %v; want %v", i, r, got, want)
			}
		}
		// Test optimization and correctness for R16.
		for k := 0; k < len(rt.R16)-1; k++ {
			if lo, hi := rt.R16[k].Lo, rt.R16[k].Hi; lo > hi {
				t.Errorf("%d: Lo (%x) > Hi (%x)", i, lo, hi)
			}
			if hi, lo := rt.R16[k].Hi, rt.R16[k+1].Lo; hi >= lo {
				t.Errorf("%d: Hi (%x) >= next Lo (%x)", i, hi, lo)
			}
			if rt.R16[k].Hi+rt.R16[k].Stride == rt.R16[k+1].Lo {
				t.Errorf("%d: missed optimization for R16 at %d between %X and %x",
					i, k, rt.R16[k], rt.R16[k+1])
			}
		}
		// Test optimization and correctness for R32.
		for k := 0; k < len(rt.R32)-1; k++ {
			if lo, hi := rt.R32[k].Lo, rt.R32[k].Hi; lo > hi {
				t.Errorf("%d: Lo (%x) > Hi (%x)", i, lo, hi)
			}
			if hi, lo := rt.R32[k].Hi, rt.R32[k+1].Lo; hi >= lo {
				t.Errorf("%d: Hi (%x) >= next Lo (%x)", i, hi, lo)
			}
			if rt.R32[k].Hi+rt.R32[k].Stride == rt.R32[k+1].Lo {
				t.Errorf("%d: missed optimization for R32 at %d between %X and %X",
					i, k, rt.R32[k], rt.R32[k+1])
			}
		}
	}
}

const runes = "Hello World in 2015!,\U0010fffd"

func BenchmarkNotMerged(t *testing.B) {
	for i := 0; i < t.N; i++ {
		for _, r := range runes {
			unicode.In(r, unicode.GraphicRanges...)
		}
	}
}

func BenchmarkMerged(t *testing.B) {
	rt := Merge(unicode.GraphicRanges...)

	for i := 0; i < t.N; i++ {
		for _, r := range runes {
			unicode.Is(rt, r)
		}
	}
}

var cased = []*unicode.RangeTable{
	unicode.Lower,
	unicode.Upper,
	unicode.Title,
	unicode.Other_Lowercase,
	unicode.Other_Uppercase,
}

func BenchmarkNotMergedCased(t *testing.B) {
	for i := 0; i < t.N; i++ {
		for _, r := range runes {
			unicode.In(r, cased...)
		}
	}
}

func BenchmarkMergedCased(t *testing.B) {
	// This reduces len(R16) from 243 to 82 and len(R32) from 65 to 35 for
	// Unicode 7.0.0.
	rt := Merge(cased...)

	for i := 0; i < t.N; i++ {
		for _, r := range runes {
			unicode.Is(rt, r)
		}
	}
}

func BenchmarkInit(t *testing.B) {
	for i := 0; i < t.N; i++ {
		Merge(cased...)
		Merge(unicode.GraphicRanges...)
	}
}

func BenchmarkInit2(t *testing.B) {
	// Hypothetical near-worst-case performance.
	for i := 0; i < t.N; i++ {
		Merge(overlap1, overlap2)
	}
}
