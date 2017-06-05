// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rangetable provides utilities for creating and inspecting
// unicode.RangeTables.
package rangetable

import (
	"sort"
	"unicode"
)

// New creates a RangeTable from the given runes, which may contain duplicates.
func New(r ...rune) *unicode.RangeTable {
	if len(r) == 0 {
		return &unicode.RangeTable{}
	}

	sort.Sort(byRune(r))

	// Remove duplicates.
	k := 1
	for i := 1; i < len(r); i++ {
		if r[k-1] != r[i] {
			r[k] = r[i]
			k++
		}
	}

	var rt unicode.RangeTable
	for _, r := range r[:k] {
		if r <= 0xFFFF {
			rt.R16 = append(rt.R16, unicode.Range16{Lo: uint16(r), Hi: uint16(r), Stride: 1})
		} else {
			rt.R32 = append(rt.R32, unicode.Range32{Lo: uint32(r), Hi: uint32(r), Stride: 1})
		}
	}

	// Optimize RangeTable.
	return Merge(&rt)
}

type byRune []rune

func (r byRune) Len() int           { return len(r) }
func (r byRune) Swap(i, j int)      { r[i], r[j] = r[j], r[i] }
func (r byRune) Less(i, j int) bool { return r[i] < r[j] }

// Visit visits all runes in the given RangeTable in order, calling fn for each.
func Visit(rt *unicode.RangeTable, fn func(rune)) {
	for _, r16 := range rt.R16 {
		for r := rune(r16.Lo); r <= rune(r16.Hi); r += rune(r16.Stride) {
			fn(r)
		}
	}
	for _, r32 := range rt.R32 {
		for r := rune(r32.Lo); r <= rune(r32.Hi); r += rune(r32.Stride) {
			fn(r)
		}
	}
}

// Assigned returns a RangeTable with all assigned code points for a given
// Unicode version. This includes graphic, format, control, and private-use
// characters. It returns nil if the data for the given version is not
// available.
func Assigned(version string) *unicode.RangeTable {
	return assigned[version]
}
