// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.10
// +build go1.10

package idna

import (
	"testing"
	"unicode"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/internal/ucd"
)

func TestTables(t *testing.T) {
	testtext.SkipIfNotLong(t)

	lookup := func(r rune) info {
		v, _ := trie.lookupString(string(r))
		return info(v)
	}

	ucd.Parse(gen.OpenUnicodeFile("idna", "", "IdnaMappingTable.txt"), func(p *ucd.Parser) {
		r := p.Rune(0)
		x := lookup(r)
		if got, want := x.category(), catFromEntry(p); got != want {
			t.Errorf("%U:category: got %x; want %x", r, got, want)
		}

		mapped := false
		switch p.String(1) {
		case "mapped", "disallowed_STD3_mapped", "deviation":
			mapped = true
		}
		if x.isMapped() != mapped {
			t.Errorf("%U:isMapped: got %v; want %v", r, x.isMapped(), mapped)
		}
		if !mapped {
			return
		}
		want := string(p.Runes(2))
		got := string(x.appendMapping(nil, string(r)))
		if got != want {
			t.Errorf("%U:mapping: got %+q; want %+q", r, got, want)
		}

		if x.isMapped() {
			return
		}
		wantMark := unicode.In(r, unicode.Mark)
		gotMark := x.isModifier()
		if gotMark != wantMark {
			t.Errorf("IsMark(%U) = %v; want %v", r, gotMark, wantMark)
		}
	})

	ucd.Parse(gen.OpenUCDFile("UnicodeData.txt"), func(p *ucd.Parser) {
		r := p.Rune(0)
		x := lookup(r)
		got := x.isViramaModifier()

		const cccVirama = 9
		want := p.Int(ucd.CanonicalCombiningClass) == cccVirama
		if got != want {
			t.Errorf("IsVirama(%U) = %v; want %v", r, got, want)
		}

		rtl := false
		switch p.String(ucd.BidiClass) {
		case "R", "AL", "AN":
			rtl = true
		}
		if got := x.isBidi("A"); got != rtl && !x.isMapped() {
			t.Errorf("IsBidi(%U) = %v; want %v", r, got, rtl)
		}
	})

	ucd.Parse(gen.OpenUCDFile("extracted/DerivedJoiningType.txt"), func(p *ucd.Parser) {
		r := p.Rune(0)
		x := lookup(r)
		if x.isMapped() {
			return
		}
		got := x.joinType()
		want := joinType[p.String(1)]
		if got != want {
			t.Errorf("JoinType(%U) = %x; want %x", r, got, want)
		}
	})
}
