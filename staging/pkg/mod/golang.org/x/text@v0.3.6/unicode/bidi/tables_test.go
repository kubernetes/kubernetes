// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bidi

import (
	"testing"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/internal/ucd"
)

var labels = []string{
	AL:  "AL",
	AN:  "AN",
	B:   "B",
	BN:  "BN",
	CS:  "CS",
	EN:  "EN",
	ES:  "ES",
	ET:  "ET",
	L:   "L",
	NSM: "NSM",
	ON:  "ON",
	R:   "R",
	S:   "S",
	WS:  "WS",

	LRO: "LRO",
	RLO: "RLO",
	LRE: "LRE",
	RLE: "RLE",
	PDF: "PDF",
	LRI: "LRI",
	RLI: "RLI",
	FSI: "FSI",
	PDI: "PDI",
}

func TestTables(t *testing.T) {
	testtext.SkipIfNotLong(t)

	ucd.Parse(gen.OpenUCDFile("BidiBrackets.txt"), func(p *ucd.Parser) {
		r1 := p.Rune(0)
		want := p.Rune(1)

		e, _ := LookupRune(r1)
		if got := e.reverseBracket(r1); got != want {
			t.Errorf("Reverse(%U) = %U; want %U", r1, got, want)
		}
	})

	done := map[rune]bool{}
	test := func(name string, r rune, want string) {
		str := string(r)
		e, _ := LookupString(str)
		if got := labels[e.Class()]; got != want {
			t.Errorf("%s:%U: got %s; want %s", name, r, got, want)
		}
		if e2, sz := LookupRune(r); e != e2 || sz != len(str) {
			t.Errorf("LookupRune(%U) = %v, %d; want %v, %d", r, e2, e, sz, len(str))
		}
		if e2, sz := Lookup([]byte(str)); e != e2 || sz != len(str) {
			t.Errorf("Lookup(%U) = %v, %d; want %v, %d", r, e2, e, sz, len(str))
		}
		done[r] = true
	}

	// Insert the derived BiDi properties.
	ucd.Parse(gen.OpenUCDFile("extracted/DerivedBidiClass.txt"), func(p *ucd.Parser) {
		r := p.Rune(0)
		test("derived", r, p.String(1))
	})
	visitDefaults(func(r rune, c Class) {
		if !done[r] {
			test("default", r, labels[c])
		}
	})

}
