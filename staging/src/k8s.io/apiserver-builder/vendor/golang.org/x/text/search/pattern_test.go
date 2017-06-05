// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package search

import (
	"reflect"
	"strings"
	"testing"

	"golang.org/x/text/language"
)

func TestCompile(t *testing.T) {
	for i, tc := range []struct {
		desc    string
		pattern string
		options []Option
		n       int
	}{{
		desc:    "empty",
		pattern: "",
		n:       0,
	}, {
		desc:    "single",
		pattern: "a",
		n:       1,
	}, {
		desc:    "keep modifier",
		pattern: "a\u0300", // U+0300: COMBINING GRAVE ACCENT
		n:       2,
	}, {
		desc:    "remove modifier",
		pattern: "a\u0300", // U+0300: COMBINING GRAVE ACCENT
		options: []Option{IgnoreDiacritics},
		n:       1,
	}, {
		desc:    "single with double collation element",
		pattern: "ä",
		n:       2,
	}, {
		desc:    "leading variable",
		pattern: " a",
		n:       2,
	}, {
		desc:    "trailing variable",
		pattern: "aa ",
		n:       3,
	}, {
		desc:    "leading and trailing variable",
		pattern: " äb ",
		n:       5,
	}, {
		desc:    "keep interior variable",
		pattern: " ä b ",
		n:       6,
	}, {
		desc:    "keep interior variables",
		pattern: " b  ä ",
		n:       7,
	}, {
		desc:    "remove ignoreables (zero-weights across the board)",
		pattern: "\u009Db\u009Dä\u009D", // U+009D: OPERATING SYSTEM COMMAND
		n:       3,
	}} {
		m := New(language.Und, tc.options...)
		p := m.CompileString(tc.pattern)
		if len(p.ce) != tc.n {
			t.Errorf("%d:%s: Compile(%+q): got %d; want %d", i, tc.desc, tc.pattern, len(p.ce), tc.n)
		}
	}
}

func TestNorm(t *testing.T) {
	// U+0300: COMBINING GRAVE ACCENT (CCC=230)
	// U+031B: COMBINING HORN (CCC=216)
	for _, tc := range []struct {
		desc string
		a    string
		b    string
		want bool // a and b compile into the same pattern?
	}{{
		"simple",
		"eee\u0300\u031b",
		"eee\u031b\u0300",
		true,
	}, {
		"large number of modifiers in pattern",
		strings.Repeat("\u0300", 29) + "\u0318",
		"\u0318" + strings.Repeat("\u0300", 29),
		true,
	}, {
		"modifier overflow in pattern",
		strings.Repeat("\u0300", 30) + "\u0318",
		"\u0318" + strings.Repeat("\u0300", 30),
		false,
	}} {
		m := New(language.Und)
		a := m.CompileString(tc.a)
		b := m.CompileString(tc.b)
		if got := reflect.DeepEqual(a, b); got != tc.want {
			t.Errorf("Compile(a) == Compile(b) == %v; want %v", got, tc.want)
		}
	}
}

func TestForwardSearch(t *testing.T) {
	for i, tc := range []struct {
		desc    string
		tag     string
		options []Option
		pattern string
		text    string
		want    []int
	}{{
		// The semantics of an empty search is to match nothing.
		// TODO: change this to be in line with strings.Index? It is quite a
		// different beast, so not sure yet.

		desc:    "empty pattern and text",
		tag:     "und",
		pattern: "",
		text:    "",
		want:    nil, // TODO: consider: []int{0, 0},
	}, {
		desc:    "non-empty pattern and empty text",
		tag:     "und",
		pattern: " ",
		text:    "",
		want:    nil,
	}, {
		desc:    "empty pattern and non-empty text",
		tag:     "und",
		pattern: "",
		text:    "abc",
		want:    nil, // TODO: consider: []int{0, 0, 1, 1, 2, 2, 3, 3},
	}, {
		// Variable-only patterns. We don't support variables at the moment,
		// but verify that, given this, the behavior is indeed as expected.

		desc:    "exact match of variable",
		tag:     "und",
		pattern: " ",
		text:    " ",
		want:    []int{0, 1},
	}, {
		desc:    "variables not handled by default",
		tag:     "und",
		pattern: "- ",
		text:    " -",
		want:    nil, // Would be (1, 2) for a median match with variable}.
	}, {
		desc:    "multiple subsequent identical variables",
		tag:     "und",
		pattern: " ",
		text:    "    ",
		want:    []int{0, 1, 1, 2, 2, 3, 3, 4},
	}, {
		desc:    "text with variables",
		tag:     "und",
		options: []Option{IgnoreDiacritics},
		pattern: "abc",
		text:    "3 abc 3",
		want:    []int{2, 5},
	}, {
		desc:    "pattern with interior variables",
		tag:     "und",
		options: []Option{IgnoreDiacritics},
		pattern: "a b c",
		text:    "3 a b c abc a  b  c 3",
		want:    []int{2, 7}, // Would have 3 matches using variable.

		// TODO: Different variable handling settings.
	}, {
		// Options.

		desc:    "match all levels",
		tag:     "und",
		pattern: "Abc",
		text:    "abcAbcABCÁbcábc",
		want:    []int{3, 6},
	}, {
		desc:    "ignore diacritics in text",
		tag:     "und",
		options: []Option{IgnoreDiacritics},
		pattern: "Abc",
		text:    "Ábc",
		want:    []int{0, 4},
	}, {
		desc:    "ignore diacritics in pattern",
		tag:     "und",
		options: []Option{IgnoreDiacritics},
		pattern: "Ábc",
		text:    "Abc",
		want:    []int{0, 3},
	}, {
		desc:    "ignore diacritics",
		tag:     "und",
		options: []Option{IgnoreDiacritics},
		pattern: "Abc",
		text:    "abcAbcABCÁbcábc",
		want:    []int{3, 6, 9, 13},
	}, {
		desc:    "ignore case",
		tag:     "und",
		options: []Option{IgnoreCase},
		pattern: "Abc",
		text:    "abcAbcABCÁbcábc",
		want:    []int{0, 3, 3, 6, 6, 9},
	}, {
		desc:    "ignore case and diacritics",
		tag:     "und",
		options: []Option{IgnoreCase, IgnoreDiacritics},
		pattern: "Abc",
		text:    "abcAbcABCÁbcábc",
		want:    []int{0, 3, 3, 6, 6, 9, 9, 13, 13, 17},
	}, {
		desc:    "ignore width to fullwidth",
		tag:     "und",
		options: []Option{IgnoreWidth},
		pattern: "abc",
		text:    "123 \uFF41\uFF42\uFF43 123", // U+FF41-3: FULLWIDTH LATIN SMALL LETTER A-C
		want:    []int{4, 13},
	}, {
		// TODO: distinguish between case and width.
		desc:    "don't ignore width to fullwidth, ignoring only case",
		tag:     "und",
		options: []Option{IgnoreCase},
		pattern: "abc",
		text:    "123 \uFF41\uFF42\uFF43 123", // U+FF41-3: FULLWIDTH LATIN SMALL LETTER A-C
		want:    []int{4, 13},
	}, {
		desc:    "ignore width to fullwidth and diacritics",
		tag:     "und",
		options: []Option{IgnoreWidth, IgnoreDiacritics},
		pattern: "abc",
		text:    "123 \uFF41\uFF42\uFF43 123", // U+FF41-3: FULLWIDTH LATIN SMALL LETTER A-C
		want:    []int{4, 13},
	}, {
		desc:    "whole grapheme, single rune",
		tag:     "und",
		pattern: "eee",
		text:    "123 eeé 123",
		want:    nil,
	}, {
		// Note: rules on when to apply contractions may, for certain languages,
		// differ between search and collation. For example, "ch" is not
		// considered a contraction for the purpose of searching in Spanish.
		// Therefore, be careful picking this test.
		desc:    "whole grapheme, contractions",
		tag:     "da",
		pattern: "aba",
		// Fails at the primary level, because "aa" is a contraction.
		text: "123 abaa 123",
		want: []int{},
	}, {
		desc:    "whole grapheme, trailing modifier",
		tag:     "und",
		pattern: "eee",
		text:    "123 eee\u0300 123", // U+0300: COMBINING GRAVE ACCENT
		want:    nil,
	}, {
		// Language-specific matching.

		desc:    "",
		tag:     "da",
		options: []Option{IgnoreCase},
		pattern: "Århus",
		text:    "AarhusÅrhus  Århus  ",
		want:    []int{0, 6, 6, 12, 14, 20},
	}, {
		desc:    "",
		tag:     "da",
		options: []Option{IgnoreCase},
		pattern: "Aarhus",
		text:    "Århus Aarhus",
		want:    []int{0, 6, 7, 13},
	}, {
		desc:    "",
		tag:     "en", // Å does not match A for English.
		options: []Option{IgnoreCase},
		pattern: "Aarhus",
		text:    "Århus",
		want:    nil,
	}, {
		desc:    "ignore modifier in text",
		options: []Option{IgnoreDiacritics},
		tag:     "und",
		pattern: "eee",
		text:    "123 eee\u0300 123", // U+0300: COMBINING GRAVE ACCENT
		want:    []int{4, 9},         // Matches on grapheme boundary.
	}, {
		desc:    "ignore multiple modifiers in text",
		options: []Option{IgnoreDiacritics},
		tag:     "und",
		pattern: "eee",
		text:    "123 eee\u0300\u0300 123", // U+0300: COMBINING GRAVE ACCENT
		want:    []int{4, 11},              // Matches on grapheme boundary.
	}, {
		desc:    "ignore modifier in pattern",
		options: []Option{IgnoreDiacritics},
		tag:     "und",
		pattern: "eee\u0300", // U+0300: COMBINING GRAVE ACCENT
		text:    "123 eee 123",
		want:    []int{4, 7},
	}, {
		desc:    "ignore multiple modifiers in pattern",
		options: []Option{IgnoreDiacritics},
		tag:     "und",
		pattern: "eee\u0300\u0300", // U+0300: COMBINING GRAVE ACCENT
		text:    "123 eee 123",
		want:    []int{4, 7},
	}, {
		desc: "match non-normalized pattern",
		tag:  "und",
		// U+0300: COMBINING GRAVE ACCENT (CCC=230)
		// U+031B: COMBINING HORN (CCC=216)
		pattern: "eee\u0300\u031b",
		text:    "123 eee\u031b\u0300 123",
		want:    []int{4, 11},
	}, {
		desc: "match non-normalized text",
		tag:  "und",
		// U+0300: COMBINING GRAVE ACCENT (CCC=230)
		// U+031B: COMBINING HORN (CCC=216)
		pattern: "eee\u031b\u0300",
		text:    "123 eee\u0300\u031b 123",
		want:    []int{4, 11},
	}} {
		m := New(language.MustParse(tc.tag), tc.options...)
		p := m.CompileString(tc.pattern)
		for j := 0; j < len(tc.text); {
			start, end := p.IndexString(tc.text[j:])
			if start == -1 && end == -1 {
				j++
				continue
			}
			start += j
			end += j
			j = end
			if len(tc.want) == 0 {
				t.Errorf("%d:%s: found unexpected result [%d %d]", i, tc.desc, start, end)
				break
			}
			if tc.want[0] != start || tc.want[1] != end {
				t.Errorf("%d:%s: got [%d %d]; want %v", i, tc.desc, start, end, tc.want[:2])
				tc.want = tc.want[2:]
				break
			}
			tc.want = tc.want[2:]
		}
		if len(tc.want) != 0 {
			t.Errorf("%d:%s: %d extra results", i, tc.desc, len(tc.want)/2)
		}
	}
}
