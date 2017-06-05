// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package precis

import (
	"testing"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/runes"
	"golang.org/x/text/unicode/rangetable"
)

type tableTest struct {
	rangeTable *unicode.RangeTable
	prop       property
}

var exceptions = runes.Predicate(func(r rune) bool {
	switch uint32(r) {
	case 0x00DF, 0x03C2, 0x06FD, 0x06FE, 0x0F0B, 0x3007, 0x00B7, 0x0375, 0x05F3,
		0x05F4, 0x30FB, 0x0660, 0x0661, 0x0662, 0x0663, 0x0664, 0x0665, 0x0666,
		0x0667, 0x0668, 0x0669, 0x06F0, 0x06F1, 0x06F2, 0x06F3, 0x06F4, 0x06F5,
		0x06F6, 0x06F7, 0x06F8, 0x06F9, 0x0640, 0x07FA, 0x302E, 0x302F, 0x3031,
		0x3032, 0x3033, 0x3034, 0x3035, 0x303B:
		return true
	default:
		return false
	}
})

// Ensure that ceratain properties were generated correctly.
func TestTable(t *testing.T) {
	tests := []tableTest{
		tableTest{
			rangetable.Merge(
				unicode.Lt, unicode.Nl, unicode.No, // Other letter digits
				unicode.Me,             // Modifiers
				unicode.Zs,             // Spaces
				unicode.So,             // Symbols
				unicode.Pi, unicode.Pf, // Punctuation
			),
			idDisOrFreePVal,
		},
		tableTest{
			rangetable.New(0x30000, 0x30101, 0xDFFFF),
			unassigned,
		},
	}

	assigned := rangetable.Assigned(UnicodeVersion)

	for _, test := range tests {
		rangetable.Visit(test.rangeTable, func(r rune) {
			if !unicode.In(r, assigned) {
				return
			}
			b := make([]byte, 4)
			n := utf8.EncodeRune(b, r)
			trieval, _ := dpTrie.lookup(b[:n])
			p := entry(trieval).property()
			if p != test.prop && !exceptions.Contains(r) {
				t.Errorf("%U: got %+x; want %+x", r, test.prop, p)
			}
		})
	}
}
