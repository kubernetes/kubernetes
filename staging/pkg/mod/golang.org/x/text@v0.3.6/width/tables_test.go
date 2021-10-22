// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package width

import (
	"testing"

	"golang.org/x/text/internal/testtext"
)

const (
	loSurrogate = 0xD800
	hiSurrogate = 0xDFFF
)

func TestTables(t *testing.T) {
	testtext.SkipIfNotLong(t)

	runes := map[rune]Kind{}
	getWidthData(func(r rune, tag elem, _ rune) {
		runes[r] = tag.kind()
	})
	for r := rune(0); r < 0x10FFFF; r++ {
		if loSurrogate <= r && r <= hiSurrogate {
			continue
		}
		p := LookupRune(r)
		if got, want := p.Kind(), runes[r]; got != want {
			t.Errorf("Kind of %U was %s; want %s.", r, got, want)
		}
		want, mapped := foldRune(r)
		if got := p.Folded(); (got == 0) == mapped || got != 0 && got != want {
			t.Errorf("Folded(%U) = %U; want %U", r, got, want)
		}
		want, mapped = widenRune(r)
		if got := p.Wide(); (got == 0) == mapped || got != 0 && got != want {
			t.Errorf("Wide(%U) = %U; want %U", r, got, want)
		}
		want, mapped = narrowRune(r)
		if got := p.Narrow(); (got == 0) == mapped || got != 0 && got != want {
			t.Errorf("Narrow(%U) = %U; want %U", r, got, want)
		}
	}
}

// TestAmbiguous verifies that ambiguous runes with a mapping always map to
// a halfwidth rune.
func TestAmbiguous(t *testing.T) {
	for r, m := range mapRunes {
		if m.e != tagAmbiguous {
			continue
		}
		if k := mapRunes[m.r].e.kind(); k != EastAsianHalfwidth {
			t.Errorf("Rune %U is ambiguous and maps to a rune of type %v", r, k)
		}
	}
}
