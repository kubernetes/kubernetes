// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package colltab

import (
	"reflect"
	"strings"
	"testing"

	"golang.org/x/text/internal/testtext"
)

const (
	digSec  = defaultSecondary
	digTert = defaultTertiary
)

var tPlus3 = e(0, 50, digTert+3)

// numWeighter is a testWeighter used for testing numericWeighter.
var numWeighter = testWeighter{
	"0": p(100),
	"０": []Elem{e(100, digSec, digTert+1)}, // U+FF10 FULLWIDTH DIGIT ZERO
	"₀": []Elem{e(100, digSec, digTert+5)}, // U+2080 SUBSCRIPT ZERO

	"1": p(101),
	// Allow non-primary collation elements to be inserted.
	"١": append(p(101), tPlus3), // U+0661 ARABIC-INDIC DIGIT ONE
	// Allow varying tertiary weight if the number is Nd.
	"１": []Elem{e(101, digSec, digTert+1)}, // U+FF11 FULLWIDTH DIGIT ONE
	"2": p(102),
	// Allow non-primary collation elements to be inserted.
	"٢": append(p(102), tPlus3), // U+0662 ARABIC-INDIC DIGIT TWO
	// Varying tertiary weights should be ignored.
	"２": []Elem{e(102, digSec, digTert+3)}, // U+FF12 FULLWIDTH DIGIT TWO
	"3": p(103),
	"4": p(104),
	"5": p(105),
	"6": p(106),
	"7": p(107),
	// Weights must be strictly monotonically increasing, but do not need to be
	// consecutive.
	"8": p(118),
	"9": p(119),
	// Allow non-primary collation elements to be inserted.
	"٩": append(p(119), tPlus3), // U+0669 ARABIC-INDIC DIGIT NINE
	// Varying tertiary weights should be ignored.
	"９": []Elem{e(119, digSec, digTert+1)}, // U+FF19 FULLWIDTH DIGIT NINE
	"₉": []Elem{e(119, digSec, digTert+5)}, // U+2089 SUBSCRIPT NINE

	"a": p(5),
	"b": p(6),
	"c": p(8, 2),

	"klm": p(99),

	"nop": p(121),

	"x": p(200),
	"y": p(201),
}

func p(w ...int) (elems []Elem) {
	for _, x := range w {
		e, _ := MakeElem(x, digSec, digTert, 0)
		elems = append(elems, e)
	}
	return elems
}

func TestNumericAppendNext(t *testing.T) {
	for _, tt := range []struct {
		in string
		w  []Elem
	}{
		{"a", p(5)},
		{"klm", p(99)},
		{"aa", p(5, 5)},
		{"1", p(120, 1, 101)},
		{"0", p(120, 0)},
		{"01", p(120, 1, 101)},
		{"0001", p(120, 1, 101)},
		{"10", p(120, 2, 101, 100)},
		{"99", p(120, 2, 119, 119)},
		{"9999", p(120, 4, 119, 119, 119, 119)},
		{"1a", p(120, 1, 101, 5)},
		{"0b", p(120, 0, 6)},
		{"01c", p(120, 1, 101, 8, 2)},
		{"10x", p(120, 2, 101, 100, 200)},
		{"99y", p(120, 2, 119, 119, 201)},
		{"9999nop", p(120, 4, 119, 119, 119, 119, 121)},

		// Allow follow-up collation elements if they have a zero non-primary.
		{"١٢٩", []Elem{e(120), e(3), e(101), tPlus3, e(102), tPlus3, e(119), tPlus3}},
		{
			"１２９",
			[]Elem{
				e(120), e(3),
				e(101, digSec, digTert+1),
				e(102, digSec, digTert+3),
				e(119, digSec, digTert+1),
			},
		},

		// Ensure AppendNext* adds to the given buffer.
		{"a10", p(5, 120, 2, 101, 100)},
	} {
		nw := NewNumericWeighter(numWeighter)

		b := []byte(tt.in)
		got := []Elem(nil)
		for n, sz := 0, 0; n < len(b); {
			got, sz = nw.AppendNext(got, b[n:])
			n += sz
		}
		if !reflect.DeepEqual(got, tt.w) {
			t.Errorf("AppendNext(%q) =\n%v; want\n%v", tt.in, got, tt.w)
		}

		got = nil
		for n, sz := 0, 0; n < len(tt.in); {
			got, sz = nw.AppendNextString(got, tt.in[n:])
			n += sz
		}
		if !reflect.DeepEqual(got, tt.w) {
			t.Errorf("AppendNextString(%q) =\n%v; want\n%v", tt.in, got, tt.w)
		}
	}
}

func TestNumericOverflow(t *testing.T) {
	manyDigits := strings.Repeat("9", maxDigits+1) + "a"

	nw := NewNumericWeighter(numWeighter)

	got, n := nw.AppendNextString(nil, manyDigits)

	if n != maxDigits {
		t.Errorf("n: got %d; want %d", n, maxDigits)
	}

	if got[1].Primary() != maxDigits {
		t.Errorf("primary(e[1]): got %d; want %d", n, maxDigits)
	}
}

func TestNumericWeighterAlloc(t *testing.T) {
	buf := make([]Elem, 100)
	w := NewNumericWeighter(numWeighter)
	s := "1234567890a"

	nNormal := testtext.AllocsPerRun(3, func() { numWeighter.AppendNextString(buf, s) })
	nNumeric := testtext.AllocsPerRun(3, func() { w.AppendNextString(buf, s) })
	if n := nNumeric - nNormal; n > 0 {
		t.Errorf("got %f; want 0", n)
	}
}
