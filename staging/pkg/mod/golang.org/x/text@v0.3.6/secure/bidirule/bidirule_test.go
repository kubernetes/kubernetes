// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bidirule

import (
	"fmt"
	"testing"

	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/unicode/bidi"
)

const (
	strL   = "ABC"    // Left to right - most letters in LTR scripts
	strR   = "עברית"  // Right to left - most letters in non-Arabic RTL scripts
	strAL  = "دبي"    // Arabic letters - most letters in the Arabic script
	strEN  = "123"    // European Number (0-9, and Extended Arabic-Indic numbers)
	strES  = "+-"     // European Number Separator (+ and -)
	strET  = "$"      // European Number Terminator (currency symbols, the hash sign, the percent sign and so on)
	strAN  = "\u0660" // Arabic Number; this encompasses the Arabic-Indic numbers, but not the Extended Arabic-Indic numbers
	strCS  = ","      // Common Number Separator (. , / : et al)
	strNSM = "\u0300" // Nonspacing Mark - most combining accents
	strBN  = "\u200d" // Boundary Neutral - control characters (ZWNJ, ZWJ, and others)
	strB   = "\u2029" // Paragraph Separator
	strS   = "\u0009" // Segment Separator
	strWS  = " "      // Whitespace, including the SPACE character
	strON  = "@"      // Other Neutrals, including @, &, parentheses, MIDDLE DOT
)

type ruleTest struct {
	in  string
	dir bidi.Direction
	n   int // position at which the rule fails
	err error

	// For tests that split the string in two.
	pSrc  int   // number of source bytes to consume first
	szDst int   // size of destination buffer
	nSrc  int   // source bytes consumed and bytes written
	err0  error // error after first run
}

func init() {
	for rule, cases := range testCases {
		for i, tc := range cases {
			if tc.err == nil {
				testCases[rule][i].n = len(tc.in)
			}
		}
	}
}

func doTests(t *testing.T, fn func(t *testing.T, tc ruleTest)) {
	for rule, cases := range testCases {
		for i, tc := range cases {
			name := fmt.Sprintf("%d/%d:%+q:%s", rule, i, tc.in, tc.in)
			testtext.Run(t, name, func(t *testing.T) {
				fn(t, tc)
			})
		}
	}
}

func TestDirection(t *testing.T) {
	doTests(t, func(t *testing.T, tc ruleTest) {
		dir := Direction([]byte(tc.in))
		if dir != tc.dir {
			t.Errorf("dir was %v; want %v", dir, tc.dir)
		}
	})
}

func TestDirectionString(t *testing.T) {
	doTests(t, func(t *testing.T, tc ruleTest) {
		dir := DirectionString(tc.in)
		if dir != tc.dir {
			t.Errorf("dir was %v; want %v", dir, tc.dir)
		}
	})
}

func TestValid(t *testing.T) {
	doTests(t, func(t *testing.T, tc ruleTest) {
		got := Valid([]byte(tc.in))
		want := tc.err == nil
		if got != want {
			t.Fatalf("Valid: got %v; want %v", got, want)
		}

		got = ValidString(tc.in)
		want = tc.err == nil
		if got != want {
			t.Fatalf("Valid: got %v; want %v", got, want)
		}
	})
}

func TestSpan(t *testing.T) {
	doTests(t, func(t *testing.T, tc ruleTest) {
		// Skip tests that test for limited destination buffer size.
		if tc.szDst > 0 {
			return
		}

		r := New()
		src := []byte(tc.in)

		n, err := r.Span(src[:tc.pSrc], tc.pSrc == len(tc.in))
		if err != tc.err0 {
			t.Errorf("err0 was %v; want %v", err, tc.err0)
		}
		if n != tc.nSrc {
			t.Fatalf("nSrc was %d; want %d", n, tc.nSrc)
		}

		n, err = r.Span(src[n:], true)
		if err != tc.err {
			t.Errorf("error was %v; want %v", err, tc.err)
		}
		if got := n + tc.nSrc; got != tc.n {
			t.Errorf("n was %d; want %d", got, tc.n)
		}
	})
}

func TestTransform(t *testing.T) {
	doTests(t, func(t *testing.T, tc ruleTest) {
		r := New()

		src := []byte(tc.in)
		dst := make([]byte, len(tc.in))
		if tc.szDst > 0 {
			dst = make([]byte, tc.szDst)
		}

		// First transform operates on a zero-length string for most tests.
		nDst, nSrc, err := r.Transform(dst, src[:tc.pSrc], tc.pSrc == len(tc.in))
		if err != tc.err0 {
			t.Errorf("err0 was %v; want %v", err, tc.err0)
		}
		if nDst != nSrc {
			t.Fatalf("nDst (%d) and nSrc (%d) should match", nDst, nSrc)
		}
		if nSrc != tc.nSrc {
			t.Fatalf("nSrc was %d; want %d", nSrc, tc.nSrc)
		}

		dst1 := make([]byte, len(tc.in))
		copy(dst1, dst[:nDst])

		nDst, nSrc, err = r.Transform(dst1[nDst:], src[nSrc:], true)
		if err != tc.err {
			t.Errorf("error was %v; want %v", err, tc.err)
		}
		if nDst != nSrc {
			t.Fatalf("nDst (%d) and nSrc (%d) should match", nDst, nSrc)
		}
		n := nSrc + tc.nSrc
		if n != tc.n {
			t.Fatalf("n was %d; want %d", n, tc.n)
		}
		if got, want := string(dst1[:n]), tc.in[:tc.n]; got != want {
			t.Errorf("got %+q; want %+q", got, want)
		}
	})
}
