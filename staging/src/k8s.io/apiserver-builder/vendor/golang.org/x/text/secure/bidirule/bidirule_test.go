// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bidirule

import (
	"testing"

	"golang.org/x/text/transform"
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

var testCases = [][]ruleTest{
	// Go-specific rules.
	// Invalid UTF-8 is invalid.
	0: []ruleTest{{
		in:  "",
		dir: bidi.LeftToRight,
	}, {
		in:  "\x80",
		dir: bidi.Neutral,
		err: ErrInvalid,
		n:   0,
	}, {
		in:  "\xcc",
		dir: bidi.Neutral,
		err: ErrInvalid,
		n:   0,
	}, {
		in:  "abc\x80",
		dir: bidi.Neutral,
		err: ErrInvalid,
		n:   3,
	}, {
		in:  "abc\xcc",
		dir: bidi.Neutral,
		err: ErrInvalid,
		n:   3,
	}, {
		in:  "abc\xccdef",
		dir: bidi.Neutral,
		err: ErrInvalid,
		n:   3,
	}, {
		in:  "\xccdef",
		dir: bidi.Neutral,
		err: ErrInvalid,
		n:   0,
	}, {
		in:  strR + "\x80",
		dir: bidi.Neutral,
		err: ErrInvalid,
		n:   len(strR),
	}, {
		in:  strR + "\xcc",
		dir: bidi.Neutral,
		err: ErrInvalid,
		n:   len(strR),
	}, {
		in:  strAL + "\xcc" + strR,
		dir: bidi.Neutral,
		err: ErrInvalid,
		n:   len(strAL),
	}, {
		in:  "\xcc" + strR,
		dir: bidi.Neutral,
		err: ErrInvalid,
		n:   0,
	}},

	// Rule 2.1: The first character must be a character with Bidi property L,
	// R, or AL.  If it has the R or AL property, it is an RTL label; if it has
	// the L property, it is an LTR label.
	1: []ruleTest{{
		in:  strL,
		dir: bidi.LeftToRight,
	}, {
		in:  strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL,
		dir: bidi.RightToLeft,
	}, {
		in:  strEN,
		dir: bidi.Neutral,
		err: ErrInvalid,
	}, {
		in:  strES,
		dir: bidi.Neutral,
		err: ErrInvalid,
	}, {
		in:  strET,
		dir: bidi.Neutral,
		err: ErrInvalid,
	}, {
		in:  strAN,
		dir: bidi.Neutral,
		err: ErrInvalid,
	}, {
		in:  strCS,
		dir: bidi.Neutral,
		err: ErrInvalid,
	}, {
		in:  strNSM,
		dir: bidi.Neutral,
		err: ErrInvalid,
	}, {
		in:  strBN,
		dir: bidi.Neutral,
		err: ErrInvalid,
	}, {
		in:  strB,
		dir: bidi.Neutral,
		err: ErrInvalid,
	}, {
		in:  strS,
		dir: bidi.Neutral,
		err: ErrInvalid,
	}, {
		in:  strWS,
		dir: bidi.Neutral,
		err: ErrInvalid,
	}, {
		in:  strON,
		dir: bidi.Neutral,
		err: ErrInvalid,
	}},

	// Rule 2.2: In an RTL label, only characters with the Bidi properties R,
	// AL, AN, EN, ES, CS, ET, ON, BN, or NSM are allowed.
	2: []ruleTest{{
		in:  strR + strR + strAL,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strAL + strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strAN + strAL,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strEN + strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strES + strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strCS + strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strET + strAL,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strON + strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strBN + strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strNSM + strAL,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strL + strR,
		dir: bidi.Neutral,
		n:   len(strR),
		err: ErrInvalid,
	}, {
		in:  strR + strB + strR,
		dir: bidi.Neutral,
		n:   len(strR),
		err: ErrInvalid,
	}, {
		in:  strR + strS + strAL,
		dir: bidi.Neutral,
		n:   len(strR),
		err: ErrInvalid,
	}, {
		in:  strR + strWS + strAL,
		dir: bidi.Neutral,
		n:   len(strR),
		err: ErrInvalid,
	}, {
		in:  strAL + strR + strAL,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strAL + strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strAN + strAL,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strEN + strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strES + strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strCS + strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strET + strAL,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strON + strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strBN + strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strNSM + strAL,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strL + strR,
		dir: bidi.Neutral,
		n:   len(strAL),
		err: ErrInvalid,
	}, {
		in:  strAL + strB + strR,
		dir: bidi.Neutral,
		n:   len(strAL),
		err: ErrInvalid,
	}, {
		in:  strAL + strS + strAL,
		dir: bidi.Neutral,
		n:   len(strAL),
		err: ErrInvalid,
	}, {
		in:  strAL + strWS + strAL,
		dir: bidi.Neutral,
		n:   len(strAL),
		err: ErrInvalid,
	}},

	// Rule 2.3: In an RTL label, the end of the label must be a character with
	// Bidi property R, AL, EN, or AN, followed by zero or more characters with
	// Bidi property NSM.
	3: []ruleTest{{
		in:  strR + strNSM,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strAL + strNSM,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strEN + strNSM + strNSM,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strAN,
		dir: bidi.RightToLeft,
	}, {
		in:  strR + strES + strNSM,
		dir: bidi.Neutral,
		n:   len(strR + strES + strNSM),
		err: ErrInvalid,
	}, {
		in:  strR + strCS + strNSM + strNSM,
		dir: bidi.Neutral,
		n:   len(strR + strCS + strNSM + strNSM),
		err: ErrInvalid,
	}, {
		in:  strR + strET,
		dir: bidi.Neutral,
		n:   len(strR + strET),
		err: ErrInvalid,
	}, {
		in:  strR + strON + strNSM,
		dir: bidi.Neutral,
		n:   len(strR + strON + strNSM),
		err: ErrInvalid,
	}, {
		in:  strR + strBN + strNSM + strNSM,
		dir: bidi.Neutral,
		n:   len(strR + strBN + strNSM + strNSM),
		err: ErrInvalid,
	}, {
		in:  strR + strL + strNSM,
		dir: bidi.Neutral,
		n:   len(strR),
		err: ErrInvalid,
	}, {
		in:  strR + strB + strNSM + strNSM,
		dir: bidi.Neutral,
		n:   len(strR),
		err: ErrInvalid,
	}, {
		in:  strR + strS,
		dir: bidi.Neutral,
		n:   len(strR),
		err: ErrInvalid,
	}, {
		in:  strR + strWS,
		dir: bidi.Neutral,
		n:   len(strR),
		err: ErrInvalid,
	}, {
		in:  strAL + strNSM,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strR,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strAL + strNSM,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strEN + strNSM + strNSM,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strAN,
		dir: bidi.RightToLeft,
	}, {
		in:  strAL + strES + strNSM,
		dir: bidi.Neutral,
		n:   len(strAL + strES + strNSM),
		err: ErrInvalid,
	}, {
		in:  strAL + strCS + strNSM + strNSM,
		dir: bidi.Neutral,
		n:   len(strAL + strCS + strNSM + strNSM),
		err: ErrInvalid,
	}, {
		in:  strAL + strET,
		dir: bidi.Neutral,
		n:   len(strAL + strET),
		err: ErrInvalid,
	}, {
		in:  strAL + strON + strNSM,
		dir: bidi.Neutral,
		n:   len(strAL + strON + strNSM),
		err: ErrInvalid,
	}, {
		in:  strAL + strBN + strNSM + strNSM,
		dir: bidi.Neutral,
		n:   len(strAL + strBN + strNSM + strNSM),
		err: ErrInvalid,
	}, {
		in:  strAL + strL + strNSM,
		dir: bidi.Neutral,
		n:   len(strAL),
		err: ErrInvalid,
	}, {
		in:  strAL + strB + strNSM + strNSM,
		dir: bidi.Neutral,
		n:   len(strAL),
		err: ErrInvalid,
	}, {
		in:  strAL + strS,
		dir: bidi.Neutral,
		n:   len(strAL),
		err: ErrInvalid,
	}, {
		in:  strAL + strWS,
		dir: bidi.Neutral,
		n:   len(strAL),
		err: ErrInvalid,
	}},

	// Rule 2.4: In an RTL label, if an EN is present, no AN may be present,
	// and vice versa.
	4: []ruleTest{{
		in:  strR + strEN + strAN,
		dir: bidi.Neutral,
		n:   len(strR + strEN),
		err: ErrInvalid,
	}, {
		in:  strR + strAN + strEN + strNSM,
		dir: bidi.Neutral,
		n:   len(strR + strAN),
		err: ErrInvalid,
	}, {
		in:  strAL + strEN + strAN,
		dir: bidi.Neutral,
		n:   len(strAL + strEN),
		err: ErrInvalid,
	}, {
		in:  strAL + strAN + strEN + strNSM,
		dir: bidi.Neutral,
		n:   len(strAL + strAN),
		err: ErrInvalid,
	}},

	// Rule 2.5: In an LTR label, only characters with the Bidi properties L,
	// EN, ES, CS, ET, ON, BN, or NSM are allowed.
	5: []ruleTest{{
		in:  strL + strL + strL,
		dir: bidi.LeftToRight,
	}, {
		in:  strL + strEN + strL,
		dir: bidi.LeftToRight,
	}, {
		in:  strL + strES + strL,
		dir: bidi.LeftToRight,
	}, {
		in:  strL + strCS + strL,
		dir: bidi.LeftToRight,
	}, {
		in:  strL + strET + strL,
		dir: bidi.LeftToRight,
	}, {
		in:  strL + strON + strL,
		dir: bidi.LeftToRight,
	}, {
		in:  strL + strBN + strL,
		dir: bidi.LeftToRight,
	}, {
		in:  strL + strNSM + strL,
		dir: bidi.LeftToRight,
	}, {
		in:  strL + strR + strL,
		dir: bidi.Neutral,
		n:   len(strL),
		err: ErrInvalid,
	}, {
		in:  strL + strAL + strL,
		dir: bidi.Neutral,
		n:   len(strL),
		err: ErrInvalid,
	}, {
		in:  strL + strAN + strL,
		dir: bidi.Neutral,
		n:   len(strL),
		err: ErrInvalid,
	}, {
		in:  strL + strB + strL,
		dir: bidi.Neutral,
		n:   len(strL),
		err: ErrInvalid,
	}, {
		in:  strL + strS + strL,
		dir: bidi.Neutral,
		n:   len(strL),
		err: ErrInvalid,
	}, {
		in:  strL + strWS + strL,
		dir: bidi.Neutral,
		n:   len(strL),
		err: ErrInvalid,
	}},

	// Rule 2.6: In an LTR label, the end of the label must be a character with
	// Bidi property L or EN, followed by zero or more characters with Bidi
	// property NSM.
	6: []ruleTest{{
		in:  strL,
		dir: bidi.LeftToRight,
	}, {
		in:  strL + strNSM,
		dir: bidi.LeftToRight,
	}, {
		in:  strL + strNSM + strNSM,
		dir: bidi.LeftToRight,
	}, {
		in:  strL + strEN,
		dir: bidi.LeftToRight,
	}, {
		in:  strL + strEN + strNSM,
		dir: bidi.LeftToRight,
	}, {
		in:  strL + strEN + strNSM + strNSM,
		dir: bidi.LeftToRight,
	}, {
		in:  strL + strES,
		dir: bidi.Neutral,
		n:   len(strL + strES),
		err: ErrInvalid,
	}, {
		in:  strL + strCS,
		dir: bidi.Neutral,
		n:   len(strL + strCS),
		err: ErrInvalid,
	}, {
		in:  strL + strET,
		dir: bidi.Neutral,
		n:   len(strL + strET),
		err: ErrInvalid,
	}, {
		in:  strL + strON,
		dir: bidi.Neutral,
		n:   len(strL + strON),
		err: ErrInvalid,
	}, {
		in:  strL + strBN,
		dir: bidi.Neutral,
		n:   len(strL + strBN),
		err: ErrInvalid,
	}, {
		in:  strL + strR,
		dir: bidi.Neutral,
		n:   len(strL),
		err: ErrInvalid,
	}, {
		in:  strL + strAL,
		dir: bidi.Neutral,
		n:   len(strL),
		err: ErrInvalid,
	}, {
		in:  strL + strAN,
		dir: bidi.Neutral,
		n:   len(strL),
		err: ErrInvalid,
	}, {
		in:  strL + strB,
		dir: bidi.Neutral,
		n:   len(strL),
		err: ErrInvalid,
	}, {
		in:  strL + strS,
		dir: bidi.Neutral,
		n:   len(strL),
		err: ErrInvalid,
	}, {
		in:  strL + strWS,
		dir: bidi.Neutral,
		n:   len(strL),
		err: ErrInvalid,
	}},

	// Incremental processing.
	9: []ruleTest{{
		in:  "e\u0301", // é
		dir: bidi.LeftToRight,

		pSrc: 2,
		nSrc: 1,
		err0: transform.ErrShortSrc,
	}, {
		in:  "e\u1000f", // é
		dir: bidi.LeftToRight,

		pSrc: 3,
		nSrc: 1,
		err0: transform.ErrShortSrc,
	}, {
		// Remain invalid once invalid.
		in:  strR + "ab",
		dir: bidi.Neutral,
		n:   len(strR),
		err: ErrInvalid,

		pSrc: len(strR) + 1,
		nSrc: len(strR),
		err0: ErrInvalid,
	}, {
		// Short destination
		in:  "abcdefghij",
		dir: bidi.LeftToRight,

		pSrc:  10,
		szDst: 5,
		nSrc:  5,
		err0:  transform.ErrShortDst,
	}, {
		// Short destination splitting input rune
		in:  "e\u0301",
		dir: bidi.LeftToRight,

		pSrc:  3,
		szDst: 2,
		nSrc:  1,
		err0:  transform.ErrShortDst,
	}},
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

func TestDirection(t *testing.T) {
	doTests(t, func(t *testing.T, tc ruleTest) {
		dir, err := Direction([]byte(tc.in))
		if err != tc.err {
			t.Errorf("error was %v; want %v", err, tc.err)
		}
		if dir != tc.dir {
			t.Errorf("dir was %v; want %v", dir, tc.dir)
		}
	})
}

func TestDirectionString(t *testing.T) {
	doTests(t, func(t *testing.T, tc ruleTest) {
		dir, err := DirectionString(tc.in)
		if err != tc.err {
			t.Errorf("error was %v; want %v", err, tc.err)
		}
		if dir != tc.dir {
			t.Errorf("dir was %v; want %v", dir, tc.dir)
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
