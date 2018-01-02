// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cases

import (
	"strings"
	"testing"
	"unicode"

	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/language"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
	"golang.org/x/text/unicode/rangetable"
)

// The following definitions are taken directly from Chapter 3 of The Unicode
// Standard.

func propCased(r rune) bool {
	return propLower(r) || propUpper(r) || unicode.IsTitle(r)
}

func propLower(r rune) bool {
	return unicode.IsLower(r) || unicode.Is(unicode.Other_Lowercase, r)
}

func propUpper(r rune) bool {
	return unicode.IsUpper(r) || unicode.Is(unicode.Other_Uppercase, r)
}

func propIgnore(r rune) bool {
	if unicode.In(r, unicode.Mn, unicode.Me, unicode.Cf, unicode.Lm, unicode.Sk) {
		return true
	}
	return caseIgnorable[r]
}

func hasBreakProp(r rune) bool {
	// binary search over ranges
	lo := 0
	hi := len(breakProp)
	for lo < hi {
		m := lo + (hi-lo)/2
		bp := &breakProp[m]
		if bp.lo <= r && r <= bp.hi {
			return true
		}
		if r < bp.lo {
			hi = m
		} else {
			lo = m + 1
		}
	}
	return false
}

func contextFromRune(r rune) *context {
	c := context{dst: make([]byte, 128), src: []byte(string(r)), atEOF: true}
	c.next()
	return &c
}

func TestCaseProperties(t *testing.T) {
	if unicode.Version != UnicodeVersion {
		// Properties of existing code points may change by Unicode version, so
		// we need to skip.
		t.Skipf("Skipping as core Unicode version %s different than %s", unicode.Version, UnicodeVersion)
	}
	assigned := rangetable.Assigned(UnicodeVersion)
	coreVersion := rangetable.Assigned(unicode.Version)
	for r := rune(0); r <= lastRuneForTesting; r++ {
		if !unicode.In(r, assigned) || !unicode.In(r, coreVersion) {
			continue
		}
		c := contextFromRune(r)
		if got, want := c.info.isCaseIgnorable(), propIgnore(r); got != want {
			t.Errorf("caseIgnorable(%U): got %v; want %v (%x)", r, got, want, c.info)
		}
		// New letters may change case types, but existing case pairings should
		// not change. See Case Pair Stability in
		// http://unicode.org/policies/stability_policy.html.
		if rf := unicode.SimpleFold(r); rf != r && unicode.In(rf, assigned) {
			if got, want := c.info.isCased(), propCased(r); got != want {
				t.Errorf("cased(%U): got %v; want %v (%x)", r, got, want, c.info)
			}
			if got, want := c.caseType() == cUpper, propUpper(r); got != want {
				t.Errorf("upper(%U): got %v; want %v (%x)", r, got, want, c.info)
			}
			if got, want := c.caseType() == cLower, propLower(r); got != want {
				t.Errorf("lower(%U): got %v; want %v (%x)", r, got, want, c.info)
			}
		}
		if got, want := c.info.isBreak(), hasBreakProp(r); got != want {
			t.Errorf("isBreak(%U): got %v; want %v (%x)", r, got, want, c.info)
		}
	}
	// TODO: get title case from unicode file.
}

func TestMapping(t *testing.T) {
	assigned := rangetable.Assigned(UnicodeVersion)
	coreVersion := rangetable.Assigned(unicode.Version)
	if coreVersion == nil {
		coreVersion = assigned
	}
	apply := func(r rune, f func(c *context) bool) string {
		c := contextFromRune(r)
		f(c)
		return string(c.dst[:c.pDst])
	}

	for r, tt := range special {
		if got, want := apply(r, lower), tt.toLower; got != want {
			t.Errorf("lowerSpecial:(%U): got %+q; want %+q", r, got, want)
		}
		if got, want := apply(r, title), tt.toTitle; got != want {
			t.Errorf("titleSpecial:(%U): got %+q; want %+q", r, got, want)
		}
		if got, want := apply(r, upper), tt.toUpper; got != want {
			t.Errorf("upperSpecial:(%U): got %+q; want %+q", r, got, want)
		}
	}

	for r := rune(0); r <= lastRuneForTesting; r++ {
		if !unicode.In(r, assigned) || !unicode.In(r, coreVersion) {
			continue
		}
		if rf := unicode.SimpleFold(r); rf == r || !unicode.In(rf, assigned) {
			continue
		}
		if _, ok := special[r]; ok {
			continue
		}
		want := string(unicode.ToLower(r))
		if got := apply(r, lower); got != want {
			t.Errorf("lower:%q (%U): got %q %U; want %q %U", r, r, got, []rune(got), want, []rune(want))
		}

		want = string(unicode.ToUpper(r))
		if got := apply(r, upper); got != want {
			t.Errorf("upper:%q (%U): got %q %U; want %q %U", r, r, got, []rune(got), want, []rune(want))
		}

		want = string(unicode.ToTitle(r))
		if got := apply(r, title); got != want {
			t.Errorf("title:%q (%U): got %q %U; want %q %U", r, r, got, []rune(got), want, []rune(want))
		}
	}
}

func runeFoldData(r rune) (x struct{ simple, full, special string }) {
	x = foldMap[r]
	if x.simple == "" {
		x.simple = string(unicode.ToLower(r))
	}
	if x.full == "" {
		x.full = string(unicode.ToLower(r))
	}
	if x.special == "" {
		x.special = x.full
	}
	return
}

func TestFoldData(t *testing.T) {
	assigned := rangetable.Assigned(UnicodeVersion)
	coreVersion := rangetable.Assigned(unicode.Version)
	if coreVersion == nil {
		coreVersion = assigned
	}
	apply := func(r rune, f func(c *context) bool) (string, info) {
		c := contextFromRune(r)
		f(c)
		return string(c.dst[:c.pDst]), c.info.cccType()
	}
	for r := rune(0); r <= lastRuneForTesting; r++ {
		if !unicode.In(r, assigned) || !unicode.In(r, coreVersion) {
			continue
		}
		x := runeFoldData(r)
		if got, info := apply(r, foldFull); got != x.full {
			t.Errorf("full:%q (%U): got %q %U; want %q %U (ccc=%x)", r, r, got, []rune(got), x.full, []rune(x.full), info)
		}
		// TODO: special and simple.
	}
}

func TestCCC(t *testing.T) {
	assigned := rangetable.Assigned(UnicodeVersion)
	normVersion := rangetable.Assigned(norm.Version)
	for r := rune(0); r <= lastRuneForTesting; r++ {
		if !unicode.In(r, assigned) || !unicode.In(r, normVersion) {
			continue
		}
		c := contextFromRune(r)

		p := norm.NFC.PropertiesString(string(r))
		want := cccOther
		switch p.CCC() {
		case 0:
			want = cccZero
		case above:
			want = cccAbove
		}
		if got := c.info.cccType(); got != want {
			t.Errorf("%U: got %x; want %x", r, got, want)
		}
	}
}

func TestWordBreaks(t *testing.T) {
	for _, tt := range breakTest {
		testtext.Run(t, tt, func(t *testing.T) {
			parts := strings.Split(tt, "|")
			want := ""
			for _, s := range parts {
				found := false
				// This algorithm implements title casing given word breaks
				// as defined in the Unicode standard 3.13 R3.
				for _, r := range s {
					title := unicode.ToTitle(r)
					lower := unicode.ToLower(r)
					if !found && title != lower {
						found = true
						want += string(title)
					} else {
						want += string(lower)
					}
				}
			}
			src := strings.Join(parts, "")
			got := Title(language.Und).String(src)
			if got != want {
				t.Errorf("got %q; want %q", got, want)
			}
		})
	}
}

func TestContext(t *testing.T) {
	tests := []struct {
		desc       string
		dstSize    int
		atEOF      bool
		src        string
		out        string
		nSrc       int
		err        error
		ops        string
		prefixArg  string
		prefixWant bool
	}{{
		desc:    "next: past end, atEOF, no checkpoint",
		dstSize: 10,
		atEOF:   true,
		src:     "12",
		out:     "",
		nSrc:    2,
		ops:     "next;next;next",
		// Test that calling prefix with a non-empty argument when the buffer
		// is depleted returns false.
		prefixArg:  "x",
		prefixWant: false,
	}, {
		desc:       "next: not at end, atEOF, no checkpoint",
		dstSize:    10,
		atEOF:      false,
		src:        "12",
		out:        "",
		nSrc:       0,
		err:        transform.ErrShortSrc,
		ops:        "next;next",
		prefixArg:  "",
		prefixWant: true,
	}, {
		desc:       "next: past end, !atEOF, no checkpoint",
		dstSize:    10,
		atEOF:      false,
		src:        "12",
		out:        "",
		nSrc:       0,
		err:        transform.ErrShortSrc,
		ops:        "next;next;next",
		prefixArg:  "",
		prefixWant: true,
	}, {
		desc:       "next: past end, !atEOF, checkpoint",
		dstSize:    10,
		atEOF:      false,
		src:        "12",
		out:        "",
		nSrc:       2,
		ops:        "next;next;checkpoint;next",
		prefixArg:  "",
		prefixWant: true,
	}, {
		desc:       "copy: exact count, atEOF, no checkpoint",
		dstSize:    2,
		atEOF:      true,
		src:        "12",
		out:        "12",
		nSrc:       2,
		ops:        "next;copy;next;copy;next",
		prefixArg:  "",
		prefixWant: true,
	}, {
		desc:       "copy: past end, !atEOF, no checkpoint",
		dstSize:    2,
		atEOF:      false,
		src:        "12",
		out:        "",
		nSrc:       0,
		err:        transform.ErrShortSrc,
		ops:        "next;copy;next;copy;next",
		prefixArg:  "",
		prefixWant: true,
	}, {
		desc:       "copy: past end, !atEOF, checkpoint",
		dstSize:    2,
		atEOF:      false,
		src:        "12",
		out:        "12",
		nSrc:       2,
		ops:        "next;copy;next;copy;checkpoint;next",
		prefixArg:  "",
		prefixWant: true,
	}, {
		desc:       "copy: short dst",
		dstSize:    1,
		atEOF:      false,
		src:        "12",
		out:        "",
		nSrc:       0,
		err:        transform.ErrShortDst,
		ops:        "next;copy;next;copy;checkpoint;next",
		prefixArg:  "12",
		prefixWant: false,
	}, {
		desc:       "copy: short dst, checkpointed",
		dstSize:    1,
		atEOF:      false,
		src:        "12",
		out:        "1",
		nSrc:       1,
		err:        transform.ErrShortDst,
		ops:        "next;copy;checkpoint;next;copy;next",
		prefixArg:  "",
		prefixWant: true,
	}, {
		desc:       "writeString: simple",
		dstSize:    3,
		atEOF:      true,
		src:        "1",
		out:        "1ab",
		nSrc:       1,
		ops:        "next;copy;writeab;next",
		prefixArg:  "",
		prefixWant: true,
	}, {
		desc:       "writeString: short dst",
		dstSize:    2,
		atEOF:      true,
		src:        "12",
		out:        "",
		nSrc:       0,
		err:        transform.ErrShortDst,
		ops:        "next;copy;writeab;next",
		prefixArg:  "2",
		prefixWant: true,
	}, {
		desc:       "writeString: simple",
		dstSize:    3,
		atEOF:      true,
		src:        "12",
		out:        "1ab",
		nSrc:       2,
		ops:        "next;copy;next;writeab;next",
		prefixArg:  "",
		prefixWant: true,
	}, {
		desc:       "writeString: short dst",
		dstSize:    2,
		atEOF:      true,
		src:        "12",
		out:        "",
		nSrc:       0,
		err:        transform.ErrShortDst,
		ops:        "next;copy;next;writeab;next",
		prefixArg:  "1",
		prefixWant: false,
	}, {
		desc:    "prefix",
		dstSize: 2,
		atEOF:   true,
		src:     "12",
		out:     "",
		nSrc:    0,
		// Context will assign an ErrShortSrc if the input wasn't exhausted.
		err:        transform.ErrShortSrc,
		prefixArg:  "12",
		prefixWant: true,
	}}
	for _, tt := range tests {
		c := context{dst: make([]byte, tt.dstSize), src: []byte(tt.src), atEOF: tt.atEOF}

		for _, op := range strings.Split(tt.ops, ";") {
			switch op {
			case "next":
				c.next()
			case "checkpoint":
				c.checkpoint()
			case "writeab":
				c.writeString("ab")
			case "copy":
				c.copy()
			case "":
			default:
				t.Fatalf("unknown op %q", op)
			}
		}
		if got := c.hasPrefix(tt.prefixArg); got != tt.prefixWant {
			t.Errorf("%s:\nprefix was %v; want %v", tt.desc, got, tt.prefixWant)
		}
		nDst, nSrc, err := c.ret()
		if err != tt.err {
			t.Errorf("%s:\nerror was %v; want %v", tt.desc, err, tt.err)
		}
		if out := string(c.dst[:nDst]); out != tt.out {
			t.Errorf("%s:\nout was %q; want %q", tt.desc, out, tt.out)
		}
		if nSrc != tt.nSrc {
			t.Errorf("%s:\nnSrc was %d; want %d", tt.desc, nSrc, tt.nSrc)
		}
	}
}
