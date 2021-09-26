// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cases

import (
	"bytes"
	"fmt"
	"path"
	"strings"
	"testing"
	"unicode/utf8"

	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/language"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

type testCase struct {
	lang  string
	src   interface{} // string, []string, or nil to skip test
	title interface{} // string, []string, or nil to skip test
	lower interface{} // string, []string, or nil to skip test
	upper interface{} // string, []string, or nil to skip test
	opts  options
}

var testCases = []testCase{
	0: {
		lang:  "und",
		src:   "abc aBc ABC abC İsıI ΕΣΆΣ",
		title: "Abc Abc Abc Abc İsıi Εσάσ",
		lower: "abc abc abc abc i\u0307sıi εσάσ",
		upper: "ABC ABC ABC ABC İSII ΕΣΆΣ",
		opts:  getOpts(HandleFinalSigma(false)),
	},

	1: {
		lang:  "und",
		src:   "abc aBc ABC abC İsıI ΕΣΆΣ Σ _Σ -Σ",
		title: "Abc Abc Abc Abc İsıi Εσάς Σ _Σ -Σ",
		lower: "abc abc abc abc i\u0307sıi εσάς σ _σ -σ",
		upper: "ABC ABC ABC ABC İSII ΕΣΆΣ Σ _Σ -Σ",
		opts:  getOpts(HandleFinalSigma(true)),
	},

	2: { // Title cased runes.
		lang:  supported,
		src:   "ǅA",
		title: "ǅa",
		lower: "ǆa",
		upper: "ǄA",
	},

	3: {
		// Title breaking.
		lang: supported,
		src: []string{
			"FOO CASE TEST",
			"DON'T DO THiS",
			"χωΡΊΣ χωΡΊΣ^a χωΡΊΣ:a χωΡΊΣ:^a χωΡΊΣ^ όμΩΣ Σ",
			"with-hyphens",
			"49ers 49ers",
			`"capitalize a^a -hyphen 0X _u a_u:a`,
			"MidNumLet a.b\u2018c\u2019d\u2024e\ufe52f\uff07f\uff0eg",
			"MidNum a,b;c\u037ed\u0589e\u060cf\u2044g\ufe50h",
			"\u0345 x\u3031x x\u05d0x \u05d0x a'.a a.a a4,a",
		},
		title: []string{
			"Foo Case Test",
			"Don't Do This",
			"Χωρίς Χωρίσ^A Χωρίσ:a Χωρίσ:^A Χωρίς^ Όμως Σ",
			"With-Hyphens",
			// Note that 49Ers is correct according to the spec.
			// TODO: provide some option to the user to treat different
			// characters as cased.
			"49Ers 49Ers",
			`"Capitalize A^A -Hyphen 0X _U A_u:a`,
			"Midnumlet A.b\u2018c\u2019d\u2024e\ufe52f\uff07f\uff0eg",
			"Midnum A,B;C\u037eD\u0589E\u060cF\u2044G\ufe50H",
			"\u0399 X\u3031X X\u05d0x \u05d0X A'.A A.a A4,A",
		},
	},

	// TODO: These are known deviations from the options{} Unicode Word Breaking
	// Algorithm.
	// {
	// 	"und",
	// 	"x_\u3031_x a4,4a",
	// 	"X_\u3031_x A4,4a", // Currently is "X_\U3031_X A4,4A".
	// 	"x_\u3031_x a4,4a",
	// 	"X_\u3031_X A4,4A",
	// 	options{},
	// },

	4: {
		// Tests title options
		lang:  "und",
		src:   "abc aBc ABC abC İsıI o'Brien",
		title: "Abc ABc ABC AbC İsıI O'Brien",
		opts:  getOpts(NoLower),
	},

	5: {
		lang:  "el",
		src:   "aBc ΟΔΌΣ Οδός Σο ΣΟ Σ oΣ ΟΣ σ ἕξ \u03ac",
		title: "Abc Οδός Οδός Σο Σο Σ Oς Ος Σ Ἕξ \u0386",
		lower: "abc οδός οδός σο σο σ oς ος σ ἕξ \u03ac",
		upper: "ABC ΟΔΟΣ ΟΔΟΣ ΣΟ ΣΟ Σ OΣ ΟΣ Σ ΕΞ \u0391", // Uppercase removes accents
	},

	6: {
		lang:  "tr az",
		src:   "Isiİ İsıI I\u0307sIiİ İsıI\u0307 I\u0300\u0307",
		title: "Isii İsıı I\u0307sıii İsıi I\u0300\u0307",
		lower: "ısii isıı isıii isıi \u0131\u0300\u0307",
		upper: "ISİİ İSII I\u0307SIİİ İSII\u0307 I\u0300\u0307",
	},

	7: {
		lang:  "lt",
		src:   "I Ï J J̈ Į Į̈ Ì Í Ĩ xi̇̈ xj̇̈ xį̇̈ xi̇̀ xi̇́ xi̇̃ XI XÏ XJ XJ̈ XĮ XĮ̈ XI̟̤",
		title: "I Ï J J̈ Į Į̈ Ì Í Ĩ Xi̇̈ Xj̇̈ Xį̇̈ Xi̇̀ Xi̇́ Xi̇̃ Xi Xi̇̈ Xj Xj̇̈ Xį Xį̇̈ Xi̟̤",
		lower: "i i̇̈ j j̇̈ į į̇̈ i̇̀ i̇́ i̇̃ xi̇̈ xj̇̈ xį̇̈ xi̇̀ xi̇́ xi̇̃ xi xi̇̈ xj xj̇̈ xį xį̇̈ xi̟̤",
		upper: "I Ï J J̈ Į Į̈ Ì Í Ĩ XÏ XJ̈ XĮ̈ XÌ XÍ XĨ XI XÏ XJ XJ̈ XĮ XĮ̈ XI̟̤",
	},

	8: {
		lang:  "lt",
		src:   "\u012e\u0300 \u00cc i\u0307\u0300 i\u0307\u0301 i\u0307\u0303 i\u0307\u0308 i\u0300\u0307",
		title: "\u012e\u0300 \u00cc \u00cc \u00cd \u0128 \u00cf I\u0300\u0307",
		lower: "\u012f\u0307\u0300 i\u0307\u0300 i\u0307\u0300 i\u0307\u0301 i\u0307\u0303 i\u0307\u0308 i\u0300\u0307",
		upper: "\u012e\u0300 \u00cc \u00cc \u00cd \u0128 \u00cf I\u0300\u0307",
	},

	9: {
		lang:  "nl",
		src:   "ijs IJs Ij Ijs İJ İJs aa aA 'ns 'S",
		title: "IJs IJs IJ IJs İj İjs Aa Aa 'ns 's",
	},

	// Note: this specification is not currently part of CLDR. The same holds
	// for the leading apostrophe handling for Dutch.
	// See https://unicode.org/cldr/trac/ticket/7078.
	10: {
		lang:  "af",
		src:   "wag 'n bietjie",
		title: "Wag 'n Bietjie",
		lower: "wag 'n bietjie",
		upper: "WAG 'N BIETJIE",
	},
}

func TestCaseMappings(t *testing.T) {
	for i, tt := range testCases {
		src, ok := tt.src.([]string)
		if !ok {
			src = strings.Split(tt.src.(string), " ")
		}

		for _, lang := range strings.Split(tt.lang, " ") {
			tag := language.MustParse(lang)
			testEntry := func(name string, mk func(language.Tag, options) transform.SpanningTransformer, gold interface{}) {
				c := Caser{mk(tag, tt.opts)}
				if gold != nil {
					wants, ok := gold.([]string)
					if !ok {
						wants = strings.Split(gold.(string), " ")
					}
					for j, want := range wants {
						if got := c.String(src[j]); got != want {
							t.Errorf("%d:%s:\n%s.String(%+q):\ngot  %+q;\nwant %+q", i, lang, name, src[j], got, want)
						}
					}
				}
				dst := make([]byte, 256) // big enough to hold any result
				src := []byte(strings.Join(src, " "))
				v := testtext.AllocsPerRun(20, func() {
					c.Transform(dst, src, true)
				})
				if v > 1.1 {
					t.Errorf("%d:%s:\n%s: number of allocs was %f; want 0", i, lang, name, v)
				}
			}
			testEntry("Upper", makeUpper, tt.upper)
			testEntry("Lower", makeLower, tt.lower)
			testEntry("Title", makeTitle, tt.title)
		}
	}
}

// TestAlloc tests that some mapping methods should not cause any allocation.
func TestAlloc(t *testing.T) {
	dst := make([]byte, 256) // big enough to hold any result
	src := []byte(txtNonASCII)

	for i, f := range []func() Caser{
		func() Caser { return Upper(language.Und) },
		func() Caser { return Lower(language.Und) },
		func() Caser { return Lower(language.Und, HandleFinalSigma(false)) },
		// TODO: use a shared copy for these casers as well, in order of
		// importance, starting with the most important:
		// func() Caser { return Title(language.Und) },
		// func() Caser { return Title(language.Und, HandleFinalSigma(false)) },
	} {
		testtext.Run(t, "", func(t *testing.T) {
			var c Caser
			v := testtext.AllocsPerRun(10, func() {
				c = f()
			})
			if v > 0 {
				// TODO: Right now only Upper has 1 allocation. Special-case Lower
				// and Title as well to have less allocations for the root locale.
				t.Errorf("%d:init: number of allocs was %f; want 0", i, v)
			}
			v = testtext.AllocsPerRun(2, func() {
				c.Transform(dst, src, true)
			})
			if v > 0 {
				t.Errorf("%d:transform: number of allocs was %f; want 0", i, v)
			}
		})
	}
}

func testHandover(t *testing.T, c Caser, src string) {
	want := c.String(src)
	// Find the common prefix.
	pSrc := 0
	for ; pSrc < len(src) && pSrc < len(want) && want[pSrc] == src[pSrc]; pSrc++ {
	}

	// Test handover for each substring of the prefix.
	for i := 0; i < pSrc; i++ {
		testtext.Run(t, fmt.Sprint("interleave/", i), func(t *testing.T) {
			dst := make([]byte, 4*len(src))
			c.Reset()
			nSpan, _ := c.Span([]byte(src[:i]), false)
			copy(dst, src[:nSpan])
			nTransform, _, _ := c.Transform(dst[nSpan:], []byte(src[nSpan:]), true)
			got := string(dst[:nSpan+nTransform])
			if got != want {
				t.Errorf("full string: got %q; want %q", got, want)
			}
		})
	}
}

func TestHandover(t *testing.T) {
	testCases := []struct {
		desc          string
		t             Caser
		first, second string
	}{{
		"title/nosigma/single midword",
		Title(language.Und, HandleFinalSigma(false)),
		"A.", "a",
	}, {
		"title/nosigma/single midword",
		Title(language.Und, HandleFinalSigma(false)),
		"A", ".a",
	}, {
		"title/nosigma/double midword",
		Title(language.Und, HandleFinalSigma(false)),
		"A..", "a",
	}, {
		"title/nosigma/double midword",
		Title(language.Und, HandleFinalSigma(false)),
		"A.", ".a",
	}, {
		"title/nosigma/double midword",
		Title(language.Und, HandleFinalSigma(false)),
		"A", "..a",
	}, {
		"title/sigma/single midword",
		Title(language.Und),
		"ΟΣ.", "a",
	}, {
		"title/sigma/single midword",
		Title(language.Und),
		"ΟΣ", ".a",
	}, {
		"title/sigma/double midword",
		Title(language.Und),
		"ΟΣ..", "a",
	}, {
		"title/sigma/double midword",
		Title(language.Und),
		"ΟΣ.", ".a",
	}, {
		"title/sigma/double midword",
		Title(language.Und),
		"ΟΣ", "..a",
	}, {
		"title/af/leading apostrophe",
		Title(language.Afrikaans),
		"'", "n bietje",
	}}
	for _, tc := range testCases {
		testtext.Run(t, tc.desc, func(t *testing.T) {
			src := tc.first + tc.second
			want := tc.t.String(src)
			tc.t.Reset()
			n, _ := tc.t.Span([]byte(tc.first), false)

			dst := make([]byte, len(want))
			copy(dst, tc.first[:n])

			nDst, _, _ := tc.t.Transform(dst[n:], []byte(src[n:]), true)
			got := string(dst[:n+nDst])
			if got != want {
				t.Errorf("got %q; want %q", got, want)
			}
		})
	}
}

// minBufSize is the size of the buffer by which the casing operation in
// this package are guaranteed to make progress.
const minBufSize = norm.MaxSegmentSize

type bufferTest struct {
	desc, src, want  string
	firstErr         error
	dstSize, srcSize int
	t                transform.SpanningTransformer
}

var bufferTests []bufferTest

func init() {
	bufferTests = []bufferTest{{
		desc:     "und/upper/short dst",
		src:      "abcdefg",
		want:     "ABCDEFG",
		firstErr: transform.ErrShortDst,
		dstSize:  3,
		srcSize:  minBufSize,
		t:        Upper(language.Und),
	}, {
		desc:     "und/upper/short src",
		src:      "123é56",
		want:     "123É56",
		firstErr: transform.ErrShortSrc,
		dstSize:  4,
		srcSize:  4,
		t:        Upper(language.Und),
	}, {
		desc:     "und/upper/no error on short",
		src:      "12",
		want:     "12",
		firstErr: nil,
		dstSize:  1,
		srcSize:  1,
		t:        Upper(language.Und),
	}, {
		desc:     "und/lower/short dst",
		src:      "ABCDEFG",
		want:     "abcdefg",
		firstErr: transform.ErrShortDst,
		dstSize:  3,
		srcSize:  minBufSize,
		t:        Lower(language.Und),
	}, {
		desc:     "und/lower/short src",
		src:      "123É56",
		want:     "123é56",
		firstErr: transform.ErrShortSrc,
		dstSize:  4,
		srcSize:  4,
		t:        Lower(language.Und),
	}, {
		desc:     "und/lower/no error on short",
		src:      "12",
		want:     "12",
		firstErr: nil,
		dstSize:  1,
		srcSize:  1,
		t:        Lower(language.Und),
	}, {
		desc:    "und/lower/simple (no final sigma)",
		src:     "ΟΣ ΟΣΣ",
		want:    "οσ οσσ",
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Lower(language.Und, HandleFinalSigma(false)),
	}, {
		desc:    "und/title/simple (no final sigma)",
		src:     "ΟΣ ΟΣΣ",
		want:    "Οσ Οσσ",
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Title(language.Und, HandleFinalSigma(false)),
	}, {
		desc:    "und/title/final sigma: no error",
		src:     "ΟΣ",
		want:    "Ος",
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Title(language.Und),
	}, {
		desc:     "und/title/final sigma: short source",
		src:      "ΟΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣ",
		want:     "Οσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσς",
		firstErr: transform.ErrShortSrc,
		dstSize:  minBufSize,
		srcSize:  10,
		t:        Title(language.Und),
	}, {
		desc:     "und/title/final sigma: short destination 1",
		src:      "ΟΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣ",
		want:     "Οσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσς",
		firstErr: transform.ErrShortDst,
		dstSize:  10,
		srcSize:  minBufSize,
		t:        Title(language.Und),
	}, {
		desc:     "und/title/final sigma: short destination 2",
		src:      "ΟΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣ",
		want:     "Οσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσς",
		firstErr: transform.ErrShortDst,
		dstSize:  9,
		srcSize:  minBufSize,
		t:        Title(language.Und),
	}, {
		desc:     "und/title/final sigma: short destination 3",
		src:      "ΟΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣΣ",
		want:     "Οσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσς",
		firstErr: transform.ErrShortDst,
		dstSize:  8,
		srcSize:  minBufSize,
		t:        Title(language.Und),
	}, {
		desc:     "und/title/clipped UTF-8 rune",
		src:      "σσσσσσσσσσσ",
		want:     "Σσσσσσσσσσσ",
		firstErr: transform.ErrShortSrc,
		dstSize:  minBufSize,
		srcSize:  5,
		t:        Title(language.Und),
	}, {
		desc:    "und/title/clipped UTF-8 rune atEOF",
		src:     "σσσ" + string([]byte{0xCF}),
		want:    "Σσσ" + string([]byte{0xCF}),
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Title(language.Und),
	}, {
		// Note: the choice to change the final sigma at the end in case of
		// too many case ignorables is arbitrary. The main reason for this
		// choice is that it results in simpler code.
		desc:    "und/title/final sigma: max ignorables",
		src:     "ΟΣ" + strings.Repeat(".", maxIgnorable) + "a",
		want:    "Οσ" + strings.Repeat(".", maxIgnorable) + "A",
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Title(language.Und),
	}, {
		// Note: the choice to change the final sigma at the end in case of
		// too many case ignorables is arbitrary. The main reason for this
		// choice is that it results in simpler code.
		desc:    "und/title/long string",
		src:     "AA" + strings.Repeat(".", maxIgnorable+1) + "a",
		want:    "Aa" + strings.Repeat(".", maxIgnorable+1) + "A",
		dstSize: minBufSize,
		srcSize: len("AA" + strings.Repeat(".", maxIgnorable+1)),
		t:       Title(language.Und),
	}, {
		// Note: the choice to change the final sigma at the end in case of
		// too many case ignorables is arbitrary. The main reason for this
		// choice is that it results in simpler code.
		desc:    "und/title/final sigma: too many ignorables",
		src:     "ΟΣ" + strings.Repeat(".", maxIgnorable+1) + "a",
		want:    "Ος" + strings.Repeat(".", maxIgnorable+1) + "A",
		dstSize: minBufSize,
		srcSize: len("ΟΣ" + strings.Repeat(".", maxIgnorable+1)),
		t:       Title(language.Und),
	}, {
		desc:    "und/title/final sigma: apostrophe",
		src:     "ΟΣ''a",
		want:    "Οσ''A",
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Title(language.Und),
	}, {
		desc:    "el/upper/max ignorables",
		src:     "ο" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0313",
		want:    "Ο" + strings.Repeat("\u0321", maxIgnorable-1),
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Upper(language.Greek),
	}, {
		desc:    "el/upper/too many ignorables",
		src:     "ο" + strings.Repeat("\u0321", maxIgnorable) + "\u0313",
		want:    "Ο" + strings.Repeat("\u0321", maxIgnorable) + "\u0313",
		dstSize: minBufSize,
		srcSize: len("ο" + strings.Repeat("\u0321", maxIgnorable)),
		t:       Upper(language.Greek),
	}, {
		desc:     "el/upper/short dst",
		src:      "123ο",
		want:     "123Ο",
		firstErr: transform.ErrShortDst,
		dstSize:  3,
		srcSize:  minBufSize,
		t:        Upper(language.Greek),
	}, {
		desc:    "lt/lower/max ignorables",
		src:     "I" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0300",
		want:    "i" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0307\u0300",
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Lower(language.Lithuanian),
	}, {
		desc:    "lt/lower/too many ignorables",
		src:     "I" + strings.Repeat("\u0321", maxIgnorable) + "\u0300",
		want:    "i" + strings.Repeat("\u0321", maxIgnorable) + "\u0300",
		dstSize: minBufSize,
		srcSize: len("I" + strings.Repeat("\u0321", maxIgnorable)),
		t:       Lower(language.Lithuanian),
	}, {
		desc:     "lt/lower/decomposition with short dst buffer 1",
		src:      "aaaaa\u00cc", // U+00CC LATIN CAPITAL LETTER I GRAVE
		firstErr: transform.ErrShortDst,
		want:     "aaaaai\u0307\u0300",
		dstSize:  5,
		srcSize:  minBufSize,
		t:        Lower(language.Lithuanian),
	}, {
		desc:     "lt/lower/decomposition with short dst buffer 2",
		src:      "aaaa\u00cc", // U+00CC LATIN CAPITAL LETTER I GRAVE
		firstErr: transform.ErrShortDst,
		want:     "aaaai\u0307\u0300",
		dstSize:  5,
		srcSize:  minBufSize,
		t:        Lower(language.Lithuanian),
	}, {
		desc:    "lt/upper/max ignorables",
		src:     "i" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0307\u0300",
		want:    "I" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0300",
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Upper(language.Lithuanian),
	}, {
		desc:    "lt/upper/too many ignorables",
		src:     "i" + strings.Repeat("\u0321", maxIgnorable) + "\u0307\u0300",
		want:    "I" + strings.Repeat("\u0321", maxIgnorable) + "\u0307\u0300",
		dstSize: minBufSize,
		srcSize: len("i" + strings.Repeat("\u0321", maxIgnorable)),
		t:       Upper(language.Lithuanian),
	}, {
		desc:     "lt/upper/short dst",
		src:      "12i\u0307\u0300",
		want:     "12\u00cc",
		firstErr: transform.ErrShortDst,
		dstSize:  3,
		srcSize:  minBufSize,
		t:        Upper(language.Lithuanian),
	}, {
		desc:    "aztr/lower/max ignorables",
		src:     "I" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0307\u0300",
		want:    "i" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0300",
		dstSize: minBufSize,
		srcSize: minBufSize,
		t:       Lower(language.Turkish),
	}, {
		desc:    "aztr/lower/too many ignorables",
		src:     "I" + strings.Repeat("\u0321", maxIgnorable) + "\u0307\u0300",
		want:    "\u0131" + strings.Repeat("\u0321", maxIgnorable) + "\u0307\u0300",
		dstSize: minBufSize,
		srcSize: len("I" + strings.Repeat("\u0321", maxIgnorable)),
		t:       Lower(language.Turkish),
	}, {
		desc:     "nl/title/pre-IJ cutoff",
		src:      "  ij",
		want:     "  IJ",
		firstErr: transform.ErrShortDst,
		dstSize:  2,
		srcSize:  minBufSize,
		t:        Title(language.Dutch),
	}, {
		desc:     "nl/title/mid-IJ cutoff",
		src:      "  ij",
		want:     "  IJ",
		firstErr: transform.ErrShortDst,
		dstSize:  3,
		srcSize:  minBufSize,
		t:        Title(language.Dutch),
	}, {
		desc:     "af/title/apostrophe",
		src:      "'n bietje",
		want:     "'n Bietje",
		firstErr: transform.ErrShortDst,
		dstSize:  3,
		srcSize:  minBufSize,
		t:        Title(language.Afrikaans),
	}}
}

func TestShortBuffersAndOverflow(t *testing.T) {
	for i, tt := range bufferTests {
		testtext.Run(t, tt.desc, func(t *testing.T) {
			buf := make([]byte, tt.dstSize)
			got := []byte{}
			var nSrc, nDst int
			var err error
			for p := 0; p < len(tt.src); p += nSrc {
				q := p + tt.srcSize
				if q > len(tt.src) {
					q = len(tt.src)
				}
				nDst, nSrc, err = tt.t.Transform(buf, []byte(tt.src[p:q]), q == len(tt.src))
				got = append(got, buf[:nDst]...)

				if p == 0 && err != tt.firstErr {
					t.Errorf("%d:%s:\n error was %v; want %v", i, tt.desc, err, tt.firstErr)
					break
				}
			}
			if string(got) != tt.want {
				t.Errorf("%d:%s:\ngot  %+q;\nwant %+q", i, tt.desc, got, tt.want)
			}
			testHandover(t, Caser{tt.t}, tt.src)
		})
	}
}

func TestSpan(t *testing.T) {
	for _, tt := range []struct {
		desc  string
		src   string
		want  string
		atEOF bool
		err   error
		t     Caser
	}{{
		desc:  "und/upper/basic",
		src:   "abcdefg",
		want:  "",
		atEOF: true,
		err:   transform.ErrEndOfSpan,
		t:     Upper(language.Und),
	}, {
		desc:  "und/upper/short src",
		src:   "123É"[:4],
		want:  "123",
		atEOF: false,
		err:   transform.ErrShortSrc,
		t:     Upper(language.Und),
	}, {
		desc:  "und/upper/no error on short",
		src:   "12",
		want:  "12",
		atEOF: false,
		t:     Upper(language.Und),
	}, {
		desc:  "und/lower/basic",
		src:   "ABCDEFG",
		want:  "",
		atEOF: true,
		err:   transform.ErrEndOfSpan,
		t:     Lower(language.Und),
	}, {
		desc:  "und/lower/short src num",
		src:   "123é"[:4],
		want:  "123",
		atEOF: false,
		err:   transform.ErrShortSrc,
		t:     Lower(language.Und),
	}, {
		desc:  "und/lower/short src greek",
		src:   "αβγé"[:7],
		want:  "αβγ",
		atEOF: false,
		err:   transform.ErrShortSrc,
		t:     Lower(language.Und),
	}, {
		desc:  "und/lower/no error on short",
		src:   "12",
		want:  "12",
		atEOF: false,
		t:     Lower(language.Und),
	}, {
		desc:  "und/lower/simple (no final sigma)",
		src:   "ος οσσ",
		want:  "οσ οσσ",
		atEOF: true,
		t:     Lower(language.Und, HandleFinalSigma(false)),
	}, {
		desc:  "und/title/simple (no final sigma)",
		src:   "Οσ Οσσ",
		want:  "Οσ Οσσ",
		atEOF: true,
		t:     Title(language.Und, HandleFinalSigma(false)),
	}, {
		desc: "und/lower/final sigma: no error",
		src:  "οΣ", // Oς
		want: "ο",  // Oς
		err:  transform.ErrEndOfSpan,
		t:    Lower(language.Und),
	}, {
		desc: "und/title/final sigma: no error",
		src:  "ΟΣ", // Oς
		want: "Ο",  // Oς
		err:  transform.ErrEndOfSpan,
		t:    Title(language.Und),
	}, {
		desc: "und/title/final sigma: no short source!",
		src:  "ΟσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσΣ",
		want: "Οσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσσ",
		err:  transform.ErrEndOfSpan,
		t:    Title(language.Und),
	}, {
		desc:  "und/title/clipped UTF-8 rune",
		src:   "Σσ" + string([]byte{0xCF}),
		want:  "Σσ",
		atEOF: false,
		err:   transform.ErrShortSrc,
		t:     Title(language.Und),
	}, {
		desc:  "und/title/clipped UTF-8 rune atEOF",
		src:   "Σσσ" + string([]byte{0xCF}),
		want:  "Σσσ" + string([]byte{0xCF}),
		atEOF: true,
		t:     Title(language.Und),
	}, {
		// Note: the choice to change the final sigma at the end in case of
		// too many case ignorables is arbitrary. The main reason for this
		// choice is that it results in simpler code.
		desc: "und/title/long string",
		src:  "A" + strings.Repeat("a", maxIgnorable+5),
		want: "A" + strings.Repeat("a", maxIgnorable+5),
		t:    Title(language.Und),
	}, {
		// Note: the choice to change the final sigma at the end in case of
		// too many case ignorables is arbitrary. The main reason for this
		// choice is that it results in simpler code.
		desc:  "und/title/cyrillic",
		src:   "При",
		want:  "При",
		atEOF: true,
		t:     Title(language.Und, HandleFinalSigma(false)),
	}, {
		// Note: the choice to change the final sigma at the end in case of
		// too many case ignorables is arbitrary. The main reason for this
		// choice is that it results in simpler code.
		desc: "und/title/final sigma: max ignorables",
		src:  "Οσ" + strings.Repeat(".", maxIgnorable) + "A",
		want: "Οσ" + strings.Repeat(".", maxIgnorable) + "A",
		t:    Title(language.Und),
	}, {
		desc: "el/upper/max ignorables - not implemented",
		src:  "Ο" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0313",
		want: "",
		err:  transform.ErrEndOfSpan,
		t:    Upper(language.Greek),
	}, {
		desc: "el/upper/too many ignorables - not implemented",
		src:  "Ο" + strings.Repeat("\u0321", maxIgnorable) + "\u0313",
		want: "",
		err:  transform.ErrEndOfSpan,
		t:    Upper(language.Greek),
	}, {
		desc: "el/upper/short dst",
		src:  "123ο",
		want: "",
		err:  transform.ErrEndOfSpan,
		t:    Upper(language.Greek),
	}, {
		desc: "lt/lower/max ignorables",
		src:  "i" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0307\u0300",
		want: "i" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0307\u0300",
		t:    Lower(language.Lithuanian),
	}, {
		desc: "lt/lower/isLower",
		src:  "I" + strings.Repeat("\u0321", maxIgnorable) + "\u0300",
		want: "",
		err:  transform.ErrEndOfSpan,
		t:    Lower(language.Lithuanian),
	}, {
		desc: "lt/lower/not identical",
		src:  "aaaaa\u00cc", // U+00CC LATIN CAPITAL LETTER I GRAVE
		err:  transform.ErrEndOfSpan,
		want: "aaaaa",
		t:    Lower(language.Lithuanian),
	}, {
		desc: "lt/lower/identical",
		src:  "aaaai\u0307\u0300", // U+00CC LATIN CAPITAL LETTER I GRAVE
		want: "aaaai\u0307\u0300",
		t:    Lower(language.Lithuanian),
	}, {
		desc: "lt/upper/not implemented",
		src:  "I" + strings.Repeat("\u0321", maxIgnorable-1) + "\u0300",
		want: "",
		err:  transform.ErrEndOfSpan,
		t:    Upper(language.Lithuanian),
	}, {
		desc: "lt/upper/not implemented, ascii",
		src:  "AB",
		want: "",
		err:  transform.ErrEndOfSpan,
		t:    Upper(language.Lithuanian),
	}, {
		desc: "nl/title/pre-IJ cutoff",
		src:  "  IJ",
		want: "  IJ",
		t:    Title(language.Dutch),
	}, {
		desc: "nl/title/mid-IJ cutoff",
		src:  "  Ia",
		want: "  Ia",
		t:    Title(language.Dutch),
	}, {
		desc: "af/title/apostrophe",
		src:  "'n Bietje",
		want: "'n Bietje",
		t:    Title(language.Afrikaans),
	}, {
		desc: "af/title/apostrophe-incorrect",
		src:  "'N Bietje",
		// The Single_Quote (a MidWord), needs to be retained as unspanned so
		// that a successive call to Transform can detect that N should not be
		// capitalized.
		want: "",
		err:  transform.ErrEndOfSpan,
		t:    Title(language.Afrikaans),
	}} {
		testtext.Run(t, tt.desc, func(t *testing.T) {
			for p := 0; p < len(tt.want); p += utf8.RuneLen([]rune(tt.src[p:])[0]) {
				tt.t.Reset()
				n, err := tt.t.Span([]byte(tt.src[:p]), false)
				if err != nil && err != transform.ErrShortSrc {
					t.Errorf("early failure:Span(%+q): %v (%d < %d)", tt.src[:p], err, n, len(tt.want))
					break
				}
			}
			tt.t.Reset()
			n, err := tt.t.Span([]byte(tt.src), tt.atEOF)
			if n != len(tt.want) || err != tt.err {
				t.Errorf("Span(%+q, %v): got %d, %v; want %d, %v", tt.src, tt.atEOF, n, err, len(tt.want), tt.err)
			}
			testHandover(t, tt.t, tt.src)
		})
	}
}

var txtASCII = strings.Repeat("The quick brown fox jumps over the lazy dog. ", 50)

// Taken from http://creativecommons.org/licenses/by-sa/3.0/vn/
const txt_vn = `Với các điều kiện sau: Ghi nhận công của tác giả.  Nếu bạn sử
dụng, chuyển đổi, hoặc xây dựng dự án từ  nội dung được chia sẻ này, bạn phải áp
dụng giấy phép này hoặc  một giấy phép khác có các điều khoản tương tự như giấy
phép này cho dự án của bạn. Hiểu rằng: Miễn — Bất kỳ các điều kiện nào trên đây
cũng có thể được miễn bỏ nếu bạn được sự cho phép của người sở hữu bản quyền.
Phạm vi công chúng — Khi tác phẩm hoặc bất kỳ chương nào của tác phẩm đã trong
vùng dành cho công chúng theo quy định của pháp luật thì tình trạng của nó không
bị ảnh hưởng bởi giấy phép trong bất kỳ trường hợp nào.`

// http://creativecommons.org/licenses/by-sa/2.5/cn/
const txt_cn = `您可以自由： 复制、发行、展览、表演、放映、
广播或通过信息网络传播本作品 创作演绎作品
对本作品进行商业性使用 惟须遵守下列条件：
署名 — 您必须按照作者或者许可人指定的方式对作品进行署名。
相同方式共享 — 如果您改变、转换本作品或者以本作品为基础进行创作，
您只能采用与本协议相同的许可协议发布基于本作品的演绎作品。`

// Taken from http://creativecommons.org/licenses/by-sa/1.0/deed.ru
const txt_ru = `При обязательном соблюдении следующих условий: Attribution — Вы
должны атрибутировать произведение (указывать автора и источник) в порядке,
предусмотренном автором или лицензиаром (но только так, чтобы никоим образом не
подразумевалось, что они поддерживают вас или использование вами данного
произведения). Υπό τις ακόλουθες προϋποθέσεις:`

// Taken from http://creativecommons.org/licenses/by-sa/3.0/gr/
const txt_gr = `Αναφορά Δημιουργού — Θα πρέπει να κάνετε την αναφορά στο έργο με
τον τρόπο που έχει οριστεί από το δημιουργό ή το χορηγούντο την άδεια (χωρίς
όμως να εννοείται με οποιονδήποτε τρόπο ότι εγκρίνουν εσάς ή τη χρήση του έργου
από εσάς). Παρόμοια Διανομή — Εάν αλλοιώσετε, τροποποιήσετε ή δημιουργήσετε
περαιτέρω βασισμένοι στο έργο θα μπορείτε να διανέμετε το έργο που θα προκύψει
μόνο με την ίδια ή παρόμοια άδεια.`

const txtNonASCII = txt_vn + txt_cn + txt_ru + txt_gr

// TODO: Improve ASCII performance.

func BenchmarkCasers(b *testing.B) {
	for _, s := range []struct{ name, text string }{
		{"ascii", txtASCII},
		{"nonASCII", txtNonASCII},
		{"short", "При"},
	} {
		src := []byte(s.text)
		// Measure case mappings in bytes package for comparison.
		for _, f := range []struct {
			name string
			fn   func(b []byte) []byte
		}{
			{"lower", bytes.ToLower},
			{"title", bytes.ToTitle},
			{"upper", bytes.ToUpper},
		} {
			testtext.Bench(b, path.Join(s.name, "bytes", f.name), func(b *testing.B) {
				b.SetBytes(int64(len(src)))
				for i := 0; i < b.N; i++ {
					f.fn(src)
				}
			})
		}
		for _, t := range []struct {
			name  string
			caser transform.SpanningTransformer
		}{
			{"fold/default", Fold()},
			{"upper/default", Upper(language.Und)},
			{"lower/sigma", Lower(language.Und)},
			{"lower/simple", Lower(language.Und, HandleFinalSigma(false))},
			{"title/sigma", Title(language.Und)},
			{"title/simple", Title(language.Und, HandleFinalSigma(false))},
		} {
			c := Caser{t.caser}
			dst := make([]byte, len(src))
			testtext.Bench(b, path.Join(s.name, t.name, "transform"), func(b *testing.B) {
				b.SetBytes(int64(len(src)))
				for i := 0; i < b.N; i++ {
					c.Reset()
					c.Transform(dst, src, true)
				}
			})
			// No need to check span for simple cases, as they will be the same
			// as sigma.
			if strings.HasSuffix(t.name, "/simple") {
				continue
			}
			spanSrc := c.Bytes(src)
			testtext.Bench(b, path.Join(s.name, t.name, "span"), func(b *testing.B) {
				c.Reset()
				if n, _ := c.Span(spanSrc, true); n < len(spanSrc) {
					b.Fatalf("spanner is not recognizing text %q as done (at %d)", spanSrc, n)
				}
				b.SetBytes(int64(len(spanSrc)))
				for i := 0; i < b.N; i++ {
					c.Reset()
					c.Span(spanSrc, true)
				}
			})
		}
	}
}
