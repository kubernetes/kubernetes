// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"unicode/utf8"

	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/transform"
)

var (
	testn = flag.Int("testn", -1, "specific test number to run or -1 for all")
)

// pc replaces any rune r that is repeated n times, for n > 1, with r{n}.
func pc(s string) []byte {
	b := bytes.NewBuffer(make([]byte, 0, len(s)))
	for i := 0; i < len(s); {
		r, sz := utf8.DecodeRuneInString(s[i:])
		n := 0
		if sz == 1 {
			// Special-case one-byte case to handle repetition for invalid UTF-8.
			for c := s[i]; i+n < len(s) && s[i+n] == c; n++ {
			}
		} else {
			for _, r2 := range s[i:] {
				if r2 != r {
					break
				}
				n++
			}
		}
		b.WriteString(s[i : i+sz])
		if n > 1 {
			fmt.Fprintf(b, "{%d}", n)
		}
		i += sz * n
	}
	return b.Bytes()
}

// pidx finds the index from which two strings start to differ, plus context.
// It returns the index and ellipsis if the index is greater than 0.
func pidx(a, b string) (i int, prefix string) {
	for ; i < len(a) && i < len(b) && a[i] == b[i]; i++ {
	}
	if i < 8 {
		return 0, ""
	}
	i -= 3 // ensure taking at least one full rune before the difference.
	for k := i - 7; i > k && !utf8.RuneStart(a[i]); i-- {
	}
	return i, "..."
}

type PositionTest struct {
	input  string
	pos    int
	buffer string // expected contents of reorderBuffer, if applicable
}

type positionFunc func(rb *reorderBuffer, s string) (int, []byte)

func runPosTests(t *testing.T, name string, f Form, fn positionFunc, tests []PositionTest) {
	rb := reorderBuffer{}
	rb.init(f, nil)
	for i, test := range tests {
		rb.reset()
		rb.src = inputString(test.input)
		rb.nsrc = len(test.input)
		pos, out := fn(&rb, test.input)
		if pos != test.pos {
			t.Errorf("%s:%d: position is %d; want %d", name, i, pos, test.pos)
		}
		if outs := string(out); outs != test.buffer {
			k, pfx := pidx(outs, test.buffer)
			t.Errorf("%s:%d: buffer \nwas  %s%+q; \nwant %s%+q", name, i, pfx, pc(outs[k:]), pfx, pc(test.buffer[k:]))
		}
	}
}

func grave(n int) string {
	return rep(0x0300, n)
}

func rep(r rune, n int) string {
	return strings.Repeat(string(r), n)
}

const segSize = maxByteBufferSize

var cgj = GraphemeJoiner

var decomposeSegmentTests = []PositionTest{
	// illegal runes
	{"\xC2", 0, ""},
	{"\xC0", 1, "\xC0"},
	{"\u00E0\x80", 2, "\u0061\u0300"},
	// starter
	{"a", 1, "a"},
	{"ab", 1, "a"},
	// starter + composing
	{"a\u0300", 3, "a\u0300"},
	{"a\u0300b", 3, "a\u0300"},
	// with decomposition
	{"\u00C0", 2, "A\u0300"},
	{"\u00C0b", 2, "A\u0300"},
	// long
	{grave(31), 60, grave(30) + cgj},
	{"a" + grave(31), 61, "a" + grave(30) + cgj},

	// Stability tests: see https://www.unicode.org/review/pr-29.html.
	// U+0300 COMBINING GRAVE ACCENT;Mn;230;NSM;;;;;N;NON-SPACING GRAVE;;;;
	// U+0B47 ORIYA VOWEL SIGN E;Mc;0;L;;;;;N;;;;;
	// U+0B3E ORIYA VOWEL SIGN AA;Mc;0;L;;;;;N;;;;;
	// U+1100 HANGUL CHOSEONG KIYEOK;Lo;0;L;;;;;N;;;;;
	// U+1161 HANGUL JUNGSEONG A;Lo;0;L;;;;;N;;;;;
	{"\u0B47\u0300\u0B3E", 8, "\u0B47\u0300\u0B3E"},
	{"\u1100\u0300\u1161", 8, "\u1100\u0300\u1161"},
	{"\u0B47\u0B3E", 6, "\u0B47\u0B3E"},
	{"\u1100\u1161", 6, "\u1100\u1161"},

	// U+04DA MALAYALAM VOWEL SIGN O;Mc;0;L;0D46 0D3E;;;;N;;;;;
	// Sequence of decomposing characters that are starters and modifiers.
	{"\u0d4a" + strings.Repeat("\u0d3e", 31), 90, "\u0d46" + strings.Repeat("\u0d3e", 30) + cgj},

	{grave(30), 60, grave(30)},
	// U+FF9E is a starter, but decomposes to U+3099, which is not.
	{grave(30) + "\uff9e", 60, grave(30) + cgj},
	// ends with incomplete UTF-8 encoding
	{"\xCC", 0, ""},
	{"\u0300\xCC", 2, "\u0300"},
}

func decomposeSegmentF(rb *reorderBuffer, s string) (int, []byte) {
	rb.initString(NFD, s)
	rb.setFlusher(nil, appendFlush)
	p := decomposeSegment(rb, 0, true)
	return p, rb.out
}

func TestDecomposeSegment(t *testing.T) {
	runPosTests(t, "TestDecomposeSegment", NFC, decomposeSegmentF, decomposeSegmentTests)
}

var firstBoundaryTests = []PositionTest{
	// no boundary
	{"", -1, ""},
	{"\u0300", -1, ""},
	{"\x80\x80", -1, ""},
	// illegal runes
	{"\xff", 0, ""},
	{"\u0300\xff", 2, ""},
	{"\u0300\xc0\x80\x80", 2, ""},
	// boundaries
	{"a", 0, ""},
	{"\u0300a", 2, ""},
	// Hangul
	{"\u1103\u1161", 0, ""},
	{"\u110B\u1173\u11B7", 0, ""},
	{"\u1161\u110B\u1173\u11B7", 3, ""},
	{"\u1173\u11B7\u1103\u1161", 6, ""},
	// too many combining characters.
	{grave(maxNonStarters - 1), -1, ""},
	{grave(maxNonStarters), 60, ""},
	{grave(maxNonStarters + 1), 60, ""},
}

func firstBoundaryF(rb *reorderBuffer, s string) (int, []byte) {
	return rb.f.form.FirstBoundary([]byte(s)), nil
}

func firstBoundaryStringF(rb *reorderBuffer, s string) (int, []byte) {
	return rb.f.form.FirstBoundaryInString(s), nil
}

func TestFirstBoundary(t *testing.T) {
	runPosTests(t, "TestFirstBoundary", NFC, firstBoundaryF, firstBoundaryTests)
	runPosTests(t, "TestFirstBoundaryInString", NFC, firstBoundaryStringF, firstBoundaryTests)
}

func TestNextBoundary(t *testing.T) {
	testCases := []struct {
		input string
		atEOF bool
		want  int
	}{
		// no boundary
		{"", true, 0},
		{"", false, -1},
		{"\u0300", true, 2},
		{"\u0300", false, -1},
		{"\x80\x80", true, 1},
		{"\x80\x80", false, 1},
		// illegal runes
		{"\xff", false, 1},
		{"\u0300\xff", false, 2},
		{"\u0300\xc0\x80\x80", false, 2},
		{"\xc2\x80\x80", false, 2},
		{"\xc2", false, -1},
		{"\xc2", true, 1},
		{"a\u0300\xc2", false, -1},
		{"a\u0300\xc2", true, 3},
		// boundaries
		{"a", true, 1},
		{"a", false, -1},
		{"aa", false, 1},
		{"\u0300", true, 2},
		{"\u0300", false, -1},
		{"\u0300a", false, 2},
		// Hangul
		{"\u1103\u1161", true, 6},
		{"\u1103\u1161", false, -1},
		{"\u110B\u1173\u11B7", false, -1},
		{"\u110B\u1173\u11B7\u110B\u1173\u11B7", false, 9},
		{"\u1161\u110B\u1173\u11B7", false, 3},
		{"\u1173\u11B7\u1103\u1161", false, 6},
		// too many combining characters.
		{grave(maxNonStarters - 1), false, -1},
		{grave(maxNonStarters), false, 60},
		{grave(maxNonStarters + 1), false, 60},
	}

	for _, tc := range testCases {
		if got := NFC.NextBoundary([]byte(tc.input), tc.atEOF); got != tc.want {
			t.Errorf("NextBoundary(%+q, %v) = %d; want %d", tc.input, tc.atEOF, got, tc.want)
		}
		if got := NFC.NextBoundaryInString(tc.input, tc.atEOF); got != tc.want {
			t.Errorf("NextBoundaryInString(%+q, %v) = %d; want %d", tc.input, tc.atEOF, got, tc.want)
		}
	}
}

var decomposeToLastTests = []PositionTest{
	// ends with inert character
	{"Hello!", 6, ""},
	{"\u0632", 2, ""},
	{"a\u0301\u0635", 5, ""},
	// ends with non-inert starter
	{"a", 0, "a"},
	{"a\u0301a", 3, "a"},
	{"a\u0301\u03B9", 3, "\u03B9"},
	{"a\u0327", 0, "a\u0327"},
	// illegal runes
	{"\xFF", 1, ""},
	{"aa\xFF", 3, ""},
	{"\xC0\x80\x80", 3, ""},
	{"\xCC\x80\x80", 3, ""},
	// ends with incomplete UTF-8 encoding
	{"a\xCC", 2, ""},
	// ends with combining characters
	{"\u0300\u0301", 0, "\u0300\u0301"},
	{"a\u0300\u0301", 0, "a\u0300\u0301"},
	{"a\u0301\u0308", 0, "a\u0301\u0308"},
	{"a\u0308\u0301", 0, "a\u0308\u0301"},
	{"aaaa\u0300\u0301", 3, "a\u0300\u0301"},
	{"\u0300a\u0300\u0301", 2, "a\u0300\u0301"},
	{"\u00C0", 0, "A\u0300"},
	{"a\u00C0", 1, "A\u0300"},
	// decomposing
	{"a\u0300\u00E0", 3, "a\u0300"},
	// multisegment decompositions (flushes leading segments)
	{"a\u0300\uFDC0", 7, "\u064A"},
	{"\uFDC0" + grave(29), 4, "\u064A" + grave(29)},
	{"\uFDC0" + grave(30), 4, "\u064A" + grave(30)},
	{"\uFDC0" + grave(31), 5, grave(30)},
	{"\uFDFA" + grave(14), 31, "\u0645" + grave(14)},
	// Overflow
	{"\u00E0" + grave(29), 0, "a" + grave(30)},
	{"\u00E0" + grave(30), 2, grave(30)},
	// Hangul
	{"a\u1103", 1, "\u1103"},
	{"a\u110B", 1, "\u110B"},
	{"a\u110B\u1173", 1, "\u110B\u1173"},
	// See comment in composition.go:compBoundaryAfter.
	{"a\u110B\u1173\u11B7", 1, "\u110B\u1173\u11B7"},
	{"a\uC73C", 1, "\u110B\u1173"},
	{"다음", 3, "\u110B\u1173\u11B7"},
	{"다", 0, "\u1103\u1161"},
	{"\u1103\u1161\u110B\u1173\u11B7", 6, "\u110B\u1173\u11B7"},
	{"\u110B\u1173\u11B7\u1103\u1161", 9, "\u1103\u1161"},
	{"다음음", 6, "\u110B\u1173\u11B7"},
	{"음다다", 6, "\u1103\u1161"},
	// maximized buffer
	{"a" + grave(30), 0, "a" + grave(30)},
	// Buffer overflow
	{"a" + grave(31), 3, grave(30)},
	// weird UTF-8
	{"a\u0300\u11B7", 0, "a\u0300\u11B7"},
}

func decomposeToLast(rb *reorderBuffer, s string) (int, []byte) {
	rb.setFlusher([]byte(s), appendFlush)
	decomposeToLastBoundary(rb)
	buf := rb.flush(nil)
	return len(rb.out), buf
}

func TestDecomposeToLastBoundary(t *testing.T) {
	runPosTests(t, "TestDecomposeToLastBoundary", NFKC, decomposeToLast, decomposeToLastTests)
}

var lastBoundaryTests = []PositionTest{
	// ends with inert character
	{"Hello!", 6, ""},
	{"\u0632", 2, ""},
	// ends with non-inert starter
	{"a", 0, ""},
	// illegal runes
	{"\xff", 1, ""},
	{"aa\xff", 3, ""},
	{"a\xff\u0300", 1, ""}, // TODO: should probably be 2.
	{"\xc0\x80\x80", 3, ""},
	{"\xc0\x80\x80\u0300", 3, ""},
	// ends with incomplete UTF-8 encoding
	{"\xCC", -1, ""},
	{"\xE0\x80", -1, ""},
	{"\xF0\x80\x80", -1, ""},
	{"a\xCC", 0, ""},
	{"\x80\xCC", 1, ""},
	{"\xCC\xCC", 1, ""},
	// ends with combining characters
	{"a\u0300\u0301", 0, ""},
	{"aaaa\u0300\u0301", 3, ""},
	{"\u0300a\u0300\u0301", 2, ""},
	{"\u00C2", 0, ""},
	{"a\u00C2", 1, ""},
	// decomposition may recombine
	{"\u0226", 0, ""},
	// no boundary
	{"", -1, ""},
	{"\u0300\u0301", -1, ""},
	{"\u0300", -1, ""},
	{"\x80\x80", -1, ""},
	{"\x80\x80\u0301", -1, ""},
	// Hangul
	{"다음", 3, ""},
	{"다", 0, ""},
	{"\u1103\u1161\u110B\u1173\u11B7", 6, ""},
	{"\u110B\u1173\u11B7\u1103\u1161", 9, ""},
	// too many combining characters.
	{grave(maxNonStarters - 1), -1, ""},
	// May still be preceded with a non-starter.
	{grave(maxNonStarters), -1, ""},
	// May still need to insert a cgj after the last combiner.
	{grave(maxNonStarters + 1), 2, ""},
	{grave(maxNonStarters + 2), 4, ""},

	{"a" + grave(maxNonStarters-1), 0, ""},
	{"a" + grave(maxNonStarters), 0, ""},
	// May still need to insert a cgj after the last combiner.
	{"a" + grave(maxNonStarters+1), 3, ""},
	{"a" + grave(maxNonStarters+2), 5, ""},
}

func lastBoundaryF(rb *reorderBuffer, s string) (int, []byte) {
	return rb.f.form.LastBoundary([]byte(s)), nil
}

func TestLastBoundary(t *testing.T) {
	runPosTests(t, "TestLastBoundary", NFC, lastBoundaryF, lastBoundaryTests)
}

type spanTest struct {
	input string
	atEOF bool
	n     int
	err   error
}

var quickSpanTests = []spanTest{
	{"", true, 0, nil},
	// starters
	{"a", true, 1, nil},
	{"abc", true, 3, nil},
	{"\u043Eb", true, 3, nil},
	// incomplete last rune.
	{"\xCC", true, 1, nil},
	{"\xCC", false, 0, transform.ErrShortSrc},
	{"a\xCC", true, 2, nil},
	{"a\xCC", false, 0, transform.ErrShortSrc}, // TODO: could be 1 for NFD
	// incorrectly ordered combining characters
	{"\u0300\u0316", true, 0, transform.ErrEndOfSpan},
	{"\u0300\u0316", false, 0, transform.ErrEndOfSpan},
	{"\u0300\u0316cd", true, 0, transform.ErrEndOfSpan},
	{"\u0300\u0316cd", false, 0, transform.ErrEndOfSpan},
	// have a maximum number of combining characters.
	{rep(0x035D, 30) + "\u035B", true, 0, transform.ErrEndOfSpan},
	{"a" + rep(0x035D, 30) + "\u035B", true, 0, transform.ErrEndOfSpan},
	{"Ɵ" + rep(0x035D, 30) + "\u035B", true, 0, transform.ErrEndOfSpan},
	{"aa" + rep(0x035D, 30) + "\u035B", true, 1, transform.ErrEndOfSpan},
	{rep(0x035D, 30) + cgj + "\u035B", true, 64, nil},
	{"a" + rep(0x035D, 30) + cgj + "\u035B", true, 65, nil},
	{"Ɵ" + rep(0x035D, 30) + cgj + "\u035B", true, 66, nil},
	{"aa" + rep(0x035D, 30) + cgj + "\u035B", true, 66, nil},

	{"a" + rep(0x035D, 30) + cgj + "\u035B", false, 61, transform.ErrShortSrc},
	{"Ɵ" + rep(0x035D, 30) + cgj + "\u035B", false, 62, transform.ErrShortSrc},
	{"aa" + rep(0x035D, 30) + cgj + "\u035B", false, 62, transform.ErrShortSrc},
}

var quickSpanNFDTests = []spanTest{
	// needs decomposing
	{"\u00C0", true, 0, transform.ErrEndOfSpan},
	{"abc\u00C0", true, 3, transform.ErrEndOfSpan},
	// correctly ordered combining characters
	{"\u0300", true, 2, nil},
	{"ab\u0300", true, 4, nil},
	{"ab\u0300cd", true, 6, nil},
	{"\u0300cd", true, 4, nil},
	{"\u0316\u0300", true, 4, nil},
	{"ab\u0316\u0300", true, 6, nil},
	{"ab\u0316\u0300cd", true, 8, nil},
	{"ab\u0316\u0300\u00C0", true, 6, transform.ErrEndOfSpan},
	{"\u0316\u0300cd", true, 6, nil},
	{"\u043E\u0308b", true, 5, nil},
	// incorrectly ordered combining characters
	{"ab\u0300\u0316", true, 1, transform.ErrEndOfSpan}, // TODO: we could skip 'b' as well.
	{"ab\u0300\u0316cd", true, 1, transform.ErrEndOfSpan},
	// Hangul
	{"같은", true, 0, transform.ErrEndOfSpan},
}

var quickSpanNFCTests = []spanTest{
	// okay composed
	{"\u00C0", true, 2, nil},
	{"abc\u00C0", true, 5, nil},
	// correctly ordered combining characters
	// TODO: b may combine with modifiers, which is why this fails. We could
	// make a more precise test that actually checks whether last
	// characters combines. Probably not worth it.
	{"ab\u0300", true, 1, transform.ErrEndOfSpan},
	{"ab\u0300cd", true, 1, transform.ErrEndOfSpan},
	{"ab\u0316\u0300", true, 1, transform.ErrEndOfSpan},
	{"ab\u0316\u0300cd", true, 1, transform.ErrEndOfSpan},
	{"\u00C0\u035D", true, 4, nil},
	// we do not special case leading combining characters
	{"\u0300cd", true, 0, transform.ErrEndOfSpan},
	{"\u0300", true, 0, transform.ErrEndOfSpan},
	{"\u0316\u0300", true, 0, transform.ErrEndOfSpan},
	{"\u0316\u0300cd", true, 0, transform.ErrEndOfSpan},
	// incorrectly ordered combining characters
	{"ab\u0300\u0316", true, 1, transform.ErrEndOfSpan},
	{"ab\u0300\u0316cd", true, 1, transform.ErrEndOfSpan},
	// Hangul
	{"같은", true, 6, nil},
	{"같은", false, 3, transform.ErrShortSrc},
	// We return the start of the violating segment in case of overflow.
	{grave(30) + "\uff9e", true, 0, transform.ErrEndOfSpan},
	{grave(30), true, 0, transform.ErrEndOfSpan},
}

func runSpanTests(t *testing.T, name string, f Form, testCases []spanTest) {
	for i, tc := range testCases {
		s := fmt.Sprintf("Bytes/%s/%d=%+q/atEOF=%v", name, i, pc(tc.input), tc.atEOF)
		ok := testtext.Run(t, s, func(t *testing.T) {
			n, err := f.Span([]byte(tc.input), tc.atEOF)
			if n != tc.n || err != tc.err {
				t.Errorf("\n got %d, %v;\nwant %d, %v", n, err, tc.n, tc.err)
			}
		})
		if !ok {
			continue // Don't do the String variant if the Bytes variant failed.
		}
		s = fmt.Sprintf("String/%s/%d=%+q/atEOF=%v", name, i, pc(tc.input), tc.atEOF)
		testtext.Run(t, s, func(t *testing.T) {
			n, err := f.SpanString(tc.input, tc.atEOF)
			if n != tc.n || err != tc.err {
				t.Errorf("\n got %d, %v;\nwant %d, %v", n, err, tc.n, tc.err)
			}
		})
	}
}

func TestSpan(t *testing.T) {
	runSpanTests(t, "NFD", NFD, quickSpanTests)
	runSpanTests(t, "NFD", NFD, quickSpanNFDTests)
	runSpanTests(t, "NFC", NFC, quickSpanTests)
	runSpanTests(t, "NFC", NFC, quickSpanNFCTests)
}

var isNormalTests = []PositionTest{
	{"", 1, ""},
	// illegal runes
	{"\xff", 1, ""},
	// starters
	{"a", 1, ""},
	{"abc", 1, ""},
	{"\u043Eb", 1, ""},
	// incorrectly ordered combining characters
	{"\u0300\u0316", 0, ""},
	{"ab\u0300\u0316", 0, ""},
	{"ab\u0300\u0316cd", 0, ""},
	{"\u0300\u0316cd", 0, ""},
}
var isNormalNFDTests = []PositionTest{
	// needs decomposing
	{"\u00C0", 0, ""},
	{"abc\u00C0", 0, ""},
	// correctly ordered combining characters
	{"\u0300", 1, ""},
	{"ab\u0300", 1, ""},
	{"ab\u0300cd", 1, ""},
	{"\u0300cd", 1, ""},
	{"\u0316\u0300", 1, ""},
	{"ab\u0316\u0300", 1, ""},
	{"ab\u0316\u0300cd", 1, ""},
	{"\u0316\u0300cd", 1, ""},
	{"\u043E\u0308b", 1, ""},
	// Hangul
	{"같은", 0, ""},
}
var isNormalNFCTests = []PositionTest{
	// okay composed
	{"\u00C0", 1, ""},
	{"abc\u00C0", 1, ""},
	// need reordering
	{"a\u0300", 0, ""},
	{"a\u0300cd", 0, ""},
	{"a\u0316\u0300", 0, ""},
	{"a\u0316\u0300cd", 0, ""},
	// correctly ordered combining characters
	{"ab\u0300", 1, ""},
	{"ab\u0300cd", 1, ""},
	{"ab\u0316\u0300", 1, ""},
	{"ab\u0316\u0300cd", 1, ""},
	{"\u00C0\u035D", 1, ""},
	{"\u0300", 1, ""},
	{"\u0316\u0300cd", 1, ""},
	// Hangul
	{"같은", 1, ""},
}

var isNormalNFKXTests = []PositionTest{
	// Special case.
	{"\u00BC", 0, ""},
}

func isNormalF(rb *reorderBuffer, s string) (int, []byte) {
	if rb.f.form.IsNormal([]byte(s)) {
		return 1, nil
	}
	return 0, nil
}

func isNormalStringF(rb *reorderBuffer, s string) (int, []byte) {
	if rb.f.form.IsNormalString(s) {
		return 1, nil
	}
	return 0, nil
}

func TestIsNormal(t *testing.T) {
	runPosTests(t, "TestIsNormalNFD1", NFD, isNormalF, isNormalTests)
	runPosTests(t, "TestIsNormalNFD2", NFD, isNormalF, isNormalNFDTests)
	runPosTests(t, "TestIsNormalNFC1", NFC, isNormalF, isNormalTests)
	runPosTests(t, "TestIsNormalNFC2", NFC, isNormalF, isNormalNFCTests)
	runPosTests(t, "TestIsNormalNFKD1", NFKD, isNormalF, isNormalTests)
	runPosTests(t, "TestIsNormalNFKD2", NFKD, isNormalF, isNormalNFDTests)
	runPosTests(t, "TestIsNormalNFKD3", NFKD, isNormalF, isNormalNFKXTests)
	runPosTests(t, "TestIsNormalNFKC1", NFKC, isNormalF, isNormalTests)
	runPosTests(t, "TestIsNormalNFKC2", NFKC, isNormalF, isNormalNFCTests)
	runPosTests(t, "TestIsNormalNFKC3", NFKC, isNormalF, isNormalNFKXTests)
}

func TestIsNormalString(t *testing.T) {
	runPosTests(t, "TestIsNormalNFD1", NFD, isNormalStringF, isNormalTests)
	runPosTests(t, "TestIsNormalNFD2", NFD, isNormalStringF, isNormalNFDTests)
	runPosTests(t, "TestIsNormalNFC1", NFC, isNormalStringF, isNormalTests)
	runPosTests(t, "TestIsNormalNFC2", NFC, isNormalStringF, isNormalNFCTests)
}

type AppendTest struct {
	left  string
	right string
	out   string
}

type appendFunc func(f Form, out []byte, s string) []byte

var fstr = []string{"NFC", "NFD", "NFKC", "NFKD"}

func runNormTests(t *testing.T, name string, fn appendFunc) {
	for f := NFC; f <= NFKD; f++ {
		runAppendTests(t, name, f, fn, normTests[f])
	}
}

func runAppendTests(t *testing.T, name string, f Form, fn appendFunc, tests []AppendTest) {
	for i, test := range tests {
		t.Run(fmt.Sprintf("%s/%d", fstr[f], i), func(t *testing.T) {
			id := pc(test.left + test.right)
			if *testn >= 0 && i != *testn {
				return
			}
			t.Run("fn", func(t *testing.T) {
				out := []byte(test.left)
				have := string(fn(f, out, test.right))
				if len(have) != len(test.out) {
					t.Errorf("%+q: length is %d; want %d (%+q vs %+q)", id, len(have), len(test.out), pc(have), pc(test.out))
				}
				if have != test.out {
					k, pf := pidx(have, test.out)
					t.Errorf("%+q:\nwas  %s%+q; \nwant %s%+q", id, pf, pc(have[k:]), pf, pc(test.out[k:]))
				}
			})

			// Bootstrap by normalizing input. Ensures that the various variants
			// behave the same.
			for g := NFC; g <= NFKD; g++ {
				if f == g {
					continue
				}
				t.Run(fstr[g], func(t *testing.T) {
					want := g.String(test.left + test.right)
					have := string(fn(g, g.AppendString(nil, test.left), test.right))
					if len(have) != len(want) {
						t.Errorf("%+q: length is %d; want %d (%+q vs %+q)", id, len(have), len(want), pc(have), pc(want))
					}
					if have != want {
						k, pf := pidx(have, want)
						t.Errorf("%+q:\nwas  %s%+q; \nwant %s%+q", id, pf, pc(have[k:]), pf, pc(want[k:]))
					}
				})
			}
		})
	}
}

var normTests = [][]AppendTest{
	appendTestsNFC,
	appendTestsNFD,
	appendTestsNFKC,
	appendTestsNFKD,
}

var appendTestsNFC = []AppendTest{
	{"", ascii, ascii},
	{"", txt_all, txt_all},
	{"\uff9e", grave(30), "\uff9e" + grave(29) + cgj + grave(1)},
	{grave(30), "\uff9e", grave(30) + cgj + "\uff9e"},

	// Tests designed for Iter.
	{ // ordering of non-composing combining characters
		"",
		"\u0305\u0316",
		"\u0316\u0305",
	},
	{ // segment overflow
		"",
		"a" + rep(0x0305, maxNonStarters+4) + "\u0316",
		"a" + rep(0x0305, maxNonStarters) + cgj + "\u0316" + rep(0x305, 4),
	},

	{ // Combine across non-blocking non-starters.
		// U+0327 COMBINING CEDILLA;Mn;202;NSM;;;;;N;NON-SPACING CEDILLA;;;;
		// U+0325 COMBINING RING BELOW;Mn;220;NSM;;;;;N;NON-SPACING RING BELOW;;;;
		"", "a\u0327\u0325", "\u1e01\u0327",
	},

	{ // Jamo V+T does not combine.
		"",
		"\u1161\u11a8",
		"\u1161\u11a8",
	},

	// Stability tests: see https://www.unicode.org/review/pr-29.html.
	{"", "\u0b47\u0300\u0b3e", "\u0b47\u0300\u0b3e"},
	{"", "\u1100\u0300\u1161", "\u1100\u0300\u1161"},
	{"", "\u0b47\u0b3e", "\u0b4b"},
	{"", "\u1100\u1161", "\uac00"},

	// U+04DA MALAYALAM VOWEL SIGN O;Mc;0;L;0D46 0D3E;;;;N;;;;;
	{ // 0d4a starts a new segment.
		"",
		"\u0d4a" + strings.Repeat("\u0d3e", 15) + "\u0d4a" + strings.Repeat("\u0d3e", 15),
		"\u0d4a" + strings.Repeat("\u0d3e", 15) + "\u0d4a" + strings.Repeat("\u0d3e", 15),
	},

	{ // Split combining characters.
		// TODO: don't insert CGJ before starters.
		"",
		"\u0d46" + strings.Repeat("\u0d3e", 31),
		"\u0d4a" + strings.Repeat("\u0d3e", 29) + cgj + "\u0d3e",
	},

	{ // Split combining characters.
		"",
		"\u0d4a" + strings.Repeat("\u0d3e", 30),
		"\u0d4a" + strings.Repeat("\u0d3e", 29) + cgj + "\u0d3e",
	},

	{ //  https://golang.org/issues/20079
		"",
		"\xeb\u0344",
		"\xeb\u0308\u0301",
	},

	{ //  https://golang.org/issues/20079
		"",
		"\uac00" + strings.Repeat("\u0300", 30),
		"\uac00" + strings.Repeat("\u0300", 29) + "\u034f\u0300",
	},

	{ //  https://golang.org/issues/20079
		"",
		"\xeb" + strings.Repeat("\u0300", 31),
		"\xeb" + strings.Repeat("\u0300", 30) + "\u034f\u0300",
	},
}

var appendTestsNFD = []AppendTest{
	// TODO: Move some of the tests here.
}

var appendTestsNFKC = []AppendTest{
	// empty buffers
	{"", "", ""},
	{"a", "", "a"},
	{"", "a", "a"},
	{"", "\u0041\u0307\u0304", "\u01E0"},
	// segment split across buffers
	{"", "a\u0300b", "\u00E0b"},
	{"a", "\u0300b", "\u00E0b"},
	{"a", "\u0300\u0316", "\u00E0\u0316"},
	{"a", "\u0316\u0300", "\u00E0\u0316"},
	{"a", "\u0300a\u0300", "\u00E0\u00E0"},
	{"a", "\u0300a\u0300a\u0300", "\u00E0\u00E0\u00E0"},
	{"a", "\u0300aaa\u0300aaa\u0300", "\u00E0aa\u00E0aa\u00E0"},
	{"a\u0300", "\u0327", "\u00E0\u0327"},
	{"a\u0327", "\u0300", "\u00E0\u0327"},
	{"a\u0316", "\u0300", "\u00E0\u0316"},
	{"\u0041\u0307", "\u0304", "\u01E0"},
	// Hangul
	{"", "\u110B\u1173", "\uC73C"},
	{"", "\u1103\u1161", "\uB2E4"},
	{"", "\u110B\u1173\u11B7", "\uC74C"},
	{"", "\u320E", "\x28\uAC00\x29"},
	{"", "\x28\u1100\u1161\x29", "\x28\uAC00\x29"},
	{"\u1103", "\u1161", "\uB2E4"},
	{"\u110B", "\u1173\u11B7", "\uC74C"},
	{"\u110B\u1173", "\u11B7", "\uC74C"},
	{"\uC73C", "\u11B7", "\uC74C"},
	// UTF-8 encoding split across buffers
	{"a\xCC", "\x80", "\u00E0"},
	{"a\xCC", "\x80b", "\u00E0b"},
	{"a\xCC", "\x80a\u0300", "\u00E0\u00E0"},
	{"a\xCC", "\x80\x80", "\u00E0\x80"},
	{"a\xCC", "\x80\xCC", "\u00E0\xCC"},
	{"a\u0316\xCC", "\x80a\u0316\u0300", "\u00E0\u0316\u00E0\u0316"},
	// ending in incomplete UTF-8 encoding
	{"", "\xCC", "\xCC"},
	{"a", "\xCC", "a\xCC"},
	{"a", "b\xCC", "ab\xCC"},
	{"\u0226", "\xCC", "\u0226\xCC"},
	// illegal runes
	{"", "\x80", "\x80"},
	{"", "\x80\x80\x80", "\x80\x80\x80"},
	{"", "\xCC\x80\x80\x80", "\xCC\x80\x80\x80"},
	{"", "a\x80", "a\x80"},
	{"", "a\x80\x80\x80", "a\x80\x80\x80"},
	{"", "a\x80\x80\x80\x80\x80\x80", "a\x80\x80\x80\x80\x80\x80"},
	{"a", "\x80\x80\x80", "a\x80\x80\x80"},
	// overflow
	{"", strings.Repeat("\x80", 33), strings.Repeat("\x80", 33)},
	{strings.Repeat("\x80", 33), "", strings.Repeat("\x80", 33)},
	{strings.Repeat("\x80", 33), strings.Repeat("\x80", 33), strings.Repeat("\x80", 66)},
	// overflow of combining characters
	{"", grave(34), grave(30) + cgj + grave(4)},
	{"", grave(36), grave(30) + cgj + grave(6)},
	{grave(29), grave(5), grave(30) + cgj + grave(4)},
	{grave(30), grave(4), grave(30) + cgj + grave(4)},
	{grave(30), grave(3), grave(30) + cgj + grave(3)},
	{grave(30) + "\xCC", "\x80", grave(30) + cgj + grave(1)},
	{"", "\uFDFA" + grave(14), "\u0635\u0644\u0649 \u0627\u0644\u0644\u0647 \u0639\u0644\u064a\u0647 \u0648\u0633\u0644\u0645" + grave(14)},
	{"", "\uFDFA" + grave(28) + "\u0316", "\u0635\u0644\u0649 \u0627\u0644\u0644\u0647 \u0639\u0644\u064a\u0647 \u0648\u0633\u0644\u0645\u0316" + grave(28)},
	// - First rune has a trailing non-starter.
	{"\u00d5", grave(30), "\u00d5" + grave(29) + cgj + grave(1)},
	// - U+FF9E decomposes into a non-starter in compatibility mode. A CGJ must be
	//   inserted even when FF9E starts a new segment.
	{"\uff9e", grave(30), "\u3099" + grave(29) + cgj + grave(1)},
	{grave(30), "\uff9e", grave(30) + cgj + "\u3099"},
	// - Many non-starter decompositions in a row causing overflow.
	{"", rep(0x340, 31), rep(0x300, 30) + cgj + "\u0300"},
	{"", rep(0xFF9E, 31), rep(0x3099, 30) + cgj + "\u3099"},

	{"", "\u0644\u0625" + rep(0x300, 31), "\u0644\u0625" + rep(0x300, 29) + cgj + "\u0300\u0300"},
	{"", "\ufef9" + rep(0x300, 31), "\u0644\u0625" + rep(0x300, 29) + cgj + rep(0x0300, 2)},
	{"", "\ufef9" + rep(0x300, 31), "\u0644\u0625" + rep(0x300, 29) + cgj + rep(0x0300, 2)},

	// U+0F81 TIBETAN VOWEL SIGN REVERSED II splits into two modifiers.
	{"", "\u0f7f" + rep(0xf71, 29) + "\u0f81", "\u0f7f" + rep(0xf71, 29) + cgj + "\u0f71\u0f80"},
	{"", "\u0f7f" + rep(0xf71, 28) + "\u0f81", "\u0f7f" + rep(0xf71, 29) + "\u0f80"},
	{"", "\u0f7f" + rep(0xf81, 16), "\u0f7f" + rep(0xf71, 15) + rep(0xf80, 15) + cgj + "\u0f71\u0f80"},

	// weird UTF-8
	{"\u00E0\xE1", "\x86", "\u00E0\xE1\x86"},
	{"a\u0300\u11B7", "\u0300", "\u00E0\u11B7\u0300"},
	{"a\u0300\u11B7\u0300", "\u0300", "\u00E0\u11B7\u0300\u0300"},
	{"\u0300", "\xF8\x80\x80\x80\x80\u0300", "\u0300\xF8\x80\x80\x80\x80\u0300"},
	{"\u0300", "\xFC\x80\x80\x80\x80\x80\u0300", "\u0300\xFC\x80\x80\x80\x80\x80\u0300"},
	{"\xF8\x80\x80\x80\x80\u0300", "\u0300", "\xF8\x80\x80\x80\x80\u0300\u0300"},
	{"\xFC\x80\x80\x80\x80\x80\u0300", "\u0300", "\xFC\x80\x80\x80\x80\x80\u0300\u0300"},
	{"\xF8\x80\x80\x80", "\x80\u0300\u0300", "\xF8\x80\x80\x80\x80\u0300\u0300"},

	{"", strings.Repeat("a\u0316\u0300", 6), strings.Repeat("\u00E0\u0316", 6)},
	// large input.
	{"", strings.Repeat("a\u0300\u0316", 31), strings.Repeat("\u00E0\u0316", 31)},
	{"", strings.Repeat("a\u0300\u0316", 4000), strings.Repeat("\u00E0\u0316", 4000)},
	{"", strings.Repeat("\x80\x80", 4000), strings.Repeat("\x80\x80", 4000)},
	{"", "\u0041\u0307\u0304", "\u01E0"},
}

var appendTestsNFKD = []AppendTest{
	{"", "a" + grave(64), "a" + grave(30) + cgj + grave(30) + cgj + grave(4)},

	{ // segment overflow on unchanged character
		"",
		"a" + grave(64) + "\u0316",
		"a" + grave(30) + cgj + grave(30) + cgj + "\u0316" + grave(4),
	},
	{ // segment overflow on unchanged character + start value
		"",
		"a" + grave(98) + "\u0316",
		"a" + grave(30) + cgj + grave(30) + cgj + grave(30) + cgj + "\u0316" + grave(8),
	},
	{ // segment overflow on decomposition. (U+0340 decomposes to U+0300.)
		"",
		"a" + grave(59) + "\u0340",
		"a" + grave(30) + cgj + grave(30),
	},
	{ // segment overflow on non-starter decomposition
		"",
		"a" + grave(33) + "\u0340" + grave(30) + "\u0320",
		"a" + grave(30) + cgj + grave(30) + cgj + "\u0320" + grave(4),
	},
	{ // start value after ASCII overflow
		"",
		rep('a', segSize) + grave(32) + "\u0320",
		rep('a', segSize) + grave(30) + cgj + "\u0320" + grave(2),
	},
	{ // Jamo overflow
		"",
		"\u1100\u1161" + grave(30) + "\u0320" + grave(2),
		"\u1100\u1161" + grave(29) + cgj + "\u0320" + grave(3),
	},
	{ // Hangul
		"",
		"\uac00",
		"\u1100\u1161",
	},
	{ // Hangul overflow
		"",
		"\uac00" + grave(32) + "\u0320",
		"\u1100\u1161" + grave(29) + cgj + "\u0320" + grave(3),
	},
	{ // Hangul overflow in Hangul mode.
		"",
		"\uac00\uac00" + grave(32) + "\u0320",
		"\u1100\u1161\u1100\u1161" + grave(29) + cgj + "\u0320" + grave(3),
	},
	{ // Hangul overflow in Hangul mode.
		"",
		strings.Repeat("\uac00", 3) + grave(32) + "\u0320",
		strings.Repeat("\u1100\u1161", 3) + grave(29) + cgj + "\u0320" + grave(3),
	},
	{ // start value after cc=0
		"",
		"您您" + grave(34) + "\u0320",
		"您您" + grave(30) + cgj + "\u0320" + grave(4),
	},
	{ // start value after normalization
		"",
		"\u0300\u0320a" + grave(34) + "\u0320",
		"\u0320\u0300a" + grave(30) + cgj + "\u0320" + grave(4),
	},
	{
		// U+0F81 TIBETAN VOWEL SIGN REVERSED II splits into two modifiers.
		"",
		"a\u0f7f" + rep(0xf71, 29) + "\u0f81",
		"a\u0f7f" + rep(0xf71, 29) + cgj + "\u0f71\u0f80",
	},
}

func TestAppend(t *testing.T) {
	runNormTests(t, "Append", func(f Form, out []byte, s string) []byte {
		return f.Append(out, []byte(s)...)
	})
}

func TestAppendString(t *testing.T) {
	runNormTests(t, "AppendString", func(f Form, out []byte, s string) []byte {
		return f.AppendString(out, s)
	})
}

func TestBytes(t *testing.T) {
	runNormTests(t, "Bytes", func(f Form, out []byte, s string) []byte {
		buf := []byte{}
		buf = append(buf, out...)
		buf = append(buf, s...)
		return f.Bytes(buf)
	})
}

func TestString(t *testing.T) {
	runNormTests(t, "String", func(f Form, out []byte, s string) []byte {
		outs := string(out) + s
		return []byte(f.String(outs))
	})
}

func runNM(code string) (string, error) {
	// Write the file.
	tmpdir, err := ioutil.TempDir(os.TempDir(), "normalize_test")
	if err != nil {
		return "", fmt.Errorf("failed to create tmpdir: %v", err)
	}
	defer os.RemoveAll(tmpdir)
	goTool := filepath.Join(runtime.GOROOT(), "bin", "go")
	filename := filepath.Join(tmpdir, "main.go")
	if err := ioutil.WriteFile(filename, []byte(code), 0644); err != nil {
		return "", fmt.Errorf("failed to write main.go: %v", err)
	}
	outputFile := filepath.Join(tmpdir, "main")

	// Build the binary.
	out, err := exec.Command(goTool, "build", "-o", outputFile, filename).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to execute command: %v", err)
	}

	// Get the symbols.
	out, err = exec.Command(goTool, "tool", "nm", outputFile).CombinedOutput()
	return string(out), err
}

func TestLinking(t *testing.T) {
	const prog = `
	package main
	import "fmt"
	import "golang.org/x/text/unicode/norm"
	func main() { fmt.Println(norm.%s) }
	`

	baseline, errB := runNM(fmt.Sprintf(prog, "MaxSegmentSize"))
	withTables, errT := runNM(fmt.Sprintf(prog, `NFC.String("")`))
	if errB != nil || errT != nil {
		t.Skipf("TestLinking failed: %v and %v", errB, errT)
	}

	symbols := []string{"norm.formTable", "norm.nfkcValues", "norm.decomps"}
	for _, symbol := range symbols {
		if strings.Contains(baseline, symbol) {
			t.Errorf("found: %q unexpectedly", symbol)
		}
		if !strings.Contains(withTables, symbol) {
			t.Errorf("didn't find: %q unexpectedly", symbol)
		}
	}
}

func appendBench(f Form, in []byte) func() {
	buf := make([]byte, 0, 4*len(in))
	return func() {
		f.Append(buf, in...)
	}
}

func bytesBench(f Form, in []byte) func() {
	return func() {
		f.Bytes(in)
	}
}

func iterBench(f Form, in []byte) func() {
	iter := Iter{}
	return func() {
		iter.Init(f, in)
		for !iter.Done() {
			iter.Next()
		}
	}
}

func transformBench(f Form, in []byte) func() {
	buf := make([]byte, 4*len(in))
	return func() {
		if _, n, err := f.Transform(buf, in, true); err != nil || len(in) != n {
			log.Panic(n, len(in), err)
		}
	}
}

func readerBench(f Form, in []byte) func() {
	buf := make([]byte, 4*len(in))
	return func() {
		r := f.Reader(bytes.NewReader(in))
		var err error
		for err == nil {
			_, err = r.Read(buf)
		}
		if err != io.EOF {
			panic("")
		}
	}
}

func writerBench(f Form, in []byte) func() {
	buf := make([]byte, 0, 4*len(in))
	return func() {
		r := f.Writer(bytes.NewBuffer(buf))
		if _, err := r.Write(in); err != nil {
			panic("")
		}
	}
}

func appendBenchmarks(bm []func(), f Form, in []byte) []func() {
	bm = append(bm, appendBench(f, in))
	bm = append(bm, iterBench(f, in))
	bm = append(bm, transformBench(f, in))
	bm = append(bm, readerBench(f, in))
	bm = append(bm, writerBench(f, in))
	return bm
}

func doFormBenchmark(b *testing.B, inf, f Form, s string) {
	b.StopTimer()
	in := inf.Bytes([]byte(s))
	bm := appendBenchmarks(nil, f, in)
	b.SetBytes(int64(len(in) * len(bm)))
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		for _, fn := range bm {
			fn()
		}
	}
}

func doSingle(b *testing.B, f func(Form, []byte) func(), s []byte) {
	b.StopTimer()
	fn := f(NFC, s)
	b.SetBytes(int64(len(s)))
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		fn()
	}
}

var (
	smallNoChange = []byte("nörmalization")
	smallChange   = []byte("No\u0308rmalization")
	ascii         = strings.Repeat("There is nothing to change here! ", 500)
)

func lowerBench(f Form, in []byte) func() {
	// Use package strings instead of bytes as it doesn't allocate memory
	// if there aren't any changes.
	s := string(in)
	return func() {
		strings.ToLower(s)
	}
}

func BenchmarkLowerCaseNoChange(b *testing.B) {
	doSingle(b, lowerBench, smallNoChange)
}
func BenchmarkLowerCaseChange(b *testing.B) {
	doSingle(b, lowerBench, smallChange)
}

func quickSpanBench(f Form, in []byte) func() {
	return func() {
		f.QuickSpan(in)
	}
}

func BenchmarkQuickSpanChangeNFC(b *testing.B) {
	doSingle(b, quickSpanBench, smallNoChange)
}

func BenchmarkBytesNoChangeNFC(b *testing.B) {
	doSingle(b, bytesBench, smallNoChange)
}
func BenchmarkBytesChangeNFC(b *testing.B) {
	doSingle(b, bytesBench, smallChange)
}

func BenchmarkAppendNoChangeNFC(b *testing.B) {
	doSingle(b, appendBench, smallNoChange)
}
func BenchmarkAppendChangeNFC(b *testing.B) {
	doSingle(b, appendBench, smallChange)
}
func BenchmarkAppendLargeNFC(b *testing.B) {
	doSingle(b, appendBench, txt_all_bytes)
}

func BenchmarkIterNoChangeNFC(b *testing.B) {
	doSingle(b, iterBench, smallNoChange)
}
func BenchmarkIterChangeNFC(b *testing.B) {
	doSingle(b, iterBench, smallChange)
}
func BenchmarkIterLargeNFC(b *testing.B) {
	doSingle(b, iterBench, txt_all_bytes)
}

func BenchmarkTransformNoChangeNFC(b *testing.B) {
	doSingle(b, transformBench, smallNoChange)
}
func BenchmarkTransformChangeNFC(b *testing.B) {
	doSingle(b, transformBench, smallChange)
}
func BenchmarkTransformLargeNFC(b *testing.B) {
	doSingle(b, transformBench, txt_all_bytes)
}

func BenchmarkNormalizeAsciiNFC(b *testing.B) {
	doFormBenchmark(b, NFC, NFC, ascii)
}
func BenchmarkNormalizeAsciiNFD(b *testing.B) {
	doFormBenchmark(b, NFC, NFD, ascii)
}
func BenchmarkNormalizeAsciiNFKC(b *testing.B) {
	doFormBenchmark(b, NFC, NFKC, ascii)
}
func BenchmarkNormalizeAsciiNFKD(b *testing.B) {
	doFormBenchmark(b, NFC, NFKD, ascii)
}

func BenchmarkNormalizeNFC2NFC(b *testing.B) {
	doFormBenchmark(b, NFC, NFC, txt_all)
}
func BenchmarkNormalizeNFC2NFD(b *testing.B) {
	doFormBenchmark(b, NFC, NFD, txt_all)
}
func BenchmarkNormalizeNFD2NFC(b *testing.B) {
	doFormBenchmark(b, NFD, NFC, txt_all)
}
func BenchmarkNormalizeNFD2NFD(b *testing.B) {
	doFormBenchmark(b, NFD, NFD, txt_all)
}

// Hangul is often special-cased, so we test it separately.
func BenchmarkNormalizeHangulNFC2NFC(b *testing.B) {
	doFormBenchmark(b, NFC, NFC, txt_kr)
}
func BenchmarkNormalizeHangulNFC2NFD(b *testing.B) {
	doFormBenchmark(b, NFC, NFD, txt_kr)
}
func BenchmarkNormalizeHangulNFD2NFC(b *testing.B) {
	doFormBenchmark(b, NFD, NFC, txt_kr)
}
func BenchmarkNormalizeHangulNFD2NFD(b *testing.B) {
	doFormBenchmark(b, NFD, NFD, txt_kr)
}

var forms = []Form{NFC, NFD, NFKC, NFKD}

func doTextBenchmark(b *testing.B, s string) {
	b.StopTimer()
	in := []byte(s)
	bm := []func(){}
	for _, f := range forms {
		bm = appendBenchmarks(bm, f, in)
	}
	b.SetBytes(int64(len(s) * len(bm)))
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		for _, f := range bm {
			f()
		}
	}
}

func BenchmarkCanonicalOrdering(b *testing.B) {
	doTextBenchmark(b, txt_canon)
}
func BenchmarkExtendedLatin(b *testing.B) {
	doTextBenchmark(b, txt_vn)
}
func BenchmarkMiscTwoByteUtf8(b *testing.B) {
	doTextBenchmark(b, twoByteUtf8)
}
func BenchmarkMiscThreeByteUtf8(b *testing.B) {
	doTextBenchmark(b, threeByteUtf8)
}
func BenchmarkHangul(b *testing.B) {
	doTextBenchmark(b, txt_kr)
}
func BenchmarkJapanese(b *testing.B) {
	doTextBenchmark(b, txt_jp)
}
func BenchmarkChinese(b *testing.B) {
	doTextBenchmark(b, txt_cn)
}
func BenchmarkOverflow(b *testing.B) {
	doTextBenchmark(b, overflow)
}

var overflow = string(bytes.Repeat([]byte("\u035D"), 4096)) + "\u035B"

// Tests sampled from the Canonical ordering tests (Part 2) of
// https://unicode.org/Public/UNIDATA/NormalizationTest.txt
const txt_canon = `\u0061\u0315\u0300\u05AE\u0300\u0062 \u0061\u0300\u0315\u0300\u05AE\u0062
\u0061\u0302\u0315\u0300\u05AE\u0062 \u0061\u0307\u0315\u0300\u05AE\u0062
\u0061\u0315\u0300\u05AE\u030A\u0062 \u0061\u059A\u0316\u302A\u031C\u0062
\u0061\u032E\u059A\u0316\u302A\u0062 \u0061\u0338\u093C\u0334\u0062 
\u0061\u059A\u0316\u302A\u0339       \u0061\u0341\u0315\u0300\u05AE\u0062
\u0061\u0348\u059A\u0316\u302A\u0062 \u0061\u0361\u0345\u035D\u035C\u0062
\u0061\u0366\u0315\u0300\u05AE\u0062 \u0061\u0315\u0300\u05AE\u0486\u0062
\u0061\u05A4\u059A\u0316\u302A\u0062 \u0061\u0315\u0300\u05AE\u0613\u0062
\u0061\u0315\u0300\u05AE\u0615\u0062 \u0061\u0617\u0315\u0300\u05AE\u0062
\u0061\u0619\u0618\u064D\u064E\u0062 \u0061\u0315\u0300\u05AE\u0654\u0062
\u0061\u0315\u0300\u05AE\u06DC\u0062 \u0061\u0733\u0315\u0300\u05AE\u0062
\u0061\u0744\u059A\u0316\u302A\u0062 \u0061\u0315\u0300\u05AE\u0745\u0062
\u0061\u09CD\u05B0\u094D\u3099\u0062 \u0061\u0E38\u0E48\u0E38\u0C56\u0062
\u0061\u0EB8\u0E48\u0E38\u0E49\u0062 \u0061\u0F72\u0F71\u0EC8\u0F71\u0062
\u0061\u1039\u05B0\u094D\u3099\u0062 \u0061\u05B0\u094D\u3099\u1A60\u0062
\u0061\u3099\u093C\u0334\u1BE6\u0062 \u0061\u3099\u093C\u0334\u1C37\u0062
\u0061\u1CD9\u059A\u0316\u302A\u0062 \u0061\u2DED\u0315\u0300\u05AE\u0062
\u0061\u2DEF\u0315\u0300\u05AE\u0062 \u0061\u302D\u302E\u059A\u0316\u0062`

// Taken from http://creativecommons.org/licenses/by-sa/3.0/vn/
const txt_vn = `Với các điều kiện sau: Ghi nhận công của tác giả. 
Nếu bạn sử dụng, chuyển đổi, hoặc xây dựng dự án từ 
nội dung được chia sẻ này, bạn phải áp dụng giấy phép này hoặc 
một giấy phép khác có các điều khoản tương tự như giấy phép này
cho dự án của bạn. Hiểu rằng: Miễn — Bất kỳ các điều kiện nào
trên đây cũng có thể được miễn bỏ nếu bạn được sự cho phép của
người sở hữu bản quyền. Phạm vi công chúng — Khi tác phẩm hoặc
bất kỳ chương nào của tác phẩm đã trong vùng dành cho công
chúng theo quy định của pháp luật thì tình trạng của nó không 
bị ảnh hưởng bởi giấy phép trong bất kỳ trường hợp nào.`

// Taken from http://creativecommons.org/licenses/by-sa/1.0/deed.ru
const txt_ru = `При обязательном соблюдении следующих условий:
Attribution — Вы должны атрибутировать произведение (указывать
автора и источник) в порядке, предусмотренном автором или
лицензиаром (но только так, чтобы никоим образом не подразумевалось,
что они поддерживают вас или использование вами данного произведения).
Υπό τις ακόλουθες προϋποθέσεις:`

// Taken from http://creativecommons.org/licenses/by-sa/3.0/gr/
const txt_gr = `Αναφορά Δημιουργού — Θα πρέπει να κάνετε την αναφορά στο έργο με τον
τρόπο που έχει οριστεί από το δημιουργό ή το χορηγούντο την άδεια
(χωρίς όμως να εννοείται με οποιονδήποτε τρόπο ότι εγκρίνουν εσάς ή
τη χρήση του έργου από εσάς). Παρόμοια Διανομή — Εάν αλλοιώσετε,
τροποποιήσετε ή δημιουργήσετε περαιτέρω βασισμένοι στο έργο θα
μπορείτε να διανέμετε το έργο που θα προκύψει μόνο με την ίδια ή
παρόμοια άδεια.`

// Taken from http://creativecommons.org/licenses/by-sa/3.0/deed.ar
const txt_ar = `بموجب الشروط التالية نسب المصنف — يجب عليك أن
تنسب العمل بالطريقة التي تحددها المؤلف أو المرخص (ولكن ليس بأي حال من
الأحوال أن توحي وتقترح بتحول أو استخدامك للعمل).
المشاركة على قدم المساواة — إذا كنت يعدل ، والتغيير ، أو الاستفادة
من هذا العمل ، قد ينتج عن توزيع العمل إلا في ظل تشابه او تطابق فى واحد
لهذا الترخيص.`

// Taken from http://creativecommons.org/licenses/by-sa/1.0/il/
const txt_il = `בכפוף לתנאים הבאים: ייחוס — עליך לייחס את היצירה (לתת קרדיט) באופן
המצויין על-ידי היוצר או מעניק הרישיון (אך לא בשום אופן המרמז על כך
שהם תומכים בך או בשימוש שלך ביצירה). שיתוף זהה — אם תחליט/י לשנות,
לעבד או ליצור יצירה נגזרת בהסתמך על יצירה זו, תוכל/י להפיץ את יצירתך
החדשה רק תחת אותו הרישיון או רישיון דומה לרישיון זה.`

const twoByteUtf8 = txt_ru + txt_gr + txt_ar + txt_il

// Taken from http://creativecommons.org/licenses/by-sa/2.0/kr/
const txt_kr = `다음과 같은 조건을 따라야 합니다: 저작자표시
(Attribution) — 저작자나 이용허락자가 정한 방법으로 저작물의
원저작자를 표시하여야 합니다(그러나 원저작자가 이용자나 이용자의
이용을 보증하거나 추천한다는 의미로 표시해서는 안됩니다). 
동일조건변경허락 — 이 저작물을 이용하여 만든 이차적 저작물에는 본
라이선스와 동일한 라이선스를 적용해야 합니다.`

// Taken from http://creativecommons.org/licenses/by-sa/3.0/th/
const txt_th = `ภายใต้เงื่อนไข ดังต่อไปนี้ : แสดงที่มา — คุณต้องแสดงที่
มาของงานดังกล่าว ตามรูปแบบที่ผู้สร้างสรรค์หรือผู้อนุญาตกำหนด (แต่
ไม่ใช่ในลักษณะที่ว่า พวกเขาสนับสนุนคุณหรือสนับสนุนการที่
คุณนำงานไปใช้) อนุญาตแบบเดียวกัน — หากคุณดัดแปลง เปลี่ยนรูป หรื
อต่อเติมงานนี้ คุณต้องใช้สัญญาอนุญาตแบบเดียวกันหรือแบบที่เหมื
อนกับสัญญาอนุญาตที่ใช้กับงานนี้เท่านั้น`

const threeByteUtf8 = txt_th

// Taken from http://creativecommons.org/licenses/by-sa/2.0/jp/
const txt_jp = `あなたの従うべき条件は以下の通りです。
表示 — あなたは原著作者のクレジットを表示しなければなりません。
継承 — もしあなたがこの作品を改変、変形または加工した場合、
あなたはその結果生じた作品をこの作品と同一の許諾条件の下でのみ
頒布することができます。`

// http://creativecommons.org/licenses/by-sa/2.5/cn/
const txt_cn = `您可以自由： 复制、发行、展览、表演、放映、
广播或通过信息网络传播本作品 创作演绎作品
对本作品进行商业性使用 惟须遵守下列条件：
署名 — 您必须按照作者或者许可人指定的方式对作品进行署名。
相同方式共享 — 如果您改变、转换本作品或者以本作品为基础进行创作，
您只能采用与本协议相同的许可协议发布基于本作品的演绎作品。`

const txt_cjk = txt_cn + txt_jp + txt_kr
const txt_all = txt_vn + twoByteUtf8 + threeByteUtf8 + txt_cjk

var txt_all_bytes = []byte(txt_all)
