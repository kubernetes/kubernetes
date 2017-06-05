// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package width

import (
	"bytes"
	"strings"
	"testing"

	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/transform"
)

func foldRune(r rune) (folded rune, ok bool) {
	alt, ok := mapRunes[r]
	if ok && alt.e&tagNeedsFold != 0 {
		return alt.r, true
	}
	return r, false
}

func widenRune(r rune) (wide rune, ok bool) {
	alt, ok := mapRunes[r]
	if k := alt.e.kind(); k == EastAsianHalfwidth || k == EastAsianNarrow {
		return alt.r, true
	}
	return r, false
}

func narrowRune(r rune) (narrow rune, ok bool) {
	alt, ok := mapRunes[r]
	if k := alt.e.kind(); k == EastAsianFullwidth || k == EastAsianWide || k == EastAsianAmbiguous {
		return alt.r, true
	}
	return r, false
}

func TestFoldSingleRunes(t *testing.T) {
	for r := rune(0); r < 0x1FFFF; r++ {
		if loSurrogate <= r && r <= hiSurrogate {
			continue
		}
		x, _ := foldRune(r)
		want := string(x)
		got := Fold.String(string(r))
		if got != want {
			t.Errorf("Fold().String(%U) = %+q; want %+q", r, got, want)
		}
	}
}

type transformTest struct {
	desc  string
	src   string
	nBuf  int
	nDst  int
	atEOF bool
	dst   string
	nSrc  int
	err   error
}

func (tc *transformTest) doTest(t *testing.T, tr Transformer) {
	b := make([]byte, tc.nBuf)
	nDst, nSrc, err := tr.Transform(b, []byte(tc.src), tc.atEOF)
	if got := string(b[:nDst]); got != tc.dst[:nDst] {
		t.Errorf("%s: dst was %+q; want %+q", tc.desc, got, tc.dst)
	}
	if nDst != tc.nDst {
		t.Errorf("%s: nDst was %d; want %d", tc.desc, nDst, tc.nDst)
	}
	if nSrc != tc.nSrc {
		t.Errorf("%s: nSrc was %d; want %d", tc.desc, nSrc, tc.nSrc)
	}
	if err != tc.err {
		t.Errorf("%s: error was %v; want %v", tc.desc, err, tc.err)
	}
	if got := tr.String(tc.src); got != tc.dst {
		t.Errorf("%s:String(%q) = %q; want %q", tc.desc, tc.src, got, tc.dst)
	}
}

func TestFold(t *testing.T) {
	for _, tc := range []transformTest{{
		desc:  "empty",
		src:   "",
		nBuf:  10,
		dst:   "",
		nDst:  0,
		nSrc:  0,
		atEOF: false,
		err:   nil,
	}, {
		desc:  "short source 1",
		src:   "a\xc2",
		nBuf:  10,
		dst:   "a\xc2",
		nDst:  1,
		nSrc:  1,
		atEOF: false,
		err:   transform.ErrShortSrc,
	}, {
		desc:  "short source 2",
		src:   "a\xe0\x80",
		nBuf:  10,
		dst:   "a\xe0\x80",
		nDst:  1,
		nSrc:  1,
		atEOF: false,
		err:   transform.ErrShortSrc,
	}, {
		desc:  "incomplete but terminated source 1",
		src:   "a\xc2",
		nBuf:  10,
		dst:   "a\xc2",
		nDst:  2,
		nSrc:  2,
		atEOF: true,
		err:   nil,
	}, {
		desc:  "incomplete but terminated source 2",
		src:   "a\xe0\x80",
		nBuf:  10,
		dst:   "a\xe0\x80",
		nDst:  3,
		nSrc:  3,
		atEOF: true,
		err:   nil,
	}, {
		desc:  "exact fit dst",
		src:   "a\uff01",
		nBuf:  2,
		dst:   "a!",
		nDst:  2,
		nSrc:  4,
		atEOF: false,
		err:   nil,
	}, {
		desc:  "exact fit dst and src ascii",
		src:   "ab",
		nBuf:  2,
		dst:   "ab",
		nDst:  2,
		nSrc:  2,
		atEOF: true,
		err:   nil,
	}, {
		desc:  "empty dst",
		src:   "\u0300",
		nBuf:  0,
		dst:   "\u0300",
		nDst:  0,
		nSrc:  0,
		atEOF: true,
		err:   transform.ErrShortDst,
	}, {
		desc:  "empty dst ascii",
		src:   "a",
		nBuf:  0,
		dst:   "a",
		nDst:  0,
		nSrc:  0,
		atEOF: true,
		err:   transform.ErrShortDst,
	}, {
		desc:  "short dst 1",
		src:   "a\uffe0", // ￠
		nBuf:  2,
		dst:   "a\u00a2", // ¢
		nDst:  1,
		nSrc:  1,
		atEOF: false,
		err:   transform.ErrShortDst,
	}, {
		desc:  "short dst 2",
		src:   "不夠",
		nBuf:  3,
		dst:   "不夠",
		nDst:  3,
		nSrc:  3,
		atEOF: true,
		err:   transform.ErrShortDst,
	}, {
		desc:  "short dst fast path",
		src:   "fast",
		nDst:  3,
		dst:   "fast",
		nBuf:  3,
		nSrc:  3,
		atEOF: true,
		err:   transform.ErrShortDst,
	}, {
		desc:  "short dst larger buffer",
		src:   "\uff21" + strings.Repeat("0", 127) + "B",
		nBuf:  128,
		dst:   "A" + strings.Repeat("0", 127) + "B",
		nDst:  128,
		nSrc:  130,
		atEOF: true,
		err:   transform.ErrShortDst,
	}, {
		desc:  "fast path alternation",
		src:   "fast路徑fast路徑",
		nBuf:  20,
		dst:   "fast路徑fast路徑",
		nDst:  20,
		nSrc:  20,
		atEOF: true,
		err:   nil,
	}} {
		tc.doTest(t, Fold)
	}
}

func TestWidenSingleRunes(t *testing.T) {
	for r := rune(0); r < 0x1FFFF; r++ {
		if loSurrogate <= r && r <= hiSurrogate {
			continue
		}
		alt, _ := widenRune(r)
		want := string(alt)
		got := Widen.String(string(r))
		if got != want {
			t.Errorf("Widen().String(%U) = %+q; want %+q", r, got, want)
		}
	}
}

func TestWiden(t *testing.T) {
	for _, tc := range []transformTest{{
		desc:  "empty",
		src:   "",
		nBuf:  10,
		dst:   "",
		nDst:  0,
		nSrc:  0,
		atEOF: false,
		err:   nil,
	}, {
		desc:  "short source 1",
		src:   "a\xc2",
		nBuf:  10,
		dst:   "ａ\xc2",
		nDst:  3,
		nSrc:  1,
		atEOF: false,
		err:   transform.ErrShortSrc,
	}, {
		desc:  "short source 2",
		src:   "a\xe0\x80",
		nBuf:  10,
		dst:   "ａ\xe0\x80",
		nDst:  3,
		nSrc:  1,
		atEOF: false,
		err:   transform.ErrShortSrc,
	}, {
		desc:  "incomplete but terminated source 1",
		src:   "a\xc2",
		nBuf:  10,
		dst:   "ａ\xc2",
		nDst:  4,
		nSrc:  2,
		atEOF: true,
		err:   nil,
	}, {
		desc:  "incomplete but terminated source 2",
		src:   "a\xe0\x80",
		nBuf:  10,
		dst:   "ａ\xe0\x80",
		nDst:  5,
		nSrc:  3,
		atEOF: true,
		err:   nil,
	}, {
		desc:  "exact fit dst",
		src:   "a!",
		nBuf:  6,
		dst:   "ａ\uff01",
		nDst:  6,
		nSrc:  2,
		atEOF: false,
		err:   nil,
	}, {
		desc:  "empty dst",
		src:   "\u0300",
		nBuf:  0,
		dst:   "\u0300",
		nDst:  0,
		nSrc:  0,
		atEOF: true,
		err:   transform.ErrShortDst,
	}, {
		desc:  "empty dst ascii",
		src:   "a",
		nBuf:  0,
		dst:   "ａ",
		nDst:  0,
		nSrc:  0,
		atEOF: true,
		err:   transform.ErrShortDst,
	}, {
		desc:  "short dst 1",
		src:   "a\uffe0",
		nBuf:  4,
		dst:   "ａ\uffe0",
		nDst:  3,
		nSrc:  1,
		atEOF: false,
		err:   transform.ErrShortDst,
	}, {
		desc:  "short dst 2",
		src:   "不夠",
		nBuf:  3,
		dst:   "不夠",
		nDst:  3,
		nSrc:  3,
		atEOF: true,
		err:   transform.ErrShortDst,
	}, {
		desc:  "short dst ascii",
		src:   "ascii",
		nBuf:  3,
		dst:   "ａｓｃｉｉ", // U+ff41, ...
		nDst:  3,
		nSrc:  1,
		atEOF: true,
		err:   transform.ErrShortDst,
	}, {
		desc:  "ambiguous",
		src:   "\uffe9",
		nBuf:  4,
		dst:   "\u2190",
		nDst:  3,
		nSrc:  3,
		atEOF: false,
		err:   nil,
	}} {
		tc.doTest(t, Widen)
	}
}

func TestNarrowSingleRunes(t *testing.T) {
	for r := rune(0); r < 0x1FFFF; r++ {
		if loSurrogate <= r && r <= hiSurrogate {
			continue
		}
		alt, _ := narrowRune(r)
		want := string(alt)
		got := Narrow.String(string(r))
		if got != want {
			t.Errorf("Narrow().String(%U) = %+q; want %+q", r, got, want)
		}
	}
}

func TestNarrow(t *testing.T) {
	for _, tc := range []transformTest{{
		desc:  "empty",
		src:   "",
		nBuf:  10,
		dst:   "",
		nDst:  0,
		nSrc:  0,
		atEOF: false,
		err:   nil,
	}, {
		desc:  "short source 1",
		src:   "a\xc2",
		nBuf:  10,
		dst:   "a\xc2",
		nDst:  1,
		nSrc:  1,
		atEOF: false,
		err:   transform.ErrShortSrc,
	}, {
		desc:  "short source 2",
		src:   "ａ\xe0\x80",
		nBuf:  10,
		dst:   "a\xe0\x80",
		nDst:  1,
		nSrc:  3,
		atEOF: false,
		err:   transform.ErrShortSrc,
	}, {
		desc:  "incomplete but terminated source 1",
		src:   "ａ\xc2",
		nBuf:  10,
		dst:   "a\xc2",
		nDst:  2,
		nSrc:  4,
		atEOF: true,
		err:   nil,
	}, {
		desc:  "incomplete but terminated source 2",
		src:   "ａ\xe0\x80",
		nBuf:  10,
		dst:   "a\xe0\x80",
		nDst:  3,
		nSrc:  5,
		atEOF: true,
		err:   nil,
	}, {
		desc:  "exact fit dst",
		src:   "ａ\uff01",
		nBuf:  2,
		dst:   "a!",
		nDst:  2,
		nSrc:  6,
		atEOF: false,
		err:   nil,
	}, {
		desc:  "exact fit dst",
		src:   "a\uff01",
		nBuf:  2,
		dst:   "a!",
		nDst:  2,
		nSrc:  4,
		atEOF: false,
		err:   nil,
	}, {
		desc:  "empty dst",
		src:   "\u0300",
		nBuf:  0,
		dst:   "\u0300",
		nDst:  0,
		nSrc:  0,
		atEOF: true,
		err:   transform.ErrShortDst,
	}, {
		desc:  "empty dst ascii",
		src:   "a",
		nBuf:  0,
		dst:   "a",
		nDst:  0,
		nSrc:  0,
		atEOF: true,
		err:   transform.ErrShortDst,
	}, {
		desc:  "short dst 1",
		src:   "ａ\uffe0", // ￠
		nBuf:  2,
		dst:   "a\u00a2", // ¢
		nDst:  1,
		nSrc:  3,
		atEOF: false,
		err:   transform.ErrShortDst,
	}, {
		desc:  "short dst 2",
		src:   "不夠",
		nBuf:  3,
		dst:   "不夠",
		nDst:  3,
		nSrc:  3,
		atEOF: true,
		err:   transform.ErrShortDst,
	}, {
		// Create a narrow variant of ambiguous runes, if they exist.
		desc:  "ambiguous",
		src:   "\u2190",
		nBuf:  4,
		dst:   "\uffe9",
		nDst:  3,
		nSrc:  3,
		atEOF: false,
		err:   nil,
	}, {
		desc:  "short dst fast path",
		src:   "fast",
		nBuf:  3,
		dst:   "fast",
		nDst:  3,
		nSrc:  3,
		atEOF: true,
		err:   transform.ErrShortDst,
	}, {
		desc:  "short dst larger buffer",
		src:   "\uff21" + strings.Repeat("0", 127) + "B",
		nBuf:  128,
		dst:   "A" + strings.Repeat("0", 127) + "B",
		nDst:  128,
		nSrc:  130,
		atEOF: true,
		err:   transform.ErrShortDst,
	}, {
		desc:  "fast path alternation",
		src:   "fast路徑fast路徑",
		nBuf:  20,
		dst:   "fast路徑fast路徑",
		nDst:  20,
		nSrc:  20,
		atEOF: true,
		err:   nil,
	}} {
		tc.doTest(t, Narrow)
	}
}

func bench(b *testing.B, t Transformer, s string) {
	dst := make([]byte, 1024)
	src := []byte(s)
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t.Transform(dst, src, true)
	}
}

func changingRunes(f func(r rune) (rune, bool)) string {
	buf := &bytes.Buffer{}
	for r := rune(0); r <= 0xFFFF; r++ {
		if _, ok := foldRune(r); ok {
			buf.WriteRune(r)
		}
	}
	return buf.String()
}

func BenchmarkFoldASCII(b *testing.B) {
	bench(b, Fold, testtext.ASCII)
}

func BenchmarkFoldCJK(b *testing.B) {
	bench(b, Fold, testtext.CJK)
}

func BenchmarkFoldNonCanonical(b *testing.B) {
	bench(b, Fold, changingRunes(foldRune))
}

func BenchmarkFoldOther(b *testing.B) {
	bench(b, Fold, testtext.TwoByteUTF8+testtext.ThreeByteUTF8)
}

func BenchmarkWideASCII(b *testing.B) {
	bench(b, Widen, testtext.ASCII)
}

func BenchmarkWideCJK(b *testing.B) {
	bench(b, Widen, testtext.CJK)
}

func BenchmarkWideNonCanonical(b *testing.B) {
	bench(b, Widen, changingRunes(widenRune))
}

func BenchmarkWideOther(b *testing.B) {
	bench(b, Widen, testtext.TwoByteUTF8+testtext.ThreeByteUTF8)
}

func BenchmarkNarrowASCII(b *testing.B) {
	bench(b, Narrow, testtext.ASCII)
}

func BenchmarkNarrowCJK(b *testing.B) {
	bench(b, Narrow, testtext.CJK)
}

func BenchmarkNarrowNonCanonical(b *testing.B) {
	bench(b, Narrow, changingRunes(narrowRune))
}

func BenchmarkNarrowOther(b *testing.B) {
	bench(b, Narrow, testtext.TwoByteUTF8+testtext.ThreeByteUTF8)
}
