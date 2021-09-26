// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runes

import (
	"strings"
	"testing"
	"unicode/utf8"

	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/transform"
)

type transformTest struct {
	desc    string
	szDst   int
	atEOF   bool
	repl    string
	in      string
	out     string // result string of first call to Transform
	outFull string // transform of entire input string
	err     error
	errSpan error
	nSpan   int

	t transform.SpanningTransformer
}

const large = 10240

func (tt *transformTest) check(t *testing.T, i int) {
	if tt.t == nil {
		return
	}
	dst := make([]byte, tt.szDst)
	src := []byte(tt.in)
	nDst, nSrc, err := tt.t.Transform(dst, src, tt.atEOF)
	if err != tt.err {
		t.Errorf("%d:%s:error: got %v; want %v", i, tt.desc, err, tt.err)
	}
	if got := string(dst[:nDst]); got != tt.out {
		t.Errorf("%d:%s:out: got %q; want %q", i, tt.desc, got, tt.out)
	}

	// Calls tt.t.Transform for the remainder of the input. We use this to test
	// the nSrc return value.
	out := make([]byte, large)
	n := copy(out, dst[:nDst])
	nDst, _, _ = tt.t.Transform(out[n:], src[nSrc:], true)
	if got, want := string(out[:n+nDst]), tt.outFull; got != want {
		t.Errorf("%d:%s:outFull: got %q; want %q", i, tt.desc, got, want)
	}

	tt.t.Reset()
	p := 0
	for ; p < len(tt.in) && p < len(tt.outFull) && tt.in[p] == tt.outFull[p]; p++ {
	}
	if tt.nSpan != 0 {
		p = tt.nSpan
	}
	if n, err = tt.t.Span([]byte(tt.in), tt.atEOF); n != p || err != tt.errSpan {
		t.Errorf("%d:%s:span: got %d, %v; want %d, %v", i, tt.desc, n, err, p, tt.errSpan)
	}
}

func idem(r rune) rune { return r }

func TestMap(t *testing.T) {
	runes := []rune{'a', 'ç', '中', '\U00012345', 'a'}
	// Default mapper used for this test.
	rotate := Map(func(r rune) rune {
		for i, m := range runes {
			if m == r {
				return runes[i+1]
			}
		}
		return r
	})

	for i, tt := range []transformTest{{
		desc:    "empty",
		szDst:   large,
		atEOF:   true,
		in:      "",
		out:     "",
		outFull: "",
		t:       rotate,
	}, {
		desc:    "no change",
		szDst:   1,
		atEOF:   true,
		in:      "b",
		out:     "b",
		outFull: "b",
		t:       rotate,
	}, {
		desc:    "short dst",
		szDst:   2,
		atEOF:   true,
		in:      "aaaa",
		out:     "ç",
		outFull: "çççç",
		err:     transform.ErrShortDst,
		errSpan: transform.ErrEndOfSpan,
		t:       rotate,
	}, {
		desc:    "short dst ascii, no change",
		szDst:   2,
		atEOF:   true,
		in:      "bbb",
		out:     "bb",
		outFull: "bbb",
		err:     transform.ErrShortDst,
		t:       rotate,
	}, {
		desc:    "short dst writing error",
		szDst:   2,
		atEOF:   false,
		in:      "a\x80",
		out:     "ç",
		outFull: "ç\ufffd",
		err:     transform.ErrShortDst,
		errSpan: transform.ErrEndOfSpan,
		t:       rotate,
	}, {
		desc:    "short dst writing incomplete rune",
		szDst:   2,
		atEOF:   true,
		in:      "a\xc0",
		out:     "ç",
		outFull: "ç\ufffd",
		err:     transform.ErrShortDst,
		errSpan: transform.ErrEndOfSpan,
		t:       rotate,
	}, {
		desc:    "short dst, longer",
		szDst:   5,
		atEOF:   true,
		in:      "Hellø",
		out:     "Hell",
		outFull: "Hellø",
		err:     transform.ErrShortDst,
		t:       rotate,
	}, {
		desc:    "short dst, single",
		szDst:   1,
		atEOF:   false,
		in:      "ø",
		out:     "",
		outFull: "ø",
		err:     transform.ErrShortDst,
		t:       Map(idem),
	}, {
		desc:    "short dst, longer, writing error",
		szDst:   8,
		atEOF:   false,
		in:      "\x80Hello\x80",
		out:     "\ufffdHello",
		outFull: "\ufffdHello\ufffd",
		err:     transform.ErrShortDst,
		errSpan: transform.ErrEndOfSpan,
		t:       rotate,
	}, {
		desc:    "short src",
		szDst:   2,
		atEOF:   false,
		in:      "a\xc2",
		out:     "ç",
		outFull: "ç\ufffd",
		err:     transform.ErrShortSrc,
		errSpan: transform.ErrEndOfSpan,
		t:       rotate,
	}, {
		desc:    "invalid input, atEOF",
		szDst:   large,
		atEOF:   true,
		in:      "\x80",
		out:     "\ufffd",
		outFull: "\ufffd",
		errSpan: transform.ErrEndOfSpan,
		t:       rotate,
	}, {
		desc:    "invalid input, !atEOF",
		szDst:   large,
		atEOF:   false,
		in:      "\x80",
		out:     "\ufffd",
		outFull: "\ufffd",
		errSpan: transform.ErrEndOfSpan,
		t:       rotate,
	}, {
		desc:    "incomplete rune !atEOF",
		szDst:   large,
		atEOF:   false,
		in:      "\xc2",
		out:     "",
		outFull: "\ufffd",
		err:     transform.ErrShortSrc,
		errSpan: transform.ErrShortSrc,
		t:       rotate,
	}, {
		desc:    "invalid input, incomplete rune atEOF",
		szDst:   large,
		atEOF:   true,
		in:      "\xc2",
		out:     "\ufffd",
		outFull: "\ufffd",
		errSpan: transform.ErrEndOfSpan,
		t:       rotate,
	}, {
		desc:    "misc correct",
		szDst:   large,
		atEOF:   true,
		in:      "a\U00012345 ç!",
		out:     "ça 中!",
		outFull: "ça 中!",
		errSpan: transform.ErrEndOfSpan,
		t:       rotate,
	}, {
		desc:    "misc correct and invalid",
		szDst:   large,
		atEOF:   true,
		in:      "Hello\x80 w\x80orl\xc0d!\xc0",
		out:     "Hello\ufffd w\ufffdorl\ufffdd!\ufffd",
		outFull: "Hello\ufffd w\ufffdorl\ufffdd!\ufffd",
		errSpan: transform.ErrEndOfSpan,
		t:       rotate,
	}, {
		desc:    "misc correct and invalid, short src",
		szDst:   large,
		atEOF:   false,
		in:      "Hello\x80 w\x80orl\xc0d!\xc2",
		out:     "Hello\ufffd w\ufffdorl\ufffdd!",
		outFull: "Hello\ufffd w\ufffdorl\ufffdd!\ufffd",
		err:     transform.ErrShortSrc,
		errSpan: transform.ErrEndOfSpan,
		t:       rotate,
	}, {
		desc:    "misc correct and invalid, short src, replacing RuneError",
		szDst:   large,
		atEOF:   false,
		in:      "Hel\ufffdlo\x80 w\x80orl\xc0d!\xc2",
		out:     "Hel?lo? w?orl?d!",
		outFull: "Hel?lo? w?orl?d!?",
		errSpan: transform.ErrEndOfSpan,
		err:     transform.ErrShortSrc,
		t: Map(func(r rune) rune {
			if r == utf8.RuneError {
				return '?'
			}
			return r
		}),
	}} {
		tt.check(t, i)
	}
}

func TestRemove(t *testing.T) {
	remove := Remove(Predicate(func(r rune) bool {
		return strings.ContainsRune("aeiou\u0300\uFF24\U00012345", r)
	}))

	for i, tt := range []transformTest{
		0: {
			szDst:   large,
			atEOF:   true,
			in:      "",
			out:     "",
			outFull: "",
			t:       remove,
		},
		1: {
			szDst:   0,
			atEOF:   true,
			in:      "aaaa",
			out:     "",
			outFull: "",
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		2: {
			szDst:   1,
			atEOF:   true,
			in:      "aaaa",
			out:     "",
			outFull: "",
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		3: {
			szDst:   1,
			atEOF:   true,
			in:      "baaaa",
			out:     "b",
			outFull: "b",
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		4: {
			szDst:   2,
			atEOF:   true,
			in:      "açaaa",
			out:     "ç",
			outFull: "ç",
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		5: {
			szDst:   2,
			atEOF:   true,
			in:      "aaaç",
			out:     "ç",
			outFull: "ç",
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		6: {
			szDst:   2,
			atEOF:   false,
			in:      "a\x80",
			out:     "",
			outFull: "\ufffd",
			err:     transform.ErrShortDst,
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		7: {
			szDst:   1,
			atEOF:   true,
			in:      "a\xc0",
			out:     "",
			outFull: "\ufffd",
			err:     transform.ErrShortDst,
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		8: {
			szDst:   1,
			atEOF:   false,
			in:      "a\xc2",
			out:     "",
			outFull: "\ufffd",
			err:     transform.ErrShortSrc,
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		9: {
			szDst:   large,
			atEOF:   true,
			in:      "\x80",
			out:     "\ufffd",
			outFull: "\ufffd",
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		10: {
			szDst:   large,
			atEOF:   false,
			in:      "\x80",
			out:     "\ufffd",
			outFull: "\ufffd",
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		11: {
			szDst:   large,
			atEOF:   true,
			in:      "\xc2",
			out:     "\ufffd",
			outFull: "\ufffd",
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		12: {
			szDst:   large,
			atEOF:   false,
			in:      "\xc2",
			out:     "",
			outFull: "\ufffd",
			err:     transform.ErrShortSrc,
			errSpan: transform.ErrShortSrc,
			t:       remove,
		},
		13: {
			szDst:   large,
			atEOF:   true,
			in:      "Hello \U00012345world!",
			out:     "Hll wrld!",
			outFull: "Hll wrld!",
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		14: {
			szDst:   large,
			atEOF:   true,
			in:      "Hello\x80 w\x80orl\xc0d!\xc0",
			out:     "Hll\ufffd w\ufffdrl\ufffdd!\ufffd",
			outFull: "Hll\ufffd w\ufffdrl\ufffdd!\ufffd",
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		15: {
			szDst:   large,
			atEOF:   false,
			in:      "Hello\x80 w\x80orl\xc0d!\xc2",
			out:     "Hll\ufffd w\ufffdrl\ufffdd!",
			outFull: "Hll\ufffd w\ufffdrl\ufffdd!\ufffd",
			err:     transform.ErrShortSrc,
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		16: {
			szDst:   large,
			atEOF:   false,
			in:      "Hel\ufffdlo\x80 w\x80orl\xc0d!\xc2",
			out:     "Hello world!",
			outFull: "Hello world!",
			err:     transform.ErrShortSrc,
			errSpan: transform.ErrEndOfSpan,
			t:       Remove(Predicate(func(r rune) bool { return r == utf8.RuneError })),
		},
		17: {
			szDst:   4,
			atEOF:   true,
			in:      "Hellø",
			out:     "Hll",
			outFull: "Hllø",
			err:     transform.ErrShortDst,
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		18: {
			szDst:   4,
			atEOF:   false,
			in:      "Hellø",
			out:     "Hll",
			outFull: "Hllø",
			err:     transform.ErrShortDst,
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		19: {
			szDst:   8,
			atEOF:   false,
			in:      "\x80Hello\uFF24\x80",
			out:     "\ufffdHll",
			outFull: "\ufffdHll\ufffd",
			err:     transform.ErrShortDst,
			errSpan: transform.ErrEndOfSpan,
			t:       remove,
		},
		20: {
			szDst:   8,
			atEOF:   false,
			in:      "Hllll",
			out:     "Hllll",
			outFull: "Hllll",
			t:       remove,
		}} {
		tt.check(t, i)
	}
}

func TestReplaceIllFormed(t *testing.T) {
	replace := ReplaceIllFormed()

	for i, tt := range []transformTest{
		0: {
			szDst:   large,
			atEOF:   true,
			in:      "",
			out:     "",
			outFull: "",
			t:       replace,
		},
		1: {
			szDst:   1,
			atEOF:   true,
			in:      "aa",
			out:     "a",
			outFull: "aa",
			err:     transform.ErrShortDst,
			t:       replace,
		},
		2: {
			szDst:   1,
			atEOF:   true,
			in:      "a\x80",
			out:     "a",
			outFull: "a\ufffd",
			err:     transform.ErrShortDst,
			errSpan: transform.ErrEndOfSpan,
			t:       replace,
		},
		3: {
			szDst:   1,
			atEOF:   true,
			in:      "a\xc2",
			out:     "a",
			outFull: "a\ufffd",
			err:     transform.ErrShortDst,
			errSpan: transform.ErrEndOfSpan,
			t:       replace,
		},
		4: {
			szDst:   large,
			atEOF:   true,
			in:      "\x80",
			out:     "\ufffd",
			outFull: "\ufffd",
			errSpan: transform.ErrEndOfSpan,
			t:       replace,
		},
		5: {
			szDst:   large,
			atEOF:   false,
			in:      "\x80",
			out:     "\ufffd",
			outFull: "\ufffd",
			errSpan: transform.ErrEndOfSpan,
			t:       replace,
		},
		6: {
			szDst:   large,
			atEOF:   true,
			in:      "\xc2",
			out:     "\ufffd",
			outFull: "\ufffd",
			errSpan: transform.ErrEndOfSpan,
			t:       replace,
		},
		7: {
			szDst:   large,
			atEOF:   false,
			in:      "\xc2",
			out:     "",
			outFull: "\ufffd",
			err:     transform.ErrShortSrc,
			errSpan: transform.ErrShortSrc,
			t:       replace,
		},
		8: {
			szDst:   large,
			atEOF:   true,
			in:      "Hello world!",
			out:     "Hello world!",
			outFull: "Hello world!",
			t:       replace,
		},
		9: {
			szDst:   large,
			atEOF:   true,
			in:      "Hello\x80 w\x80orl\xc2d!\xc2",
			out:     "Hello\ufffd w\ufffdorl\ufffdd!\ufffd",
			outFull: "Hello\ufffd w\ufffdorl\ufffdd!\ufffd",
			errSpan: transform.ErrEndOfSpan,
			t:       replace,
		},
		10: {
			szDst:   large,
			atEOF:   false,
			in:      "Hello\x80 w\x80orl\xc2d!\xc2",
			out:     "Hello\ufffd w\ufffdorl\ufffdd!",
			outFull: "Hello\ufffd w\ufffdorl\ufffdd!\ufffd",
			err:     transform.ErrShortSrc,
			errSpan: transform.ErrEndOfSpan,
			t:       replace,
		},
		16: {
			szDst:   10,
			atEOF:   false,
			in:      "\x80Hello\x80",
			out:     "\ufffdHello",
			outFull: "\ufffdHello\ufffd",
			err:     transform.ErrShortDst,
			errSpan: transform.ErrEndOfSpan,
			t:       replace,
		},
		17: {
			szDst:   10,
			atEOF:   false,
			in:      "\ufffdHello\ufffd",
			out:     "\ufffdHello",
			outFull: "\ufffdHello\ufffd",
			err:     transform.ErrShortDst,
			t:       replace,
		},
	} {
		tt.check(t, i)
	}
}

func TestMapAlloc(t *testing.T) {
	if n := testtext.AllocsPerRun(3, func() {
		Map(idem).Transform(nil, nil, false)
	}); n > 0 {
		t.Errorf("got %f; want 0", n)
	}
}

func rmNop(r rune) bool { return false }

func TestRemoveAlloc(t *testing.T) {
	if n := testtext.AllocsPerRun(3, func() {
		Remove(Predicate(rmNop)).Transform(nil, nil, false)
	}); n > 0 {
		t.Errorf("got %f; want 0", n)
	}
}

func TestReplaceIllFormedAlloc(t *testing.T) {
	if n := testtext.AllocsPerRun(3, func() {
		ReplaceIllFormed().Transform(nil, nil, false)
	}); n > 0 {
		t.Errorf("got %f; want 0", n)
	}
}

func doBench(b *testing.B, t Transformer) {
	for _, bc := range []struct{ name, data string }{
		{"ascii", testtext.ASCII},
		{"3byte", testtext.ThreeByteUTF8},
	} {
		dst := make([]byte, 2*len(bc.data))
		src := []byte(bc.data)

		testtext.Bench(b, bc.name+"/transform", func(b *testing.B) {
			b.SetBytes(int64(len(src)))
			for i := 0; i < b.N; i++ {
				t.Transform(dst, src, true)
			}
		})
		src = t.Bytes(src)
		t.Reset()
		testtext.Bench(b, bc.name+"/span", func(b *testing.B) {
			b.SetBytes(int64(len(src)))
			for i := 0; i < b.N; i++ {
				t.Span(src, true)
			}
		})
	}
}

func BenchmarkRemove(b *testing.B) {
	doBench(b, Remove(Predicate(func(r rune) bool { return r == 'e' })))
}

func BenchmarkMapAll(b *testing.B) {
	doBench(b, Map(func(r rune) rune { return 'a' }))
}

func BenchmarkMapNone(b *testing.B) {
	doBench(b, Map(func(r rune) rune { return r }))
}

func BenchmarkReplaceIllFormed(b *testing.B) {
	doBench(b, ReplaceIllFormed())
}

var (
	input = strings.Repeat("Thé qüick brøwn føx jumps øver the lazy døg. ", 100)
)
