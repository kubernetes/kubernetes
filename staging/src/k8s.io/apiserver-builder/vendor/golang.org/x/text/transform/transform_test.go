// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package transform

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"golang.org/x/text/internal/testtext"
)

type lowerCaseASCII struct{ NopResetter }

func (lowerCaseASCII) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	n := len(src)
	if n > len(dst) {
		n, err = len(dst), ErrShortDst
	}
	for i, c := range src[:n] {
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		dst[i] = c
	}
	return n, n, err
}

// lowerCaseASCIILookahead lowercases the string and reports ErrShortSrc as long
// as the input is not atEOF.
type lowerCaseASCIILookahead struct{ NopResetter }

func (lowerCaseASCIILookahead) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	n := len(src)
	if n > len(dst) {
		n, err = len(dst), ErrShortDst
	}
	for i, c := range src[:n] {
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		dst[i] = c
	}
	if !atEOF {
		err = ErrShortSrc
	}
	return n, n, err
}

var errYouMentionedX = errors.New("you mentioned X")

type dontMentionX struct{ NopResetter }

func (dontMentionX) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	n := len(src)
	if n > len(dst) {
		n, err = len(dst), ErrShortDst
	}
	for i, c := range src[:n] {
		if c == 'X' {
			return i, i, errYouMentionedX
		}
		dst[i] = c
	}
	return n, n, err
}

var errAtEnd = errors.New("error after all text")

type errorAtEnd struct{ NopResetter }

func (errorAtEnd) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	n := copy(dst, src)
	if n < len(src) {
		return n, n, ErrShortDst
	}
	if atEOF {
		return n, n, errAtEnd
	}
	return n, n, nil
}

type replaceWithConstant struct {
	replacement string
	written     int
}

func (t *replaceWithConstant) Reset() {
	t.written = 0
}

func (t *replaceWithConstant) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	if atEOF {
		nDst = copy(dst, t.replacement[t.written:])
		t.written += nDst
		if t.written < len(t.replacement) {
			err = ErrShortDst
		}
	}
	return nDst, len(src), err
}

type addAnXAtTheEnd struct{ NopResetter }

func (addAnXAtTheEnd) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	n := copy(dst, src)
	if n < len(src) {
		return n, n, ErrShortDst
	}
	if !atEOF {
		return n, n, nil
	}
	if len(dst) == n {
		return n, n, ErrShortDst
	}
	dst[n] = 'X'
	return n + 1, n, nil
}

// doublerAtEOF is a strange Transformer that transforms "this" to "tthhiiss",
// but only if atEOF is true.
type doublerAtEOF struct{ NopResetter }

func (doublerAtEOF) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	if !atEOF {
		return 0, 0, ErrShortSrc
	}
	for i, c := range src {
		if 2*i+2 >= len(dst) {
			return 2 * i, i, ErrShortDst
		}
		dst[2*i+0] = c
		dst[2*i+1] = c
	}
	return 2 * len(src), len(src), nil
}

// rleDecode and rleEncode implement a toy run-length encoding: "aabbbbbbbbbb"
// is encoded as "2a10b". The decoding is assumed to not contain any numbers.

type rleDecode struct{ NopResetter }

func (rleDecode) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
loop:
	for len(src) > 0 {
		n := 0
		for i, c := range src {
			if '0' <= c && c <= '9' {
				n = 10*n + int(c-'0')
				continue
			}
			if i == 0 {
				return nDst, nSrc, errors.New("rleDecode: bad input")
			}
			if n > len(dst) {
				return nDst, nSrc, ErrShortDst
			}
			for j := 0; j < n; j++ {
				dst[j] = c
			}
			dst, src = dst[n:], src[i+1:]
			nDst, nSrc = nDst+n, nSrc+i+1
			continue loop
		}
		if atEOF {
			return nDst, nSrc, errors.New("rleDecode: bad input")
		}
		return nDst, nSrc, ErrShortSrc
	}
	return nDst, nSrc, nil
}

type rleEncode struct {
	NopResetter

	// allowStutter means that "xxxxxxxx" can be encoded as "5x3x"
	// instead of always as "8x".
	allowStutter bool
}

func (e rleEncode) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	for len(src) > 0 {
		n, c0 := len(src), src[0]
		for i, c := range src[1:] {
			if c != c0 {
				n = i + 1
				break
			}
		}
		if n == len(src) && !atEOF && !e.allowStutter {
			return nDst, nSrc, ErrShortSrc
		}
		s := strconv.Itoa(n)
		if len(s) >= len(dst) {
			return nDst, nSrc, ErrShortDst
		}
		copy(dst, s)
		dst[len(s)] = c0
		dst, src = dst[len(s)+1:], src[n:]
		nDst, nSrc = nDst+len(s)+1, nSrc+n
	}
	return nDst, nSrc, nil
}

// trickler consumes all input bytes, but writes a single byte at a time to dst.
type trickler []byte

func (t *trickler) Reset() {
	*t = nil
}

func (t *trickler) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	*t = append(*t, src...)
	if len(*t) == 0 {
		return 0, 0, nil
	}
	if len(dst) == 0 {
		return 0, len(src), ErrShortDst
	}
	dst[0] = (*t)[0]
	*t = (*t)[1:]
	if len(*t) > 0 {
		err = ErrShortDst
	}
	return 1, len(src), err
}

// delayedTrickler is like trickler, but delays writing output to dst. This is
// highly unlikely to be relevant in practice, but it seems like a good idea
// to have some tolerance as long as progress can be detected.
type delayedTrickler []byte

func (t *delayedTrickler) Reset() {
	*t = nil
}

func (t *delayedTrickler) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	if len(*t) > 0 && len(dst) > 0 {
		dst[0] = (*t)[0]
		*t = (*t)[1:]
		nDst = 1
	}
	*t = append(*t, src...)
	if len(*t) > 0 {
		err = ErrShortDst
	}
	return nDst, len(src), err
}

type testCase struct {
	desc     string
	t        Transformer
	src      string
	dstSize  int
	srcSize  int
	ioSize   int
	wantStr  string
	wantErr  error
	wantIter int // number of iterations taken; 0 means we don't care.
}

func (t testCase) String() string {
	return tstr(t.t) + "; " + t.desc
}

func tstr(t Transformer) string {
	if stringer, ok := t.(fmt.Stringer); ok {
		return stringer.String()
	}
	s := fmt.Sprintf("%T", t)
	return s[1+strings.Index(s, "."):]
}

func (c chain) String() string {
	buf := &bytes.Buffer{}
	buf.WriteString("Chain(")
	for i, l := range c.link[:len(c.link)-1] {
		if i != 0 {
			fmt.Fprint(buf, ", ")
		}
		buf.WriteString(tstr(l.t))
	}
	buf.WriteString(")")
	return buf.String()
}

var testCases = []testCase{
	{
		desc:    "empty",
		t:       lowerCaseASCII{},
		src:     "",
		dstSize: 100,
		srcSize: 100,
		wantStr: "",
	},

	{
		desc:    "basic",
		t:       lowerCaseASCII{},
		src:     "Hello WORLD.",
		dstSize: 100,
		srcSize: 100,
		wantStr: "hello world.",
	},

	{
		desc:    "small dst",
		t:       lowerCaseASCII{},
		src:     "Hello WORLD.",
		dstSize: 3,
		srcSize: 100,
		wantStr: "hello world.",
	},

	{
		desc:    "small src",
		t:       lowerCaseASCII{},
		src:     "Hello WORLD.",
		dstSize: 100,
		srcSize: 4,
		wantStr: "hello world.",
	},

	{
		desc:    "small buffers",
		t:       lowerCaseASCII{},
		src:     "Hello WORLD.",
		dstSize: 3,
		srcSize: 4,
		wantStr: "hello world.",
	},

	{
		desc:    "very small buffers",
		t:       lowerCaseASCII{},
		src:     "Hello WORLD.",
		dstSize: 1,
		srcSize: 1,
		wantStr: "hello world.",
	},

	{
		desc:    "small dst with lookahead",
		t:       lowerCaseASCIILookahead{},
		src:     "Hello WORLD.",
		dstSize: 3,
		srcSize: 100,
		wantStr: "hello world.",
	},

	{
		desc:    "small src with lookahead",
		t:       lowerCaseASCIILookahead{},
		src:     "Hello WORLD.",
		dstSize: 100,
		srcSize: 4,
		wantStr: "hello world.",
	},

	{
		desc:    "small buffers with lookahead",
		t:       lowerCaseASCIILookahead{},
		src:     "Hello WORLD.",
		dstSize: 3,
		srcSize: 4,
		wantStr: "hello world.",
	},

	{
		desc:    "very small buffers with lookahead",
		t:       lowerCaseASCIILookahead{},
		src:     "Hello WORLD.",
		dstSize: 1,
		srcSize: 2,
		wantStr: "hello world.",
	},

	{
		desc:    "user error",
		t:       dontMentionX{},
		src:     "The First Rule of Transform Club: don't mention Mister X, ever.",
		dstSize: 100,
		srcSize: 100,
		wantStr: "The First Rule of Transform Club: don't mention Mister ",
		wantErr: errYouMentionedX,
	},

	{
		desc:    "user error at end",
		t:       errorAtEnd{},
		src:     "All goes well until it doesn't.",
		dstSize: 100,
		srcSize: 100,
		wantStr: "All goes well until it doesn't.",
		wantErr: errAtEnd,
	},

	{
		desc:    "user error at end, incremental",
		t:       errorAtEnd{},
		src:     "All goes well until it doesn't.",
		dstSize: 10,
		srcSize: 10,
		wantStr: "All goes well until it doesn't.",
		wantErr: errAtEnd,
	},

	{
		desc:    "replace entire non-empty string with one byte",
		t:       &replaceWithConstant{replacement: "X"},
		src:     "none of this will be copied",
		dstSize: 1,
		srcSize: 10,
		wantStr: "X",
	},

	{
		desc:    "replace entire empty string with one byte",
		t:       &replaceWithConstant{replacement: "X"},
		src:     "",
		dstSize: 1,
		srcSize: 10,
		wantStr: "X",
	},

	{
		desc:    "replace entire empty string with seven bytes",
		t:       &replaceWithConstant{replacement: "ABCDEFG"},
		src:     "",
		dstSize: 3,
		srcSize: 10,
		wantStr: "ABCDEFG",
	},

	{
		desc:    "add an X (initialBufSize-1)",
		t:       addAnXAtTheEnd{},
		src:     aaa[:initialBufSize-1],
		dstSize: 10,
		srcSize: 10,
		wantStr: aaa[:initialBufSize-1] + "X",
	},

	{
		desc:    "add an X (initialBufSize+0)",
		t:       addAnXAtTheEnd{},
		src:     aaa[:initialBufSize+0],
		dstSize: 10,
		srcSize: 10,
		wantStr: aaa[:initialBufSize+0] + "X",
	},

	{
		desc:    "add an X (initialBufSize+1)",
		t:       addAnXAtTheEnd{},
		src:     aaa[:initialBufSize+1],
		dstSize: 10,
		srcSize: 10,
		wantStr: aaa[:initialBufSize+1] + "X",
	},

	{
		desc:    "small buffers",
		t:       dontMentionX{},
		src:     "The First Rule of Transform Club: don't mention Mister X, ever.",
		dstSize: 10,
		srcSize: 10,
		wantStr: "The First Rule of Transform Club: don't mention Mister ",
		wantErr: errYouMentionedX,
	},

	{
		desc:    "very small buffers",
		t:       dontMentionX{},
		src:     "The First Rule of Transform Club: don't mention Mister X, ever.",
		dstSize: 1,
		srcSize: 1,
		wantStr: "The First Rule of Transform Club: don't mention Mister ",
		wantErr: errYouMentionedX,
	},

	{
		desc:    "only transform at EOF",
		t:       doublerAtEOF{},
		src:     "this",
		dstSize: 100,
		srcSize: 100,
		wantStr: "tthhiiss",
	},

	{
		desc:    "basic",
		t:       rleDecode{},
		src:     "1a2b3c10d11e0f1g",
		dstSize: 100,
		srcSize: 100,
		wantStr: "abbcccddddddddddeeeeeeeeeeeg",
	},

	{
		desc:    "long",
		t:       rleDecode{},
		src:     "12a23b34c45d56e99z",
		dstSize: 100,
		srcSize: 100,
		wantStr: strings.Repeat("a", 12) +
			strings.Repeat("b", 23) +
			strings.Repeat("c", 34) +
			strings.Repeat("d", 45) +
			strings.Repeat("e", 56) +
			strings.Repeat("z", 99),
	},

	{
		desc:    "tight buffers",
		t:       rleDecode{},
		src:     "1a2b3c10d11e0f1g",
		dstSize: 11,
		srcSize: 3,
		wantStr: "abbcccddddddddddeeeeeeeeeeeg",
	},

	{
		desc:    "short dst",
		t:       rleDecode{},
		src:     "1a2b3c10d11e0f1g",
		dstSize: 10,
		srcSize: 3,
		wantStr: "abbcccdddddddddd",
		wantErr: ErrShortDst,
	},

	{
		desc:    "short src",
		t:       rleDecode{},
		src:     "1a2b3c10d11e0f1g",
		dstSize: 11,
		srcSize: 2,
		ioSize:  2,
		wantStr: "abbccc",
		wantErr: ErrShortSrc,
	},

	{
		desc:    "basic",
		t:       rleEncode{},
		src:     "abbcccddddddddddeeeeeeeeeeeg",
		dstSize: 100,
		srcSize: 100,
		wantStr: "1a2b3c10d11e1g",
	},

	{
		desc: "long",
		t:    rleEncode{},
		src: strings.Repeat("a", 12) +
			strings.Repeat("b", 23) +
			strings.Repeat("c", 34) +
			strings.Repeat("d", 45) +
			strings.Repeat("e", 56) +
			strings.Repeat("z", 99),
		dstSize: 100,
		srcSize: 100,
		wantStr: "12a23b34c45d56e99z",
	},

	{
		desc:    "tight buffers",
		t:       rleEncode{},
		src:     "abbcccddddddddddeeeeeeeeeeeg",
		dstSize: 3,
		srcSize: 12,
		wantStr: "1a2b3c10d11e1g",
	},

	{
		desc:    "short dst",
		t:       rleEncode{},
		src:     "abbcccddddddddddeeeeeeeeeeeg",
		dstSize: 2,
		srcSize: 12,
		wantStr: "1a2b3c",
		wantErr: ErrShortDst,
	},

	{
		desc:    "short src",
		t:       rleEncode{},
		src:     "abbcccddddddddddeeeeeeeeeeeg",
		dstSize: 3,
		srcSize: 11,
		ioSize:  11,
		wantStr: "1a2b3c10d",
		wantErr: ErrShortSrc,
	},

	{
		desc:    "allowStutter = false",
		t:       rleEncode{allowStutter: false},
		src:     "aaaabbbbbbbbccccddddd",
		dstSize: 10,
		srcSize: 10,
		wantStr: "4a8b4c5d",
	},

	{
		desc:    "allowStutter = true",
		t:       rleEncode{allowStutter: true},
		src:     "aaaabbbbbbbbccccddddd",
		dstSize: 10,
		srcSize: 10,
		ioSize:  10,
		wantStr: "4a6b2b4c4d1d",
	},

	{
		desc:    "trickler",
		t:       &trickler{},
		src:     "abcdefghijklm",
		dstSize: 3,
		srcSize: 15,
		wantStr: "abcdefghijklm",
	},

	{
		desc:    "delayedTrickler",
		t:       &delayedTrickler{},
		src:     "abcdefghijklm",
		dstSize: 3,
		srcSize: 15,
		wantStr: "abcdefghijklm",
	},
}

func TestReader(t *testing.T) {
	for _, tc := range testCases {
		testtext.Run(t, tc.desc, func(t *testing.T) {
			r := NewReader(strings.NewReader(tc.src), tc.t)
			// Differently sized dst and src buffers are not part of the
			// exported API. We override them manually.
			r.dst = make([]byte, tc.dstSize)
			r.src = make([]byte, tc.srcSize)
			got, err := ioutil.ReadAll(r)
			str := string(got)
			if str != tc.wantStr || err != tc.wantErr {
				t.Errorf("\ngot  %q, %v\nwant %q, %v", str, err, tc.wantStr, tc.wantErr)
			}
		})
	}
}

func TestWriter(t *testing.T) {
	tests := append(testCases, chainTests()...)
	for _, tc := range tests {
		sizes := []int{1, 2, 3, 4, 5, 10, 100, 1000}
		if tc.ioSize > 0 {
			sizes = []int{tc.ioSize}
		}
		for _, sz := range sizes {
			testtext.Run(t, fmt.Sprintf("%s/%d", tc.desc, sz), func(t *testing.T) {
				bb := &bytes.Buffer{}
				w := NewWriter(bb, tc.t)
				// Differently sized dst and src buffers are not part of the
				// exported API. We override them manually.
				w.dst = make([]byte, tc.dstSize)
				w.src = make([]byte, tc.srcSize)
				src := make([]byte, sz)
				var err error
				for b := tc.src; len(b) > 0 && err == nil; {
					n := copy(src, b)
					b = b[n:]
					m := 0
					m, err = w.Write(src[:n])
					if m != n && err == nil {
						t.Errorf("did not consume all bytes %d < %d", m, n)
					}
				}
				if err == nil {
					err = w.Close()
				}
				str := bb.String()
				if str != tc.wantStr || err != tc.wantErr {
					t.Errorf("\ngot  %q, %v\nwant %q, %v", str, err, tc.wantStr, tc.wantErr)
				}
			})
		}
	}
}

func TestNop(t *testing.T) {
	testCases := []struct {
		str     string
		dstSize int
		err     error
	}{
		{"", 0, nil},
		{"", 10, nil},
		{"a", 0, ErrShortDst},
		{"a", 1, nil},
		{"a", 10, nil},
	}
	for i, tc := range testCases {
		dst := make([]byte, tc.dstSize)
		nDst, nSrc, err := Nop.Transform(dst, []byte(tc.str), true)
		want := tc.str
		if tc.dstSize < len(want) {
			want = want[:tc.dstSize]
		}
		if got := string(dst[:nDst]); got != want || err != tc.err || nSrc != nDst {
			t.Errorf("%d:\ngot %q, %d, %v\nwant %q, %d, %v", i, got, nSrc, err, want, nDst, tc.err)
		}
	}
}

func TestDiscard(t *testing.T) {
	testCases := []struct {
		str     string
		dstSize int
	}{
		{"", 0},
		{"", 10},
		{"a", 0},
		{"ab", 10},
	}
	for i, tc := range testCases {
		nDst, nSrc, err := Discard.Transform(make([]byte, tc.dstSize), []byte(tc.str), true)
		if nDst != 0 || nSrc != len(tc.str) || err != nil {
			t.Errorf("%d:\ngot %q, %d, %v\nwant 0, %d, nil", i, nDst, nSrc, err, len(tc.str))
		}
	}
}

// mkChain creates a Chain transformer. x must be alternating between transformer
// and bufSize, like T, (sz, T)*
func mkChain(x ...interface{}) *chain {
	t := []Transformer{}
	for i := 0; i < len(x); i += 2 {
		t = append(t, x[i].(Transformer))
	}
	c := Chain(t...).(*chain)
	for i, j := 1, 1; i < len(x); i, j = i+2, j+1 {
		c.link[j].b = make([]byte, x[i].(int))
	}
	return c
}

func chainTests() []testCase {
	return []testCase{
		{
			desc:     "nil error",
			t:        mkChain(rleEncode{}, 100, lowerCaseASCII{}),
			src:      "ABB",
			dstSize:  100,
			srcSize:  100,
			wantStr:  "1a2b",
			wantErr:  nil,
			wantIter: 1,
		},

		{
			desc:    "short dst buffer",
			t:       mkChain(lowerCaseASCII{}, 3, rleDecode{}),
			src:     "1a2b3c10d11e0f1g",
			dstSize: 10,
			srcSize: 3,
			wantStr: "abbcccdddddddddd",
			wantErr: ErrShortDst,
		},

		{
			desc:    "short internal dst buffer",
			t:       mkChain(lowerCaseASCII{}, 3, rleDecode{}, 10, Nop),
			src:     "1a2b3c10d11e0f1g",
			dstSize: 100,
			srcSize: 3,
			wantStr: "abbcccdddddddddd",
			wantErr: errShortInternal,
		},

		{
			desc:    "short internal dst buffer from input",
			t:       mkChain(rleDecode{}, 10, Nop),
			src:     "1a2b3c10d11e0f1g",
			dstSize: 100,
			srcSize: 3,
			wantStr: "abbcccdddddddddd",
			wantErr: errShortInternal,
		},

		{
			desc:    "empty short internal dst buffer",
			t:       mkChain(lowerCaseASCII{}, 3, rleDecode{}, 10, Nop),
			src:     "4a7b11e0f1g",
			dstSize: 100,
			srcSize: 3,
			wantStr: "aaaabbbbbbb",
			wantErr: errShortInternal,
		},

		{
			desc:    "empty short internal dst buffer from input",
			t:       mkChain(rleDecode{}, 10, Nop),
			src:     "4a7b11e0f1g",
			dstSize: 100,
			srcSize: 3,
			wantStr: "aaaabbbbbbb",
			wantErr: errShortInternal,
		},

		{
			desc:     "short internal src buffer after full dst buffer",
			t:        mkChain(Nop, 5, rleEncode{}, 10, Nop),
			src:      "cccccddddd",
			dstSize:  100,
			srcSize:  100,
			wantStr:  "",
			wantErr:  errShortInternal,
			wantIter: 1,
		},

		{
			desc:    "short internal src buffer after short dst buffer; test lastFull",
			t:       mkChain(rleDecode{}, 5, rleEncode{}, 4, Nop),
			src:     "2a1b4c6d",
			dstSize: 100,
			srcSize: 100,
			wantStr: "2a1b",
			wantErr: errShortInternal,
		},

		{
			desc:     "short internal src buffer after successful complete fill",
			t:        mkChain(Nop, 3, rleDecode{}),
			src:      "123a4b",
			dstSize:  4,
			srcSize:  3,
			wantStr:  "",
			wantErr:  errShortInternal,
			wantIter: 1,
		},

		{
			desc:    "short internal src buffer after short dst buffer; test lastFull",
			t:       mkChain(rleDecode{}, 5, rleEncode{}),
			src:     "2a1b4c6d",
			dstSize: 4,
			srcSize: 100,
			wantStr: "2a1b",
			wantErr: errShortInternal,
		},

		{
			desc:    "short src buffer",
			t:       mkChain(rleEncode{}, 5, Nop),
			src:     "abbcccddddeeeee",
			dstSize: 4,
			srcSize: 4,
			ioSize:  4,
			wantStr: "1a2b3c",
			wantErr: ErrShortSrc,
		},

		{
			desc:     "process all in one go",
			t:        mkChain(rleEncode{}, 5, Nop),
			src:      "abbcccddddeeeeeffffff",
			dstSize:  100,
			srcSize:  100,
			wantStr:  "1a2b3c4d5e6f",
			wantErr:  nil,
			wantIter: 1,
		},

		{
			desc:    "complete processing downstream after error",
			t:       mkChain(dontMentionX{}, 2, rleDecode{}, 5, Nop),
			src:     "3a4b5eX",
			dstSize: 100,
			srcSize: 100,
			ioSize:  100,
			wantStr: "aaabbbbeeeee",
			wantErr: errYouMentionedX,
		},

		{
			desc:    "return downstream fatal errors first (followed by short dst)",
			t:       mkChain(dontMentionX{}, 8, rleDecode{}, 4, Nop),
			src:     "3a4b5eX",
			dstSize: 100,
			srcSize: 100,
			ioSize:  100,
			wantStr: "aaabbbb",
			wantErr: errShortInternal,
		},

		{
			desc:    "return downstream fatal errors first (followed by short src)",
			t:       mkChain(dontMentionX{}, 5, Nop, 1, rleDecode{}),
			src:     "1a5bX",
			dstSize: 100,
			srcSize: 100,
			ioSize:  100,
			wantStr: "",
			wantErr: errShortInternal,
		},

		{
			desc:    "short internal",
			t:       mkChain(Nop, 11, rleEncode{}, 3, Nop),
			src:     "abbcccddddddddddeeeeeeeeeeeg",
			dstSize: 3,
			srcSize: 100,
			wantStr: "1a2b3c10d",
			wantErr: errShortInternal,
		},
	}
}

func doTransform(tc testCase) (res string, iter int, err error) {
	tc.t.Reset()
	dst := make([]byte, tc.dstSize)
	out, in := make([]byte, 0, 2*len(tc.src)), []byte(tc.src)
	for {
		iter++
		src, atEOF := in, true
		if len(src) > tc.srcSize {
			src, atEOF = src[:tc.srcSize], false
		}
		nDst, nSrc, err := tc.t.Transform(dst, src, atEOF)
		out = append(out, dst[:nDst]...)
		in = in[nSrc:]
		switch {
		case err == nil && len(in) != 0:
		case err == ErrShortSrc && nSrc > 0:
		case err == ErrShortDst && (nDst > 0 || nSrc > 0):
		default:
			return string(out), iter, err
		}
	}
}

func TestChain(t *testing.T) {
	if c, ok := Chain().(nop); !ok {
		t.Errorf("empty chain: %v; want Nop", c)
	}

	// Test Chain for a single Transformer.
	for _, tc := range testCases {
		tc.t = Chain(tc.t)
		str, _, err := doTransform(tc)
		if str != tc.wantStr || err != tc.wantErr {
			t.Errorf("%s:\ngot  %q, %v\nwant %q, %v", tc, str, err, tc.wantStr, tc.wantErr)
		}
	}

	tests := chainTests()
	sizes := []int{1, 2, 3, 4, 5, 7, 10, 100, 1000}
	addTest := func(tc testCase, t *chain) {
		if t.link[0].t != tc.t && tc.wantErr == ErrShortSrc {
			tc.wantErr = errShortInternal
		}
		if t.link[len(t.link)-2].t != tc.t && tc.wantErr == ErrShortDst {
			tc.wantErr = errShortInternal
		}
		tc.t = t
		tests = append(tests, tc)
	}
	for _, tc := range testCases {
		for _, sz := range sizes {
			tt := tc
			tt.dstSize = sz
			addTest(tt, mkChain(tc.t, tc.dstSize, Nop))
			addTest(tt, mkChain(tc.t, tc.dstSize, Nop, 2, Nop))
			addTest(tt, mkChain(Nop, tc.srcSize, tc.t, tc.dstSize, Nop))
			if sz >= tc.dstSize && (tc.wantErr != ErrShortDst || sz == tc.dstSize) {
				addTest(tt, mkChain(Nop, tc.srcSize, tc.t))
				addTest(tt, mkChain(Nop, 100, Nop, tc.srcSize, tc.t))
			}
		}
	}
	for _, tc := range testCases {
		tt := tc
		tt.dstSize = 1
		tt.wantStr = ""
		addTest(tt, mkChain(tc.t, tc.dstSize, Discard))
		addTest(tt, mkChain(Nop, tc.srcSize, tc.t, tc.dstSize, Discard))
		addTest(tt, mkChain(Nop, tc.srcSize, tc.t, tc.dstSize, Nop, tc.dstSize, Discard))
	}
	for _, tc := range testCases {
		tt := tc
		tt.dstSize = 100
		tt.wantStr = strings.Replace(tc.src, "0f", "", -1)
		// Chain encoders and decoders.
		if _, ok := tc.t.(rleEncode); ok && tc.wantErr == nil {
			addTest(tt, mkChain(tc.t, tc.dstSize, Nop, 1000, rleDecode{}))
			addTest(tt, mkChain(tc.t, tc.dstSize, Nop, tc.dstSize, rleDecode{}))
			addTest(tt, mkChain(Nop, tc.srcSize, tc.t, tc.dstSize, Nop, 100, rleDecode{}))
			// decoding needs larger destinations
			addTest(tt, mkChain(Nop, tc.srcSize, tc.t, tc.dstSize, rleDecode{}, 100, Nop))
			addTest(tt, mkChain(Nop, tc.srcSize, tc.t, tc.dstSize, Nop, 100, rleDecode{}, 100, Nop))
		} else if _, ok := tc.t.(rleDecode); ok && tc.wantErr == nil {
			// The internal buffer size may need to be the sum of the maximum segment
			// size of the two encoders!
			addTest(tt, mkChain(tc.t, 2*tc.dstSize, rleEncode{}))
			addTest(tt, mkChain(tc.t, tc.dstSize, Nop, 101, rleEncode{}))
			addTest(tt, mkChain(Nop, tc.srcSize, tc.t, tc.dstSize, Nop, 100, rleEncode{}))
			addTest(tt, mkChain(Nop, tc.srcSize, tc.t, tc.dstSize, Nop, 200, rleEncode{}, 100, Nop))
		}
	}
	for _, tc := range tests {
		str, iter, err := doTransform(tc)
		mi := tc.wantIter != 0 && tc.wantIter != iter
		if str != tc.wantStr || err != tc.wantErr || mi {
			t.Errorf("%s:\ngot  iter:%d, %q, %v\nwant iter:%d, %q, %v", tc, iter, str, err, tc.wantIter, tc.wantStr, tc.wantErr)
		}
		break
	}
}

func TestRemoveFunc(t *testing.T) {
	filter := RemoveFunc(func(r rune) bool {
		return strings.IndexRune("ab\u0300\u1234,", r) != -1
	})
	tests := []testCase{
		{
			src:     ",",
			wantStr: "",
		},

		{
			src:     "c",
			wantStr: "c",
		},

		{
			src:     "\u2345",
			wantStr: "\u2345",
		},

		{
			src:     "tschüß",
			wantStr: "tschüß",
		},

		{
			src:     ",до,свидания,",
			wantStr: "досвидания",
		},

		{
			src:     "a\xbd\xb2=\xbc ⌘",
			wantStr: "\uFFFD\uFFFD=\uFFFD ⌘",
		},

		{
			// If we didn't replace illegal bytes with RuneError, the result
			// would be \u0300 or the code would need to be more complex.
			src:     "\xcc\u0300\x80",
			wantStr: "\uFFFD\uFFFD",
		},

		{
			src:      "\xcc\u0300\x80",
			dstSize:  3,
			wantStr:  "\uFFFD\uFFFD",
			wantIter: 2,
		},

		{
			// Test a long buffer greater than the internal buffer size
			src:      "hello\xcc\xcc\xccworld",
			srcSize:  13,
			wantStr:  "hello\uFFFD\uFFFD\uFFFDworld",
			wantIter: 1,
		},

		{
			src:     "\u2345",
			dstSize: 2,
			wantStr: "",
			wantErr: ErrShortDst,
		},

		{
			src:     "\xcc",
			dstSize: 2,
			wantStr: "",
			wantErr: ErrShortDst,
		},

		{
			src:     "\u0300",
			dstSize: 2,
			srcSize: 1,
			wantStr: "",
			wantErr: ErrShortSrc,
		},

		{
			t: RemoveFunc(func(r rune) bool {
				return r == utf8.RuneError
			}),
			src:     "\xcc\u0300\x80",
			wantStr: "\u0300",
		},
	}

	for _, tc := range tests {
		tc.desc = tc.src
		if tc.t == nil {
			tc.t = filter
		}
		if tc.dstSize == 0 {
			tc.dstSize = 100
		}
		if tc.srcSize == 0 {
			tc.srcSize = 100
		}
		str, iter, err := doTransform(tc)
		mi := tc.wantIter != 0 && tc.wantIter != iter
		if str != tc.wantStr || err != tc.wantErr || mi {
			t.Errorf("%+q:\ngot  iter:%d, %+q, %v\nwant iter:%d, %+q, %v", tc.src, iter, str, err, tc.wantIter, tc.wantStr, tc.wantErr)
		}

		tc.src = str
		idem, _, _ := doTransform(tc)
		if str != idem {
			t.Errorf("%+q: found %+q; want %+q", tc.src, idem, str)
		}
	}
}

func testString(t *testing.T, f func(Transformer, string) (string, int, error)) {
	for _, tt := range append(testCases, chainTests()...) {
		if tt.desc == "allowStutter = true" {
			// We don't have control over the buffer size, so we eliminate tests
			// that depend on a specific buffer size being set.
			continue
		}
		if tt.wantErr == ErrShortDst || tt.wantErr == ErrShortSrc {
			// The result string will be different.
			continue
		}
		testtext.Run(t, tt.desc, func(t *testing.T) {
			got, n, err := f(tt.t, tt.src)
			if tt.wantErr != err {
				t.Errorf("error: got %v; want %v", err, tt.wantErr)
			}
			// Check that err == nil implies that n == len(tt.src). Note that vice
			// versa isn't necessarily true.
			if err == nil && n != len(tt.src) {
				t.Errorf("err == nil: got %d bytes, want %d", n, err)
			}
			if got != tt.wantStr {
				t.Errorf("string: got %q; want %q", got, tt.wantStr)
			}
		})
	}
}

func TestBytes(t *testing.T) {
	testString(t, func(z Transformer, s string) (string, int, error) {
		b, n, err := Bytes(z, []byte(s))
		return string(b), n, err
	})
}

func TestAppend(t *testing.T) {
	// Create a bunch of subtests for different buffer sizes.
	testCases := [][]byte{
		nil,
		make([]byte, 0, 0),
		make([]byte, 0, 1),
		make([]byte, 1, 1),
		make([]byte, 1, 5),
		make([]byte, 100, 100),
		make([]byte, 100, 200),
	}
	for _, tc := range testCases {
		testString(t, func(z Transformer, s string) (string, int, error) {
			b, n, err := Append(z, tc, []byte(s))
			return string(b[len(tc):]), n, err
		})
	}
}

func TestString(t *testing.T) {
	testtext.Run(t, "transform", func(t *testing.T) { testString(t, String) })

	// Overrun the internal destination buffer.
	for i, s := range []string{
		aaa[:1*initialBufSize-1],
		aaa[:1*initialBufSize+0],
		aaa[:1*initialBufSize+1],
		AAA[:1*initialBufSize-1],
		AAA[:1*initialBufSize+0],
		AAA[:1*initialBufSize+1],
		AAA[:2*initialBufSize-1],
		AAA[:2*initialBufSize+0],
		AAA[:2*initialBufSize+1],
		aaa[:1*initialBufSize-2] + "A",
		aaa[:1*initialBufSize-1] + "A",
		aaa[:1*initialBufSize+0] + "A",
		aaa[:1*initialBufSize+1] + "A",
	} {
		testtext.Run(t, fmt.Sprint("dst buffer test using lower/", i), func(t *testing.T) {
			got, _, _ := String(lowerCaseASCII{}, s)
			if want := strings.ToLower(s); got != want {
				t.Errorf("got %s (%d); want %s (%d)", got, len(got), want, len(want))
			}
		})
	}

	// Overrun the internal source buffer.
	for i, s := range []string{
		aaa[:1*initialBufSize-1],
		aaa[:1*initialBufSize+0],
		aaa[:1*initialBufSize+1],
		aaa[:2*initialBufSize+1],
		aaa[:2*initialBufSize+0],
		aaa[:2*initialBufSize+1],
	} {
		testtext.Run(t, fmt.Sprint("src buffer test using rleEncode/", i), func(t *testing.T) {
			got, _, _ := String(rleEncode{}, s)
			if want := fmt.Sprintf("%da", len(s)); got != want {
				t.Errorf("got %s (%d); want %s (%d)", got, len(got), want, len(want))
			}
		})
	}

	// Test allocations for non-changing strings.
	// Note we still need to allocate a single buffer.
	for i, s := range []string{
		"",
		"123456789",
		aaa[:initialBufSize-1],
		aaa[:initialBufSize+0],
		aaa[:initialBufSize+1],
		aaa[:10*initialBufSize],
	} {
		testtext.Run(t, fmt.Sprint("alloc/", i), func(t *testing.T) {
			if n := testtext.AllocsPerRun(5, func() { String(&lowerCaseASCIILookahead{}, s) }); n > 1 {
				t.Errorf("#allocs was %f; want 1", n)
			}
		})
	}
}

// TestBytesAllocation tests that buffer growth stays limited with the trickler
// transformer, which behaves oddly but within spec. In case buffer growth is
// not correctly handled, the test will either panic with a failed allocation or
// thrash. To ensure the tests terminate under the last condition, we time out
// after some sufficiently long period of time.
func TestBytesAllocation(t *testing.T) {
	done := make(chan bool)
	go func() {
		in := bytes.Repeat([]byte{'a'}, 1000)
		tr := trickler(make([]byte, 1))
		Bytes(&tr, in)
		done <- true
	}()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Error("time out, likely due to excessive allocation")
	}
}

// TestStringAllocation tests that buffer growth stays limited with the trickler
// transformer, which behaves oddly but within spec. In case buffer growth is
// not correctly handled, the test will either panic with a failed allocation or
// thrash. To ensure the tests terminate under the last condition, we time out
// after some sufficiently long period of time.
func TestStringAllocation(t *testing.T) {
	done := make(chan bool)
	go func() {
		tr := trickler(make([]byte, 1))
		String(&tr, aaa[:1000])
		done <- true
	}()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Error("time out, likely due to excessive allocation")
	}
}

func BenchmarkStringLowerEmpty(b *testing.B) {
	for i := 0; i < b.N; i++ {
		String(&lowerCaseASCIILookahead{}, "")
	}
}

func BenchmarkStringLowerIdentical(b *testing.B) {
	for i := 0; i < b.N; i++ {
		String(&lowerCaseASCIILookahead{}, aaa[:4096])
	}
}

func BenchmarkStringLowerChanged(b *testing.B) {
	for i := 0; i < b.N; i++ {
		String(&lowerCaseASCIILookahead{}, AAA[:4096])
	}
}

var (
	aaa = strings.Repeat("a", 4096)
	AAA = strings.Repeat("A", 4096)
)
