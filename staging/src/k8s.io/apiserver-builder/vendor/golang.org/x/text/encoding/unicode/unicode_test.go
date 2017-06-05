// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unicode

import (
	"testing"

	"golang.org/x/text/transform"
)

func TestUTF8Decoder(t *testing.T) {
	testCases := []struct {
		desc    string
		src     string
		notEOF  bool // the inverse of atEOF
		sizeDst int
		want    string
		nSrc    int
		err     error
	}{{
		desc: "empty string, empty dest buffer",
	}, {
		desc:    "empty string",
		sizeDst: 8,
	}, {
		desc:    "empty string, streaming",
		notEOF:  true,
		sizeDst: 8,
	}, {
		desc:    "ascii",
		src:     "abcde",
		sizeDst: 8,
		want:    "abcde",
		nSrc:    5,
	}, {
		desc:    "ascii and error",
		src:     "ab\x80de",
		sizeDst: 7,
		want:    "ab\ufffdde",
		nSrc:    5,
	}, {
		desc:    "valid two-byte sequence",
		src:     "a\u0300bc",
		sizeDst: 7,
		want:    "a\u0300bc",
		nSrc:    5,
	}, {
		desc:    "valid three-byte sequence",
		src:     "a\u0300中",
		sizeDst: 7,
		want:    "a\u0300中",
		nSrc:    6,
	}, {
		desc:    "valid four-byte sequence",
		src:     "a中\U00016F50",
		sizeDst: 8,
		want:    "a中\U00016F50",
		nSrc:    8,
	}, {
		desc:    "short source buffer",
		src:     "abc\xf0\x90",
		notEOF:  true,
		sizeDst: 10,
		want:    "abc",
		nSrc:    3,
		err:     transform.ErrShortSrc,
	}, {
		// We don't check for the maximal subpart of an ill-formed subsequence
		// at the end of an open segment.
		desc:    "complete invalid that looks like short at end",
		src:     "abc\xf0\x80",
		notEOF:  true,
		sizeDst: 10,
		want:    "abc", // instead of "abc\ufffd\ufffd",
		nSrc:    3,
		err:     transform.ErrShortSrc,
	}, {
		desc:    "incomplete sequence at end",
		src:     "a\x80bc\xf0\x90",
		sizeDst: 9,
		want:    "a\ufffdbc\ufffd",
		nSrc:    6,
	}, {
		desc:    "invalid second byte",
		src:     "abc\xf0dddd",
		sizeDst: 10,
		want:    "abc\ufffddddd",
		nSrc:    8,
	}, {
		desc:    "invalid second byte at end",
		src:     "abc\xf0d",
		sizeDst: 10,
		want:    "abc\ufffdd",
		nSrc:    5,
	}, {
		desc:    "invalid third byte",
		src:     "a\u0300bc\xf0\x90dddd",
		sizeDst: 12,
		want:    "a\u0300bc\ufffddddd",
		nSrc:    11,
	}, {
		desc:    "invalid third byte at end",
		src:     "a\u0300bc\xf0\x90d",
		sizeDst: 12,
		want:    "a\u0300bc\ufffdd",
		nSrc:    8,
	}, {
		desc:    "invalid fourth byte, tight buffer",
		src:     "a\u0300bc\xf0\x90\x80d",
		sizeDst: 9,
		want:    "a\u0300bc\ufffdd",
		nSrc:    9,
	}, {
		desc:    "invalid fourth byte at end",
		src:     "a\u0300bc\xf0\x90\x80",
		sizeDst: 8,
		want:    "a\u0300bc\ufffd",
		nSrc:    8,
	}, {
		desc:    "invalid fourth byte and short four byte sequence",
		src:     "a\u0300bc\xf0\x90\x80\xf0\x90\x80",
		notEOF:  true,
		sizeDst: 20,
		want:    "a\u0300bc\ufffd",
		nSrc:    8,
		err:     transform.ErrShortSrc,
	}, {
		desc:    "valid four-byte sequence overflowing short buffer",
		src:     "a\u0300bc\xf0\x90\x80\x80",
		notEOF:  true,
		sizeDst: 8,
		want:    "a\u0300bc",
		nSrc:    5,
		err:     transform.ErrShortDst,
	}, {
		desc:    "invalid fourth byte at end short, but short dst",
		src:     "a\u0300bc\xf0\x90\x80\xf0\x90\x80",
		notEOF:  true,
		sizeDst: 8,
		// More bytes would fit in the buffer, but this seems to require a more
		// complicated and slower algorithm.
		want: "a\u0300bc", // instead of "a\u0300bc"
		nSrc: 5,
		err:  transform.ErrShortDst,
	}, {
		desc:    "short dst for error",
		src:     "abc\x80",
		notEOF:  true,
		sizeDst: 5,
		want:    "abc",
		nSrc:    3,
		err:     transform.ErrShortDst,
	}, {
		desc:    "adjusting short dst buffer",
		src:     "abc\x80ef",
		notEOF:  true,
		sizeDst: 6,
		want:    "abc\ufffd",
		nSrc:    4,
		err:     transform.ErrShortDst,
	}}
	tr := UTF8.NewDecoder()
	for i, tc := range testCases {
		b := make([]byte, tc.sizeDst)
		nDst, nSrc, err := tr.Transform(b, []byte(tc.src), !tc.notEOF)
		if err != tc.err {
			t.Errorf("%d:%s: error was %v; want %v", i, tc.desc, err, tc.err)
		}
		if got := string(b[:nDst]); got != tc.want {
			t.Errorf("%d:%s: result was %q: want %q", i, tc.desc, got, tc.want)
		}
		if nSrc != tc.nSrc {
			t.Errorf("%d:%s: nSrc was %d; want %d", i, tc.desc, nSrc, tc.nSrc)
		}
	}
}
