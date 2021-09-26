// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unicode

import (
	"testing"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/charmap"
	"golang.org/x/text/encoding/internal/enctest"
	"golang.org/x/text/transform"
)

func TestBasics(t *testing.T) {
	testCases := []struct {
		e         encoding.Encoding
		encPrefix string
		encSuffix string
		encoded   string
		utf8      string
	}{{
		e:       utf16BEIB,
		encoded: "\x00\x57\x00\xe4\xd8\x35\xdd\x65",
		utf8:    "\x57\u00e4\U0001d565",
	}, {
		e:         utf16BEEB,
		encPrefix: "\xfe\xff",
		encoded:   "\x00\x57\x00\xe4\xd8\x35\xdd\x65",
		utf8:      "\x57\u00e4\U0001d565",
	}, {
		e:       utf16LEIB,
		encoded: "\x57\x00\xe4\x00\x35\xd8\x65\xdd",
		utf8:    "\x57\u00e4\U0001d565",
	}, {
		e:         utf16LEEB,
		encPrefix: "\xff\xfe",
		encoded:   "\x57\x00\xe4\x00\x35\xd8\x65\xdd",
		utf8:      "\x57\u00e4\U0001d565",
	}}

	for _, tc := range testCases {
		enctest.TestEncoding(t, tc.e, tc.encoded, tc.utf8, tc.encPrefix, tc.encSuffix)
	}
}

func TestFiles(t *testing.T) {
	enctest.TestFile(t, UTF8)
	enctest.TestFile(t, utf16LEIB)
}

func BenchmarkEncoding(b *testing.B) {
	enctest.Benchmark(b, UTF8)
	enctest.Benchmark(b, utf16LEIB)
}

var (
	utf16LEIB = UTF16(LittleEndian, IgnoreBOM) // UTF-16LE (atypical interpretation)
	utf16LEUB = UTF16(LittleEndian, UseBOM)    // UTF-16, LE
	utf16LEEB = UTF16(LittleEndian, ExpectBOM) // UTF-16, LE, Expect
	utf16BEIB = UTF16(BigEndian, IgnoreBOM)    // UTF-16BE (atypical interpretation)
	utf16BEUB = UTF16(BigEndian, UseBOM)       // UTF-16 default
	utf16BEEB = UTF16(BigEndian, ExpectBOM)    // UTF-16 Expect
)

func TestUTF16(t *testing.T) {
	testCases := []struct {
		desc    string
		src     string
		notEOF  bool // the inverse of atEOF
		sizeDst int
		want    string
		nSrc    int
		err     error
		t       transform.Transformer
	}{{
		desc: "utf-16 IgnoreBOM dec: empty string",
		t:    utf16BEIB.NewDecoder(),
	}, {
		desc: "utf-16 UseBOM dec: empty string",
		t:    utf16BEUB.NewDecoder(),
	}, {
		desc: "utf-16 ExpectBOM dec: empty string",
		err:  ErrMissingBOM,
		t:    utf16BEEB.NewDecoder(),
	}, {
		desc:    "utf-16 dec: BOM determines encoding BE (RFC 2781:3.3)",
		src:     "\xFE\xFF\xD8\x08\xDF\x45\x00\x3D\x00\x52\x00\x61",
		sizeDst: 100,
		want:    "\U00012345=Ra",
		nSrc:    12,
		t:       utf16BEUB.NewDecoder(),
	}, {
		desc:    "utf-16 dec: BOM determines encoding LE (RFC 2781:3.3)",
		src:     "\xFF\xFE\x08\xD8\x45\xDF\x3D\x00\x52\x00\x61\x00",
		sizeDst: 100,
		want:    "\U00012345=Ra",
		nSrc:    12,
		t:       utf16LEUB.NewDecoder(),
	}, {
		desc:    "utf-16 dec: BOM determines encoding LE, change default (RFC 2781:3.3)",
		src:     "\xFF\xFE\x08\xD8\x45\xDF\x3D\x00\x52\x00\x61\x00",
		sizeDst: 100,
		want:    "\U00012345=Ra",
		nSrc:    12,
		t:       utf16BEUB.NewDecoder(),
	}, {
		desc:    "utf-16 dec: Fail on missing BOM when required",
		src:     "\x08\xD8\x45\xDF\x3D\x00\xFF\xFE\xFE\xFF\x00\x52\x00\x61",
		sizeDst: 100,
		want:    "",
		nSrc:    0,
		err:     ErrMissingBOM,
		t:       utf16BEEB.NewDecoder(),
	}, {
		desc:    "utf-16 dec: Fail on single byte missing BOM when required",
		src:     "\x00",
		sizeDst: 4,
		t:       utf16BEEB.NewDecoder(),
		err:     ErrMissingBOM,
	}, {
		desc:    "utf-16 dec: Fail on short src missing BOM when required",
		src:     "\x00",
		notEOF:  true,
		sizeDst: 4,
		t:       utf16BEEB.NewDecoder(),
		err:     transform.ErrShortSrc,
	}, {
		desc:    "utf-16 dec: SHOULD interpret text as big-endian when BOM not present (RFC 2781:4.3)",
		src:     "\xD8\x08\xDF\x45\x00\x3D\x00\x52\x00\x61",
		sizeDst: 100,
		want:    "\U00012345=Ra",
		nSrc:    10,
		t:       utf16BEUB.NewDecoder(),
	}, {
		desc:    "utf-16 dec: incorrect UTF-16: odd bytes",
		src:     "\x00",
		sizeDst: 100,
		want:    "\uFFFD",
		nSrc:    1,
		t:       utf16BEUB.NewDecoder(),
	}, {
		desc:    "utf-16 dec: Fail on incorrect UTF-16: short source odd bytes",
		src:     "\x00",
		notEOF:  true,
		sizeDst: 100,
		t:       utf16BEUB.NewDecoder(),
		err:     transform.ErrShortSrc,
	}, {
		// This is an error according to RFC 2781. But errors in RFC 2781 are
		// open to interpretations, so I guess this is fine.
		desc:    "utf-16le dec: incorrect BOM is an error (RFC 2781:4.1)",
		src:     "\xFE\xFF\x08\xD8\x45\xDF\x3D\x00\x52\x00\x61\x00",
		sizeDst: 100,
		want:    "\uFFFE\U00012345=Ra",
		nSrc:    12,
		t:       utf16LEIB.NewDecoder(),
	}, {
		desc:    "utf-16 enc: SHOULD write BOM (RFC 2781:3.3)",
		src:     "\U00012345=Ra",
		sizeDst: 100,
		want:    "\xFF\xFE\x08\xD8\x45\xDF\x3D\x00\x52\x00\x61\x00",
		nSrc:    7,
		t:       utf16LEUB.NewEncoder(),
	}, {
		desc:    "utf-16 enc: SHOULD write BOM (RFC 2781:3.3)",
		src:     "\U00012345=Ra",
		sizeDst: 100,
		want:    "\xFE\xFF\xD8\x08\xDF\x45\x00\x3D\x00\x52\x00\x61",
		nSrc:    7,
		t:       utf16BEUB.NewEncoder(),
	}, {
		desc:    "utf-16le enc: MUST NOT write BOM (RFC 2781:3.3)",
		src:     "\U00012345=Ra",
		sizeDst: 100,
		want:    "\x08\xD8\x45\xDF\x3D\x00\x52\x00\x61\x00",
		nSrc:    7,
		t:       utf16LEIB.NewEncoder(),
	}, {
		desc:    "utf-16be dec: incorrect UTF-16: odd bytes",
		src:     "\x00",
		sizeDst: 100,
		want:    "\uFFFD",
		nSrc:    1,
		t:       utf16BEIB.NewDecoder(),
	}, {
		desc:    "utf-16be dec: unpaired surrogate, odd bytes",
		src:     "\xD8\x45\x00",
		sizeDst: 100,
		want:    "\uFFFD\uFFFD",
		nSrc:    3,
		t:       utf16BEIB.NewDecoder(),
	}, {
		desc:    "utf-16be dec: unpaired low surrogate + valid text",
		src:     "\xD8\x45\x00a",
		sizeDst: 100,
		want:    "\uFFFDa",
		nSrc:    4,
		t:       utf16BEIB.NewDecoder(),
	}, {
		desc:    "utf-16be dec: unpaired low surrogate + valid text + single byte",
		src:     "\xD8\x45\x00ab",
		sizeDst: 100,
		want:    "\uFFFDa\uFFFD",
		nSrc:    5,
		t:       utf16BEIB.NewDecoder(),
	}, {
		desc:    "utf-16le dec: unpaired high surrogate",
		src:     "\x00\x00\x00\xDC\x12\xD8",
		sizeDst: 100,
		want:    "\x00\uFFFD\uFFFD",
		nSrc:    6,
		t:       utf16LEIB.NewDecoder(),
	}, {
		desc:    "utf-16be dec: two unpaired low surrogates",
		src:     "\xD8\x45\xD8\x12",
		sizeDst: 100,
		want:    "\uFFFD\uFFFD",
		nSrc:    4,
		t:       utf16BEIB.NewDecoder(),
	}, {
		desc:    "utf-16be dec: short dst",
		src:     "\x00a",
		sizeDst: 0,
		want:    "",
		nSrc:    0,
		t:       utf16BEIB.NewDecoder(),
		err:     transform.ErrShortDst,
	}, {
		desc:    "utf-16be dec: short dst surrogate",
		src:     "\xD8\xF5\xDC\x12",
		sizeDst: 3,
		want:    "",
		nSrc:    0,
		t:       utf16BEIB.NewDecoder(),
		err:     transform.ErrShortDst,
	}, {
		desc:    "utf-16be dec: short dst trailing byte",
		src:     "\x00",
		sizeDst: 2,
		want:    "",
		nSrc:    0,
		t:       utf16BEIB.NewDecoder(),
		err:     transform.ErrShortDst,
	}, {
		desc:    "utf-16be dec: short src",
		src:     "\x00",
		notEOF:  true,
		sizeDst: 3,
		want:    "",
		nSrc:    0,
		t:       utf16BEIB.NewDecoder(),
		err:     transform.ErrShortSrc,
	}, {
		desc:    "utf-16 enc",
		src:     "\U00012345=Ra",
		sizeDst: 100,
		want:    "\xFE\xFF\xD8\x08\xDF\x45\x00\x3D\x00\x52\x00\x61",
		nSrc:    7,
		t:       utf16BEUB.NewEncoder(),
	}, {
		desc:    "utf-16 enc: short dst normal",
		src:     "\U00012345=Ra",
		sizeDst: 9,
		want:    "\xD8\x08\xDF\x45\x00\x3D\x00\x52",
		nSrc:    6,
		t:       utf16BEIB.NewEncoder(),
		err:     transform.ErrShortDst,
	}, {
		desc:    "utf-16 enc: short dst surrogate",
		src:     "\U00012345=Ra",
		sizeDst: 3,
		want:    "",
		nSrc:    0,
		t:       utf16BEIB.NewEncoder(),
		err:     transform.ErrShortDst,
	}, {
		desc:    "utf-16 enc: short src",
		src:     "\U00012345=Ra\xC2",
		notEOF:  true,
		sizeDst: 100,
		want:    "\xD8\x08\xDF\x45\x00\x3D\x00\x52\x00\x61",
		nSrc:    7,
		t:       utf16BEIB.NewEncoder(),
		err:     transform.ErrShortSrc,
	}, {
		desc:    "utf-16be dec: don't change byte order mid-stream",
		src:     "\xFE\xFF\xD8\x08\xDF\x45\x00\x3D\xFF\xFE\x00\x52\x00\x61",
		sizeDst: 100,
		want:    "\U00012345=\ufffeRa",
		nSrc:    14,
		t:       utf16BEUB.NewDecoder(),
	}, {
		desc:    "utf-16le dec: don't change byte order mid-stream",
		src:     "\xFF\xFE\x08\xD8\x45\xDF\x3D\x00\xFF\xFE\xFE\xFF\x52\x00\x61\x00",
		sizeDst: 100,
		want:    "\U00012345=\ufeff\ufffeRa",
		nSrc:    16,
		t:       utf16LEUB.NewDecoder(),
	}}
	for i, tc := range testCases {
		for j := 0; j < 2; j++ {
			b := make([]byte, tc.sizeDst)
			nDst, nSrc, err := tc.t.Transform(b, []byte(tc.src), !tc.notEOF)
			if err != tc.err {
				t.Errorf("%d:%s: error was %v; want %v", i, tc.desc, err, tc.err)
			}
			if got := string(b[:nDst]); got != tc.want {
				t.Errorf("%d:%s: result was %q: want %q", i, tc.desc, got, tc.want)
			}
			if nSrc != tc.nSrc {
				t.Errorf("%d:%s: nSrc was %d; want %d", i, tc.desc, nSrc, tc.nSrc)
			}
			// Since Transform is stateful, run failures again
			// to ensure that the same error occurs a second time.
			if err == nil {
				break
			}
		}
	}
}

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

func TestUTF8BOMDecoder(t *testing.T) {
	testCases := []struct {
		desc    string
		src     string
		notEOF  bool // the inverse of atEOF
		sizeDst int
		want    string
		nSrc    int
		err     error
		wantAll string
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
		wantAll: "abcde",
	}, {
		desc:    "ascii with bom",
		src:     utf8BOM + "abcde",
		sizeDst: 11,
		want:    "abcde",
		nSrc:    8,
		wantAll: "abcde",
	}, {
		desc:    "error with bom",
		src:     utf8BOM + "ab\x80de",
		sizeDst: 11,
		want:    "ab\ufffdde",
		nSrc:    8,
		wantAll: "ab\ufffdde",
	}, {
		desc:    "short bom",
		src:     utf8BOM[:2],
		notEOF:  true,
		sizeDst: 7,
		want:    "",
		nSrc:    0,
		wantAll: "\ufffd", // needs to be 1 replacement
		err:     transform.ErrShortSrc,
	}, {
		desc:    "short bom at end",
		src:     utf8BOM[:2],
		sizeDst: 7,
		want:    "\ufffd", // needs to be 1 replacement
		nSrc:    2,
		wantAll: "\ufffd", // needs to be 1 replacement
		err:     nil,
	}, {
		desc:    "short source buffer",
		src:     "abc\xf0\x90",
		notEOF:  true,
		sizeDst: 10,
		want:    "abc",
		nSrc:    3,
		wantAll: "abc\ufffd",
		err:     transform.ErrShortSrc,
	}, {
		desc:    "short source buffer with bom",
		src:     utf8BOM + "abc\xf0\x90",
		notEOF:  true,
		sizeDst: 15,
		want:    "abc",
		nSrc:    6,
		wantAll: "abc\ufffd",
		err:     transform.ErrShortSrc,
	}, {
		desc:    "short dst for error",
		src:     utf8BOM + "abc\x80",
		notEOF:  true,
		sizeDst: 5,
		want:    "abc",
		nSrc:    6,
		wantAll: "abc\ufffd",
		err:     transform.ErrShortDst,
	}}
	tr := UTF8BOM.NewDecoder()
	for i, tc := range testCases {
		tr.Reset()
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
		if got, _ := tr.String(tc.src); got != tc.wantAll {
			t.Errorf("%d:%s: String was %s; want %s", i, tc.desc, got, tc.wantAll)
		}
	}
}

func TestUTF8SigEncoder(t *testing.T) {
	testCases := []struct {
		desc    string
		src     string
		notEOF  bool // the inverse of atEOF
		sizeDst int
		want    string
		wantAll string // converting all bytes
		nSrc    int
		err     error
	}{{
		desc:    "empty string, empty dest buffer",
		err:     transform.ErrShortDst,
		wantAll: utf8BOM,
	}, {
		desc:    "empty string",
		sizeDst: 8,
		want:    utf8BOM,
		wantAll: utf8BOM,
	}, {
		desc:    "empty string, streaming",
		notEOF:  true,
		sizeDst: 8,
		want:    utf8BOM,
		wantAll: utf8BOM,
	}, {
		desc:    "ascii",
		src:     "abcde",
		sizeDst: 8,
		want:    utf8BOM + "abcde",
		nSrc:    5,
		wantAll: utf8BOM + "abcde",
	}, {
		desc:    "short bom at end",
		src:     utf8BOM[:2],
		sizeDst: 11,
		want:    utf8BOM + "\ufffd",
		nSrc:    2,
		wantAll: utf8BOM + "\ufffd",
	}, {
		desc:    "short bom",
		src:     utf8BOM[:2],
		notEOF:  true,
		sizeDst: 7,
		want:    utf8BOM,
		nSrc:    0,
		err:     transform.ErrShortSrc,
		wantAll: utf8BOM + "\ufffd",
	}, {
		desc:    "short bom at end",
		src:     utf8BOM[:2],
		sizeDst: 7,
		want:    utf8BOM + "\ufffd", // needs to be 1 replacement
		nSrc:    2,
		err:     nil,
		wantAll: utf8BOM + "\ufffd",
	}, {
		desc:    "short dst buffer 2",
		src:     "ab",
		sizeDst: 2,
		want:    "",
		nSrc:    0,
		err:     transform.ErrShortDst,
		wantAll: utf8BOM + "ab",
	}, {
		desc:    "short dst buffer 3",
		src:     "ab",
		sizeDst: 3,
		want:    utf8BOM,
		nSrc:    0,
		err:     transform.ErrShortDst,
		wantAll: utf8BOM + "ab",
	}, {
		desc:    "short dst buffer 4",
		src:     "ab",
		sizeDst: 4,
		want:    utf8BOM + "a",
		nSrc:    1,
		err:     transform.ErrShortDst,
		wantAll: utf8BOM + "ab",
	}}
	tr := UTF8BOM.NewEncoder()
	for i, tc := range testCases {
		tr.Reset()
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
		if got, _ := tr.String(tc.src); got != tc.wantAll {
			t.Errorf("%d:%s: String was %s; want %s", i, tc.desc, got, tc.wantAll)
		}
	}
}

func TestBOMOverride(t *testing.T) {
	dec := BOMOverride(charmap.CodePage437.NewDecoder())
	dst := make([]byte, 100)
	for i, tc := range []struct {
		src   string
		atEOF bool
		dst   string
		nSrc  int
		err   error
	}{
		0:  {"H\x82ll\x93", true, "Héllô", 5, nil},
		1:  {"\uFEFFHéllö", true, "Héllö", 10, nil},
		2:  {"\xFE\xFF\x00H\x00e\x00l\x00l\x00o", true, "Hello", 12, nil},
		3:  {"\xFF\xFEH\x00e\x00l\x00l\x00o\x00", true, "Hello", 12, nil},
		4:  {"\uFEFF", true, "", 3, nil},
		5:  {"\xFE\xFF", true, "", 2, nil},
		6:  {"\xFF\xFE", true, "", 2, nil},
		7:  {"\xEF\xBB", true, "\u2229\u2557", 2, nil},
		8:  {"\xEF", true, "\u2229", 1, nil},
		9:  {"", true, "", 0, nil},
		10: {"\xFE", true, "\u25a0", 1, nil},
		11: {"\xFF", true, "\u00a0", 1, nil},
		12: {"\xEF\xBB", false, "", 0, transform.ErrShortSrc},
		13: {"\xEF", false, "", 0, transform.ErrShortSrc},
		14: {"", false, "", 0, transform.ErrShortSrc},
		15: {"\xFE", false, "", 0, transform.ErrShortSrc},
		16: {"\xFF", false, "", 0, transform.ErrShortSrc},
		17: {"\xFF\xFE", false, "", 0, transform.ErrShortSrc},
	} {
		dec.Reset()
		nDst, nSrc, err := dec.Transform(dst, []byte(tc.src), tc.atEOF)
		got := string(dst[:nDst])
		if nSrc != tc.nSrc {
			t.Errorf("%d: nSrc: got %d; want %d", i, nSrc, tc.nSrc)
		}
		if got != tc.dst {
			t.Errorf("%d: got %+q; want %+q", i, got, tc.dst)
		}
		if err != tc.err {
			t.Errorf("%d: error: got %v; want %v", i, err, tc.err)
		}
	}
}
