// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utf32

import (
	"testing"

	"golang.org/x/text/transform"
)

var (
	utf32LEIB = UTF32(LittleEndian, IgnoreBOM) // UTF-32LE (atypical interpretation)
	utf32LEUB = UTF32(LittleEndian, UseBOM)    // UTF-32, LE
	//	utf32LEEB = UTF32(LittleEndian, ExpectBOM) // UTF-32, LE, Expect - covered in encoding_test.go
	utf32BEIB = UTF32(BigEndian, IgnoreBOM) // UTF-32BE (atypical interpretation)
	utf32BEUB = UTF32(BigEndian, UseBOM)    // UTF-32 default
	utf32BEEB = UTF32(BigEndian, ExpectBOM) // UTF-32 Expect
)

func TestUTF32(t *testing.T) {
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
		desc: "utf-32 IgnoreBOM dec: empty string",
		t:    utf32BEIB.NewDecoder(),
	}, {
		desc: "utf-32 UseBOM dec: empty string",
		t:    utf32BEUB.NewDecoder(),
	}, {
		desc: "utf-32 ExpectBOM dec: empty string",
		err:  ErrMissingBOM,
		t:    utf32BEEB.NewDecoder(),
	}, {
		desc:    "utf-32be dec: Doesn't interpret U+FEFF as BOM",
		src:     "\x00\x00\xFE\xFF\x00\x01\x23\x45\x00\x00\x00\x3D\x00\x00\x00\x52\x00\x00\x00\x61",
		sizeDst: 100,
		want:    "\uFEFF\U00012345=Ra",
		nSrc:    20,
		t:       utf32BEIB.NewDecoder(),
	}, {
		desc:    "utf-32be dec: Interprets little endian U+FEFF as invalid",
		src:     "\xFF\xFE\x00\x00\x00\x01\x23\x45\x00\x00\x00\x3D\x00\x00\x00\x52\x00\x00\x00\x61",
		sizeDst: 100,
		want:    "\uFFFD\U00012345=Ra",
		nSrc:    20,
		t:       utf32BEIB.NewDecoder(),
	}, {
		desc:    "utf-32le dec: Doesn't interpret U+FEFF as BOM",
		src:     "\xFF\xFE\x00\x00\x45\x23\x01\x00\x3D\x00\x00\x00\x52\x00\x00\x00\x61\x00\x00\x00",
		sizeDst: 100,
		want:    "\uFEFF\U00012345=Ra",
		nSrc:    20,
		t:       utf32LEIB.NewDecoder(),
	}, {
		desc:    "utf-32le dec: Interprets big endian U+FEFF as invalid",
		src:     "\x00\x00\xFE\xFF\x45\x23\x01\x00\x3D\x00\x00\x00\x52\x00\x00\x00\x61\x00\x00\x00",
		sizeDst: 100,
		want:    "\uFFFD\U00012345=Ra",
		nSrc:    20,
		t:       utf32LEIB.NewDecoder(),
	}, {
		desc:    "utf-32 enc: Writes big-endian BOM",
		src:     "\U00012345=Ra",
		sizeDst: 100,
		want:    "\x00\x00\xFE\xFF\x00\x01\x23\x45\x00\x00\x00\x3D\x00\x00\x00\x52\x00\x00\x00\x61",
		nSrc:    7,
		t:       utf32BEUB.NewEncoder(),
	}, {
		desc:    "utf-32 enc: Writes little-endian BOM",
		src:     "\U00012345=Ra",
		sizeDst: 100,
		want:    "\xFF\xFE\x00\x00\x45\x23\x01\x00\x3D\x00\x00\x00\x52\x00\x00\x00\x61\x00\x00\x00",
		nSrc:    7,
		t:       utf32LEUB.NewEncoder(),
	}, {
		desc:    "utf-32 dec: Interprets text using big-endian default when BOM not present",
		src:     "\x00\x01\x23\x45\x00\x00\x00\x3D\x00\x00\x00\x52\x00\x00\x00\x61",
		sizeDst: 100,
		want:    "\U00012345=Ra",
		nSrc:    16,
		t:       utf32BEUB.NewDecoder(),
	}, {
		desc:    "utf-32 dec: Interprets text using little-endian default when BOM not present",
		src:     "\x45\x23\x01\x00\x3D\x00\x00\x00\x52\x00\x00\x00\x61\x00\x00\x00",
		sizeDst: 100,
		want:    "\U00012345=Ra",
		nSrc:    16,
		t:       utf32LEUB.NewDecoder(),
	}, {
		desc:    "utf-32 dec: BOM determines encoding BE",
		src:     "\x00\x00\xFE\xFF\x00\x01\x23\x45\x00\x00\x00\x3D\x00\x00\x00\x52\x00\x00\x00\x61",
		sizeDst: 100,
		want:    "\U00012345=Ra",
		nSrc:    20,
		t:       utf32BEUB.NewDecoder(),
	}, {
		desc:    "utf-32 dec: BOM determines encoding LE",
		src:     "\xFF\xFE\x00\x00\x45\x23\x01\x00\x3D\x00\x00\x00\x52\x00\x00\x00\x61\x00\x00\x00",
		sizeDst: 100,
		want:    "\U00012345=Ra",
		nSrc:    20,
		t:       utf32LEUB.NewDecoder(),
	}, {
		desc:    "utf-32 dec: BOM determines encoding LE, change default",
		src:     "\xFF\xFE\x00\x00\x45\x23\x01\x00\x3D\x00\x00\x00\x52\x00\x00\x00\x61\x00\x00\x00",
		sizeDst: 100,
		want:    "\U00012345=Ra",
		nSrc:    20,
		t:       utf32BEUB.NewDecoder(),
	}, {
		desc:    "utf-32 dec: BOM determines encoding BE, change default",
		src:     "\x00\x00\xFE\xFF\x00\x01\x23\x45\x00\x00\x00\x3D\x00\x00\x00\x52\x00\x00\x00\x61",
		sizeDst: 100,
		want:    "\U00012345=Ra",
		nSrc:    20,
		t:       utf32LEUB.NewDecoder(),
	}, {
		desc:    "utf-32 dec: Don't change big-endian byte order mid-stream",
		src:     "\x00\x01\x23\x45\x00\x00\x00\x3D\xFF\xFE\x00\x00\x00\x00\xFE\xFF\x00\x00\x00\x52\x00\x00\x00\x61",
		sizeDst: 100,
		want:    "\U00012345=\uFFFD\uFEFFRa",
		nSrc:    24,
		t:       utf32BEUB.NewDecoder(),
	}, {
		desc:    "utf-32 dec: Don't change little-endian byte order mid-stream",
		src:     "\x45\x23\x01\x00\x3D\x00\x00\x00\x00\x00\xFE\xFF\xFF\xFE\x00\x00\x52\x00\x00\x00\x61\x00\x00\x00",
		sizeDst: 100,
		want:    "\U00012345=\uFFFD\uFEFFRa",
		nSrc:    24,
		t:       utf32LEUB.NewDecoder(),
	}, {
		desc:    "utf-32 dec: Fail on missing BOM when required",
		src:     "\x00\x01\x23\x45\x00\x00\x00\x3D\x00\x00\x00\x52\x00\x00\x00\x61",
		sizeDst: 100,
		want:    "",
		nSrc:    0,
		err:     ErrMissingBOM,
		t:       utf32BEEB.NewDecoder(),
	}, {
		desc:    "utf-32 enc: Short dst",
		src:     "\U00012345=Ra",
		sizeDst: 15,
		want:    "\x00\x01\x23\x45\x00\x00\x00\x3D\x00\x00\x00\x52",
		nSrc:    6,
		err:     transform.ErrShortDst,
		t:       utf32BEIB.NewEncoder(),
	}, {
		desc:    "utf-32 enc: Short src",
		src:     "\U00012345=Ra\xC2",
		notEOF:  true,
		sizeDst: 100,
		want:    "\x00\x01\x23\x45\x00\x00\x00\x3D\x00\x00\x00\x52\x00\x00\x00\x61",
		nSrc:    7,
		err:     transform.ErrShortSrc,
		t:       utf32BEIB.NewEncoder(),
	}, {
		desc:    "utf-32 enc: Invalid input",
		src:     "\x80\xC1\xC2\x7F\xC2",
		sizeDst: 100,
		want:    "\x00\x00\xFF\xFD\x00\x00\xFF\xFD\x00\x00\xFF\xFD\x00\x00\x00\x7F\x00\x00\xFF\xFD",
		nSrc:    5,
		t:       utf32BEIB.NewEncoder(),
	}, {
		desc:    "utf-32 dec: Short dst",
		src:     "\x00\x00\x00\x41",
		sizeDst: 0,
		want:    "",
		nSrc:    0,
		err:     transform.ErrShortDst,
		t:       utf32BEIB.NewDecoder(),
	}, {
		desc:    "utf-32 dec: Short src",
		src:     "\x00\x00\x00",
		notEOF:  true,
		sizeDst: 4,
		want:    "",
		nSrc:    0,
		err:     transform.ErrShortSrc,
		t:       utf32BEIB.NewDecoder(),
	}, {
		desc:    "utf-32 dec: Invalid input",
		src:     "\x00\x00\xD8\x00\x00\x00\xDF\xFF\x00\x11\x00\x00\x00\x00\x00",
		sizeDst: 100,
		want:    "\uFFFD\uFFFD\uFFFD\uFFFD",
		nSrc:    15,
		t:       utf32BEIB.NewDecoder(),
	}}
	for i, tc := range testCases {
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
	}
}
