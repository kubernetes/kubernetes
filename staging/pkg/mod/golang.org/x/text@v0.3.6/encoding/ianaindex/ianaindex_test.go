// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ianaindex

import (
	"testing"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/charmap"
	"golang.org/x/text/encoding/internal/identifier"
	"golang.org/x/text/encoding/japanese"
	"golang.org/x/text/encoding/korean"
	"golang.org/x/text/encoding/simplifiedchinese"
	"golang.org/x/text/encoding/traditionalchinese"
	"golang.org/x/text/encoding/unicode"
)

var All = [][]encoding.Encoding{
	unicode.All,
	charmap.All,
	japanese.All,
	korean.All,
	simplifiedchinese.All,
	traditionalchinese.All,
}

// TestAllIANA tests whether an Encoding supported in x/text is defined by IANA but
// not supported by this package.
func TestAllIANA(t *testing.T) {
	for _, ea := range All {
		for _, e := range ea {
			mib, _ := e.(identifier.Interface).ID()
			if x := findMIB(ianaToMIB, mib); x != -1 && encodings[x] == nil {
				t.Errorf("supported MIB %v (%v) not in index", mib, e)
			}
		}
	}
}

// TestNotSupported reports the encodings in IANA, but not by x/text.
func TestNotSupported(t *testing.T) {
	mibs := map[identifier.MIB]bool{}
	for _, ea := range All {
		for _, e := range ea {
			mib, _ := e.(identifier.Interface).ID()
			mibs[mib] = true
		}
	}

	// Many encodings in the IANA index will likely not be suppored by the
	// Go encodings. That is fine.
	// TODO: consider wheter we should add this test.
	// for code, mib := range ianaToMIB {
	// 	t.Run(fmt.Sprint("IANA:", mib), func(t *testing.T) {
	// 		if !mibs[mib] {
	// 			t.Skipf("IANA encoding %s (MIB %v) not supported",
	// 				ianaNames[code], mib)
	// 		}
	// 	})
	// }
}

func TestEncoding(t *testing.T) {
	testCases := []struct {
		index     *Index
		name      string
		canonical string
		err       error
	}{
		{MIME, "utf-8", "UTF-8", nil},
		{MIME, "  utf-8  ", "UTF-8", nil},
		{MIME, "  l5  ", "ISO-8859-9", nil},
		{MIME, "latin5 ", "ISO-8859-9", nil},
		{MIME, "LATIN5 ", "ISO-8859-9", nil},
		{MIME, "us-ascii", "US-ASCII", nil},
		{MIME, "latin 5", "", errInvalidName},
		{MIME, "latin-5", "", errInvalidName},

		{IANA, "utf-8", "UTF-8", nil},
		{IANA, "  utf-8  ", "UTF-8", nil},
		{IANA, "  l5  ", "ISO_8859-9:1989", nil},
		{IANA, "latin5 ", "ISO_8859-9:1989", nil},
		{IANA, "LATIN5 ", "ISO_8859-9:1989", nil},
		{IANA, "latin 5", "", errInvalidName},
		{IANA, "latin-5", "", errInvalidName},

		{MIB, "utf-8", "UTF8", nil},
		{MIB, "  utf-8  ", "UTF8", nil},
		{MIB, "  l5  ", "ISOLatin5", nil},
		{MIB, "latin5 ", "ISOLatin5", nil},
		{MIB, "LATIN5 ", "ISOLatin5", nil},
		{MIB, "latin 5", "", errInvalidName},
		{MIB, "latin-5", "", errInvalidName},
	}
	for i, tc := range testCases {
		enc, err := tc.index.Encoding(tc.name)
		if err != tc.err {
			t.Errorf("%d: error was %v; want %v", i, err, tc.err)
		}
		if err != nil {
			continue
		}
		if got, err := tc.index.Name(enc); got != tc.canonical {
			t.Errorf("%d: Name(Encoding(%q)) = %q; want %q (%v)", i, tc.name, got, tc.canonical, err)
		}
	}
}

func TestTables(t *testing.T) {
	for i, x := range []*Index{MIME, IANA} {
		for name, index := range x.alias {
			got, err := x.Encoding(name)
			if err != nil {
				t.Errorf("%d%s:err: unexpected error %v", i, name, err)
			}
			if want := x.enc[index]; got != want {
				t.Errorf("%d%s:encoding: got %v; want %v", i, name, got, want)
			}
			if got != nil {
				mib, _ := got.(identifier.Interface).ID()
				if i := findMIB(x.toMIB, mib); i != index {
					t.Errorf("%d%s:mib: got %d; want %d", i, name, i, index)
				}
			}
		}
	}
}

type unsupported struct {
	encoding.Encoding
}

func (unsupported) ID() (identifier.MIB, string) { return 9999, "" }

func TestName(t *testing.T) {
	testCases := []struct {
		desc string
		enc  encoding.Encoding
		f    func(e encoding.Encoding) (string, error)
		name string
		err  error
	}{{
		"defined encoding",
		charmap.ISO8859_2,
		MIME.Name,
		"ISO-8859-2",
		nil,
	}, {
		"defined Unicode encoding",
		unicode.UTF16(unicode.BigEndian, unicode.IgnoreBOM),
		IANA.Name,
		"UTF-16BE",
		nil,
	}, {
		"another defined Unicode encoding",
		unicode.UTF16(unicode.BigEndian, unicode.UseBOM),
		MIME.Name,
		"UTF-16",
		nil,
	}, {
		"unknown Unicode encoding",
		unicode.UTF16(unicode.BigEndian, unicode.ExpectBOM),
		MIME.Name,
		"",
		errUnknown,
	}, {
		"undefined encoding",
		unsupported{},
		MIME.Name,
		"",
		errUnsupported,
	}, {
		"undefined other encoding in HTML standard",
		charmap.CodePage437,
		IANA.Name,
		"IBM437",
		nil,
	}, {
		"unknown encoding",
		encoding.Nop,
		IANA.Name,
		"",
		errUnknown,
	}}
	for i, tc := range testCases {
		name, err := tc.f(tc.enc)
		if name != tc.name || err != tc.err {
			t.Errorf("%d:%s: got %q, %v; want %q, %v", i, tc.desc, name, err, tc.name, tc.err)
		}
	}
}
