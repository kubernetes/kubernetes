// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package korean

import (
	"strings"
	"testing"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/internal"
	"golang.org/x/text/encoding/internal/enctest"
	"golang.org/x/text/transform"
)

func dec(e encoding.Encoding) (dir string, t transform.Transformer, err error) {
	return "Decode", e.NewDecoder(), nil
}
func enc(e encoding.Encoding) (dir string, t transform.Transformer, err error) {
	return "Encode", e.NewEncoder(), internal.ErrASCIIReplacement
}

func TestNonRepertoire(t *testing.T) {
	// Pick n large enough to cause an overflow in the destination buffer of
	// transform.String.
	const n = 10000
	testCases := []struct {
		init      func(e encoding.Encoding) (string, transform.Transformer, error)
		e         encoding.Encoding
		src, want string
	}{
		{dec, EUCKR, "\xfe\xfe", "\ufffd"},
		// {dec, EUCKR, "א", "\ufffd"}, // TODO: why is this different?

		{enc, EUCKR, "א", ""},
		{enc, EUCKR, "aא", "a"},
		{enc, EUCKR, "\uac00א", "\xb0\xa1"},
		// TODO: should we also handle Jamo?

		{dec, EUCKR, "\x80", "\ufffd"},
		{dec, EUCKR, "\xff", "\ufffd"},
		{dec, EUCKR, "\x81", "\ufffd"},
		{dec, EUCKR, "\xb0\x40", "\ufffd@"},
		{dec, EUCKR, "\xb0\xff", "\ufffd"},
		{dec, EUCKR, "\xd0\x20", "\ufffd "},
		{dec, EUCKR, "\xd0\xff", "\ufffd"},

		{dec, EUCKR, strings.Repeat("\x81", n), strings.Repeat("걖", n/2)},
	}
	for _, tc := range testCases {
		dir, tr, wantErr := tc.init(tc.e)

		dst, _, err := transform.String(tr, tc.src)
		if err != wantErr {
			t.Errorf("%s %v(%q): got %v; want %v", dir, tc.e, tc.src, err, wantErr)
		}
		if got := string(dst); got != tc.want {
			t.Errorf("%s %v(%q):\ngot  %q\nwant %q", dir, tc.e, tc.src, got, tc.want)
		}
	}
}

func TestBasics(t *testing.T) {
	// The encoded forms can be verified by the iconv program:
	// $ echo 月日は百代 | iconv -f UTF-8 -t SHIFT-JIS | xxd
	testCases := []struct {
		e       encoding.Encoding
		encoded string
		utf8    string
	}{{
		// Korean tests.
		//
		// "A\uac02\uac35\uac56\ud401B\ud408\ud620\ud624C\u4f3d\u8a70D" is a
		// nonsense string that contains ASCII, Hangul and CJK ideographs.
		//
		// "세계야, 안녕" translates as "Hello, world".
		e:       EUCKR,
		encoded: "A\x81\x41\x81\x61\x81\x81\xc6\xfeB\xc7\xa1\xc7\xfe\xc8\xa1C\xca\xa1\xfd\xfeD",
		utf8:    "A\uac02\uac35\uac56\ud401B\ud408\ud620\ud624C\u4f3d\u8a70D",
	}, {
		e:       EUCKR,
		encoded: "\xbc\xbc\xb0\xe8\xbe\xdf\x2c\x20\xbe\xc8\xb3\xe7",
		utf8:    "세계야, 안녕",
	}}

	for _, tc := range testCases {
		enctest.TestEncoding(t, tc.e, tc.encoded, tc.utf8, "", "")
	}
}

func TestFiles(t *testing.T) { enctest.TestFile(t, EUCKR) }

func BenchmarkEncoding(b *testing.B) { enctest.Benchmark(b, EUCKR) }
