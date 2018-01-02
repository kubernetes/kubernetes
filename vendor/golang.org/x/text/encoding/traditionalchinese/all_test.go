// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traditionalchinese

import (
	"fmt"
	"io/ioutil"
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
	testCases := []struct {
		init      func(e encoding.Encoding) (string, transform.Transformer, error)
		e         encoding.Encoding
		src, want string
	}{
		{dec, Big5, "\x80", "\ufffd"},
		{dec, Big5, "\x81", "\ufffd"},
		{dec, Big5, "\x81\x30", "\ufffd\x30"},
		{dec, Big5, "\x81\x40", "\ufffd"},
		{dec, Big5, "\x81\xa0", "\ufffd"},
		{dec, Big5, "\xff", "\ufffd"},

		{enc, Big5, "갂", ""},
		{enc, Big5, "a갂", "a"},
		{enc, Big5, "\u43f0갂", "\x87@"},
	}
	for _, tc := range testCases {
		dir, tr, wantErr := tc.init(tc.e)
		t.Run(fmt.Sprintf("%s/%v/%q", dir, tc.e, tc.src), func(t *testing.T) {
			dst := make([]byte, 100)
			src := []byte(tc.src)
			for i := 0; i <= len(tc.src); i++ {
				nDst, nSrc, err := tr.Transform(dst, src[:i], false)
				if err != nil && err != transform.ErrShortSrc && err != wantErr {
					t.Fatalf("error on first call to Transform: %v", err)
				}
				n, _, err := tr.Transform(dst[nDst:], src[nSrc:], true)
				nDst += n
				if err != wantErr {
					t.Fatalf("(%q|%q): got %v; want %v", tc.src[:i], tc.src[i:], err, wantErr)
				}
				if got := string(dst[:nDst]); got != tc.want {
					t.Errorf("(%q|%q):\ngot  %q\nwant %q", tc.src[:i], tc.src[i:], got, tc.want)
				}
			}
		})
	}
}

func TestBasics(t *testing.T) {
	// The encoded forms can be verified by the iconv program:
	// $ echo 月日は百代 | iconv -f UTF-8 -t SHIFT-JIS | xxd
	testCases := []struct {
		e         encoding.Encoding
		encPrefix string
		encSuffix string
		encoded   string
		utf8      string
	}{{
		e:       Big5,
		encoded: "A\x87\x40\x87\x41\x87\x45\xa1\x40\xfe\xfd\xfe\xfeZ\xa3\xe1",
		utf8:    "A\u43f0\u4c32\U00027267\u3000\U0002910d\u79d4Z€",
	}, {
		e: Big5,
		encoded: "\xaa\xe1\xb6\xa1\xa4\x40\xb3\xfd\xb0\x73\xa1\x41\xbf\x57\xb0\x75" +
			"\xb5\x4c\xac\xdb\xbf\xcb\xa1\x43",
		utf8: "花間一壺酒，獨酌無相親。",
	}}

	for _, tc := range testCases {
		enctest.TestEncoding(t, tc.e, tc.encoded, tc.utf8, "", "")
	}
}

func TestFiles(t *testing.T) { enctest.TestFile(t, Big5) }

func BenchmarkEncoding(b *testing.B) { enctest.Benchmark(b, Big5) }

// TestBig5CircumflexAndMacron tests the special cases listed in
// http://encoding.spec.whatwg.org/#big5
// Note that these special cases aren't preserved by round-tripping through
// decoding and encoding (since
// http://encoding.spec.whatwg.org/index-big5.txt does not have an entry for
// U+0304 or U+030C), so we can't test this in TestBasics.
func TestBig5CircumflexAndMacron(t *testing.T) {
	src := "\x88\x5f\x88\x60\x88\x61\x88\x62\x88\x63\x88\x64\x88\x65\x88\x66 " +
		"\x88\xa2\x88\xa3\x88\xa4\x88\xa5\x88\xa6"
	want := "ÓǑÒ\u00ca\u0304Ế\u00ca\u030cỀÊ " +
		"ü\u00ea\u0304ế\u00ea\u030cề"
	dst, err := ioutil.ReadAll(transform.NewReader(
		strings.NewReader(src), Big5.NewDecoder()))
	if err != nil {
		t.Fatal(err)
	}
	if got := string(dst); got != want {
		t.Fatalf("\ngot  %q\nwant %q", got, want)
	}
}
