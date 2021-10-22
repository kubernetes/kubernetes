// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package japanese

import (
	"fmt"
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
	// Pick n to cause the destination buffer in transform.String to overflow.
	const n = 100
	long := strings.Repeat(".", n)
	testCases := []struct {
		init      func(e encoding.Encoding) (string, transform.Transformer, error)
		e         encoding.Encoding
		src, want string
	}{
		{enc, EUCJP, "갂", ""},
		{enc, EUCJP, "a갂", "a"},
		{enc, EUCJP, "丌갂", "\x8f\xb0\xa4"},

		{enc, ISO2022JP, "갂", ""},
		{enc, ISO2022JP, "a갂", "a"},
		{enc, ISO2022JP, "朗갂", "\x1b$BzF\x1b(B"}, // switch back to ASCII mode at end

		{enc, ShiftJIS, "갂", ""},
		{enc, ShiftJIS, "a갂", "a"},
		{enc, ShiftJIS, "\u2190갂", "\x81\xa9"},

		// Continue correctly after errors
		{dec, EUCJP, "\x8e\xa0", "\ufffd\ufffd"},
		{dec, EUCJP, "\x8e\xe0", "\ufffd"},
		{dec, EUCJP, "\x8e\xff", "\ufffd\ufffd"},
		{dec, EUCJP, "\x8ea", "\ufffda"},
		{dec, EUCJP, "\x8f\xa0", "\ufffd\ufffd"},
		{dec, EUCJP, "\x8f\xa1\xa0", "\ufffd\ufffd"},
		{dec, EUCJP, "\x8f\xa1a", "\ufffda"},
		{dec, EUCJP, "\x8f\xa1a", "\ufffda"},
		{dec, EUCJP, "\x8f\xa1a", "\ufffda"},
		{dec, EUCJP, "\x8f\xa2\xa2", "\ufffd"},
		{dec, EUCJP, "\xfe", "\ufffd"},
		{dec, EUCJP, "\xfe\xfc", "\ufffd"},
		{dec, EUCJP, "\xfe\xff", "\ufffd\ufffd"},
		// Correct handling of end of source
		{dec, EUCJP, strings.Repeat("\x8e", n), strings.Repeat("\ufffd", n)},
		{dec, EUCJP, strings.Repeat("\x8f", n), strings.Repeat("\ufffd", n)},
		{dec, EUCJP, strings.Repeat("\x8f\xa0", n), strings.Repeat("\ufffd", 2*n)},
		{dec, EUCJP, "a" + strings.Repeat("\x8f\xa1", n), "a" + strings.Repeat("\ufffd", n)},
		{dec, EUCJP, "a" + strings.Repeat("\x8f\xa1\xff", n), "a" + strings.Repeat("\ufffd", 2*n)},

		// Continue correctly after errors
		{dec, ShiftJIS, "\x80", "\u0080"}, // It's what the spec says.
		{dec, ShiftJIS, "\x81", "\ufffd"},
		{dec, ShiftJIS, "\x81\x7f", "\ufffd\u007f"},
		{dec, ShiftJIS, "\xe0", "\ufffd"},
		{dec, ShiftJIS, "\xe0\x39", "\ufffd\u0039"},
		{dec, ShiftJIS, "\xe0\x9f", "燹"},
		{dec, ShiftJIS, "\xe0\xfd", "\ufffd"},
		{dec, ShiftJIS, "\xef\xfc", "\ufffd"},
		{dec, ShiftJIS, "\xfc\xfc", "\ufffd"},
		{dec, ShiftJIS, "\xfc\xfd", "\ufffd"},
		{dec, ShiftJIS, "\xfdaa", "\ufffdaa"},

		{dec, ShiftJIS, strings.Repeat("\x81\x81", n), strings.Repeat("＝", n)},
		{dec, ShiftJIS, strings.Repeat("\xe0\xfd", n), strings.Repeat("\ufffd", n)},
		{dec, ShiftJIS, "a" + strings.Repeat("\xe0\xfd", n), "a" + strings.Repeat("\ufffd", n)},

		{dec, ISO2022JP, "\x1b$", "\ufffd$"},
		{dec, ISO2022JP, "\x1b(", "\ufffd("},
		{dec, ISO2022JP, "\x1b@", "\ufffd@"},
		{dec, ISO2022JP, "\x1bZ", "\ufffdZ"},
		// incomplete escapes
		{dec, ISO2022JP, "\x1b$", "\ufffd$"},
		{dec, ISO2022JP, "\x1b$J.", "\ufffd$J."},             // illegal
		{dec, ISO2022JP, "\x1b$B.", "\ufffd"},                // JIS208
		{dec, ISO2022JP, "\x1b$(", "\ufffd$("},               // JIS212
		{dec, ISO2022JP, "\x1b$(..", "\ufffd$(.."},           // JIS212
		{dec, ISO2022JP, "\x1b$(" + long, "\ufffd$(" + long}, // JIS212
		{dec, ISO2022JP, "\x1b$(D.", "\ufffd"},               // JIS212
		{dec, ISO2022JP, "\x1b$(D..", "\ufffd"},              // JIS212
		{dec, ISO2022JP, "\x1b$(D...", "\ufffd\ufffd"},       // JIS212
		{dec, ISO2022JP, "\x1b(B.", "."},                     // ascii
		{dec, ISO2022JP, "\x1b(B..", ".."},                   // ascii
		{dec, ISO2022JP, "\x1b(J.", "."},                     // roman
		{dec, ISO2022JP, "\x1b(J..", ".."},                   // roman
		{dec, ISO2022JP, "\x1b(I\x20", "\ufffd"},             // katakana
		{dec, ISO2022JP, "\x1b(I\x20\x20", "\ufffd\ufffd"},   // katakana
		// recover to same state
		{dec, ISO2022JP, "\x1b(B\x1b.", "\ufffd."},
		{dec, ISO2022JP, "\x1b(I\x1b.", "\ufffdｮ"},
		{dec, ISO2022JP, "\x1b(I\x1b$.", "\ufffd､ｮ"},
		{dec, ISO2022JP, "\x1b(I\x1b(.", "\ufffdｨｮ"},
		{dec, ISO2022JP, "\x1b$B\x7e\x7e", "\ufffd"},
		{dec, ISO2022JP, "\x1b$@\x0a.", "\x0a."},
		{dec, ISO2022JP, "\x1b$B\x0a.", "\x0a."},
		{dec, ISO2022JP, "\x1b$(D\x0a.", "\x0a."},
		{dec, ISO2022JP, "\x1b$(D\x7e\x7e", "\ufffd"},
		{dec, ISO2022JP, "\x80", "\ufffd"},

		// TODO: according to https://encoding.spec.whatwg.org/#iso-2022-jp,
		// these should all be correct.
		// {dec, ISO2022JP, "\x1b(B\x0E", "\ufffd"},
		// {dec, ISO2022JP, "\x1b(B\x0F", "\ufffd"},
		{dec, ISO2022JP, "\x1b(B\x5C", "\u005C"},
		{dec, ISO2022JP, "\x1b(B\x7E", "\u007E"},
		// {dec, ISO2022JP, "\x1b(J\x0E", "\ufffd"},
		// {dec, ISO2022JP, "\x1b(J\x0F", "\ufffd"},
		// {dec, ISO2022JP, "\x1b(J\x5C", "\u00A5"},
		// {dec, ISO2022JP, "\x1b(J\x7E", "\u203E"},
	}
	for _, tc := range testCases {
		dir, tr, wantErr := tc.init(tc.e)
		t.Run(fmt.Sprintf("%s/%v/%q", dir, tc.e, tc.src), func(t *testing.T) {
			dst := make([]byte, 100000)
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

func TestCorrect(t *testing.T) {
	testCases := []struct {
		init      func(e encoding.Encoding) (string, transform.Transformer, error)
		e         encoding.Encoding
		src, want string
	}{
		{dec, ShiftJIS, "\x9f\xfc", "滌"},
		{dec, ShiftJIS, "\xfb\xfc", "髙"},
		{dec, ShiftJIS, "\xfa\xb1", "﨑"},
		{enc, ShiftJIS, "滌", "\x9f\xfc"},
		{enc, ShiftJIS, "﨑", "\xed\x95"},
	}
	for _, tc := range testCases {
		dir, tr, _ := tc.init(tc.e)

		dst, _, err := transform.String(tr, tc.src)
		if err != nil {
			t.Errorf("%s %v(%q): got %v; want %v", dir, tc.e, tc.src, err, nil)
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
		e         encoding.Encoding
		encPrefix string
		encSuffix string
		encoded   string
		utf8      string
	}{{
		// "A｡ｶﾟ 0208: etc 0212: etc" is a nonsense string that contains ASCII, half-width
		// kana, JIS X 0208 (including two near the kink in the Shift JIS second byte
		// encoding) and JIS X 0212 encodable codepoints.
		//
		// "月日は百代の過客にして、行かふ年も又旅人也。" is from the 17th century poem
		// "Oku no Hosomichi" and contains both hiragana and kanji.
		e: EUCJP,
		encoded: "A\x8e\xa1\x8e\xb6\x8e\xdf " +
			"0208: \xa1\xa1\xa1\xa2\xa1\xdf\xa1\xe0\xa1\xfd\xa1\xfe\xa2\xa1\xa2\xa2\xf4\xa6 " +
			"0212: \x8f\xa2\xaf\x8f\xed\xe3",
		utf8: "A｡ｶﾟ " +
			"0208: \u3000\u3001\u00d7\u00f7\u25ce\u25c7\u25c6\u25a1\u7199 " +
			"0212: \u02d8\u9fa5",
	}, {
		e: EUCJP,
		encoded: "\xb7\xee\xc6\xfc\xa4\xcf\xc9\xb4\xc2\xe5\xa4\xce\xb2\xe1\xb5\xd2" +
			"\xa4\xcb\xa4\xb7\xa4\xc6\xa1\xa2\xb9\xd4\xa4\xab\xa4\xd5\xc7\xaf" +
			"\xa4\xe2\xcb\xf4\xce\xb9\xbf\xcd\xcc\xe9\xa1\xa3",
		utf8: "月日は百代の過客にして、行かふ年も又旅人也。",
	}, {
		e:         ISO2022JP,
		encSuffix: "\x1b\x28\x42",
		encoded: "\x1b\x28\x49\x21\x36\x5f\x1b\x28\x42 " +
			"0208: \x1b\x24\x42\x21\x21\x21\x22\x21\x5f\x21\x60\x21\x7d\x21\x7e\x22\x21\x22\x22\x74\x26",
		utf8: "｡ｶﾟ " +
			"0208: \u3000\u3001\u00d7\u00f7\u25ce\u25c7\u25c6\u25a1\u7199",
	}, {
		e:         ISO2022JP,
		encPrefix: "\x1b\x24\x42",
		encSuffix: "\x1b\x28\x42",
		encoded: "\x37\x6e\x46\x7c\x24\x4f\x49\x34\x42\x65\x24\x4e\x32\x61\x35\x52" +
			"\x24\x4b\x24\x37\x24\x46\x21\x22\x39\x54\x24\x2b\x24\x55\x47\x2f" +
			"\x24\x62\x4b\x74\x4e\x39\x3f\x4d\x4c\x69\x21\x23",
		utf8: "月日は百代の過客にして、行かふ年も又旅人也。",
	}, {
		e: ShiftJIS,
		encoded: "A\xa1\xb6\xdf " +
			"0208: \x81\x40\x81\x41\x81\x7e\x81\x80\x81\x9d\x81\x9e\x81\x9f\x81\xa0\xea\xa4",
		utf8: "A｡ｶﾟ " +
			"0208: \u3000\u3001\u00d7\u00f7\u25ce\u25c7\u25c6\u25a1\u7199",
	}, {
		e: ShiftJIS,
		encoded: "\x8c\x8e\x93\xfa\x82\xcd\x95\x53\x91\xe3\x82\xcc\x89\xdf\x8b\x71" +
			"\x82\xc9\x82\xb5\x82\xc4\x81\x41\x8d\x73\x82\xa9\x82\xd3\x94\x4e" +
			"\x82\xe0\x96\x94\x97\xb7\x90\x6c\x96\xe7\x81\x42",
		utf8: "月日は百代の過客にして、行かふ年も又旅人也。",
	}}

	for _, tc := range testCases {
		enctest.TestEncoding(t, tc.e, tc.encoded, tc.utf8, tc.encPrefix, tc.encSuffix)
	}
}

func TestFiles(t *testing.T) {
	enctest.TestFile(t, EUCJP)
	enctest.TestFile(t, ISO2022JP)
	enctest.TestFile(t, ShiftJIS)
}

func BenchmarkEncoding(b *testing.B) {
	enctest.Benchmark(b, EUCJP)
	enctest.Benchmark(b, ISO2022JP)
	enctest.Benchmark(b, ShiftJIS)
}
