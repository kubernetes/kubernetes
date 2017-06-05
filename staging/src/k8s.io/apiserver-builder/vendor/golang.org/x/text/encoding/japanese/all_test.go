// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package japanese

import (
	"testing"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/internal"
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
		{dec, EUCJP, "\xfe\xfc", "\ufffd"},
		{dec, ISO2022JP, "\x1b$B\x7e\x7e", "\ufffd"},
		{dec, ShiftJIS, "\xef\xfc", "\ufffd"},

		{enc, EUCJP, "갂", ""},
		{enc, EUCJP, "a갂", "a"},
		{enc, EUCJP, "丌갂", "\x8f\xb0\xa4"},

		{enc, ISO2022JP, "갂", ""},
		{enc, ISO2022JP, "a갂", "a"},
		{enc, ISO2022JP, "朗갂", "\x1b$BzF\x1b(B"}, // switch back to ASCII mode at end

		{enc, ShiftJIS, "갂", ""},
		{enc, ShiftJIS, "a갂", "a"},
		{enc, ShiftJIS, "\u2190갂", "\x81\xa9"},
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
