// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package simplifiedchinese

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
		{dec, GBK, "a\xfe\xfeb", "a\ufffdb"},
		{dec, HZGB2312, "~{z~", "\ufffd"},

		{enc, GBK, "갂", ""},
		{enc, GBK, "a갂", "a"},
		{enc, GBK, "\u4e02갂", "\x81@"},

		{enc, HZGB2312, "갂", ""},
		{enc, HZGB2312, "a갂", "a"},
		{enc, HZGB2312, "\u6cf5갂", "~{1C~}"},
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
