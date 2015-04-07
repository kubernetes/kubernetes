// Copyright 2014 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oauth2

import "testing"

func TestTokenExtra(t *testing.T) {
	type testCase struct {
		key  string
		val  interface{}
		want interface{}
	}
	const key = "extra-key"
	cases := []testCase{
		{key: key, val: "abc", want: "abc"},
		{key: key, val: 123, want: 123},
		{key: key, val: "", want: ""},
		{key: "other-key", val: "def", want: nil},
	}
	for _, tc := range cases {
		extra := make(map[string]interface{})
		extra[tc.key] = tc.val
		tok := &Token{raw: extra}
		if got, want := tok.Extra(key), tc.want; got != want {
			t.Errorf("Extra(%q) = %q; want %q", key, got, want)
		}
	}
}
