// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oauth2

import (
	"testing"
	"time"
)

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

func TestTokenExpiry(t *testing.T) {
	now := time.Now()
	timeNow = func() time.Time { return now }
	defer func() { timeNow = time.Now }()

	cases := []struct {
		name string
		tok  *Token
		want bool
	}{
		{name: "12 seconds", tok: &Token{Expiry: now.Add(12 * time.Second)}, want: false},
		{name: "10 seconds", tok: &Token{Expiry: now.Add(expiryDelta)}, want: false},
		{name: "10 seconds-1ns", tok: &Token{Expiry: now.Add(expiryDelta - 1*time.Nanosecond)}, want: true},
		{name: "-1 hour", tok: &Token{Expiry: now.Add(-1 * time.Hour)}, want: true},
	}
	for _, tc := range cases {
		if got, want := tc.tok.expired(), tc.want; got != want {
			t.Errorf("expired (%q) = %v; want %v", tc.name, got, want)
		}
	}
}

func TestTokenTypeMethod(t *testing.T) {
	cases := []struct {
		name string
		tok  *Token
		want string
	}{
		{name: "bearer-mixed_case", tok: &Token{TokenType: "beAREr"}, want: "Bearer"},
		{name: "default-bearer", tok: &Token{}, want: "Bearer"},
		{name: "basic", tok: &Token{TokenType: "basic"}, want: "Basic"},
		{name: "basic-capitalized", tok: &Token{TokenType: "Basic"}, want: "Basic"},
		{name: "mac", tok: &Token{TokenType: "mac"}, want: "MAC"},
		{name: "mac-caps", tok: &Token{TokenType: "MAC"}, want: "MAC"},
		{name: "mac-mixed_case", tok: &Token{TokenType: "mAc"}, want: "MAC"},
	}
	for _, tc := range cases {
		if got, want := tc.tok.Type(), tc.want; got != want {
			t.Errorf("TokenType(%q) = %v; want %v", tc.name, got, want)
		}
	}
}
