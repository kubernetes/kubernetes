// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google

import (
	"reflect"
	"strings"
	"testing"
)

func TestSDKConfig(t *testing.T) {
	sdkConfigPath = func() (string, error) {
		return "testdata/gcloud", nil
	}

	tests := []struct {
		account     string
		accessToken string
		err         bool
	}{
		{"", "bar_access_token", false},
		{"foo@example.com", "foo_access_token", false},
		{"bar@example.com", "bar_access_token", false},
		{"baz@serviceaccount.example.com", "", true},
	}
	for _, tt := range tests {
		c, err := NewSDKConfig(tt.account)
		if got, want := err != nil, tt.err; got != want {
			if !tt.err {
				t.Errorf("got %v, want nil", err)
			} else {
				t.Errorf("got nil, want error")
			}
			continue
		}
		if err != nil {
			continue
		}
		tok := c.initialToken
		if tok == nil {
			t.Errorf("got nil, want %q", tt.accessToken)
			continue
		}
		if tok.AccessToken != tt.accessToken {
			t.Errorf("got %q, want %q", tok.AccessToken, tt.accessToken)
		}
	}
}

func TestParseINI(t *testing.T) {
	tests := []struct {
		ini  string
		want map[string]map[string]string
	}{
		{
			`root = toor
[foo]
bar = hop
ini = nin
`,
			map[string]map[string]string{
				"":    {"root": "toor"},
				"foo": {"bar": "hop", "ini": "nin"},
			},
		},
		{
			"\t  extra \t =  whitespace  \t\r\n \t [everywhere] \t \r\n  here \t =  \t there  \t \r\n",
			map[string]map[string]string{
				"":           {"extra": "whitespace"},
				"everywhere": {"here": "there"},
			},
		},
		{
			`[empty]
[section]
empty=
`,
			map[string]map[string]string{
				"":        {},
				"empty":   {},
				"section": {"empty": ""},
			},
		},
		{
			`ignore
[invalid
=stuff
;comment=true
`,
			map[string]map[string]string{
				"": {},
			},
		},
	}
	for _, tt := range tests {
		result, err := parseINI(strings.NewReader(tt.ini))
		if err != nil {
			t.Errorf("parseINI(%q) error %v, want: no error", tt.ini, err)
			continue
		}
		if !reflect.DeepEqual(result, tt.want) {
			t.Errorf("parseINI(%q) = %#v, want: %#v", tt.ini, result, tt.want)
		}
	}
}
