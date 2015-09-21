// Copyright 2015 The appc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package types

import (
	"encoding/json"
	"net/url"
	"reflect"
	"testing"
)

func mustParseURL(t *testing.T, s string) url.URL {
	u, err := url.Parse(s)
	if err != nil {
		t.Fatalf("error parsing URL: %v", err)
	}
	return *u
}

func TestMarshalURL(t *testing.T) {
	tests := []struct {
		u url.URL

		w string
	}{
		{
			mustParseURL(t, "http://foo.com"),

			`"http://foo.com"`,
		},
		{
			mustParseURL(t, "http://foo.com/huh/what?is=this"),

			`"http://foo.com/huh/what?is=this"`,
		},
		{
			mustParseURL(t, "https://example.com/bar"),

			`"https://example.com/bar"`,
		},
	}
	for i, tt := range tests {
		u := URL(tt.u)
		b, err := json.Marshal(u)
		if g := string(b); g != tt.w {
			t.Errorf("#%d: got %q, want %q", i, g, tt.w)
		}
		if err != nil {
			t.Errorf("#%d: err=%v, want nil", i, err)
		}

	}
}

func TestMarshalURLBad(t *testing.T) {
	tests := []url.URL{
		mustParseURL(t, "ftp://foo.com"),
		mustParseURL(t, "unix:///hello"),
	}
	for i, tt := range tests {
		u := URL(tt)
		b, err := json.Marshal(u)
		if b != nil {
			t.Errorf("#%d: got %v, want nil", i, b)
		}
		if err == nil {
			t.Errorf("#%d: got unexpected err=nil", i)
		}
	}
}

func TestUnmarshalURL(t *testing.T) {
	tests := []struct {
		in string

		w URL
	}{
		{
			`"http://foo.com"`,

			URL(mustParseURL(t, "http://foo.com")),
		},
		{
			`"http://yis.com/hello?goodbye=yes"`,

			URL(mustParseURL(t, "http://yis.com/hello?goodbye=yes")),
		},
		{
			`"https://ohai.net"`,

			URL(mustParseURL(t, "https://ohai.net")),
		},
	}
	for i, tt := range tests {
		var g URL
		err := json.Unmarshal([]byte(tt.in), &g)
		if err != nil {
			t.Errorf("#%d: want err=nil, got %v", i, err)
		}
		if !reflect.DeepEqual(g, tt.w) {
			t.Errorf("#%d: got url=%v, want %v", i, g, tt.w)
		}
	}
}

func TestUnmarshalURLBad(t *testing.T) {
	var empty = URL{}
	tests := []string{
		"badjson",
		"http://google.com",
		`"ftp://example.com"`,
		`"unix://file.net"`,
		`"not a url"`,
	}
	for i, tt := range tests {
		var g URL
		err := json.Unmarshal([]byte(tt), &g)
		if err == nil {
			t.Errorf("#%d: want err, got nil", i)
		}
		if !reflect.DeepEqual(g, empty) {
			t.Errorf("#%d: got %v, want %v", i, g, empty)
		}
	}
}
