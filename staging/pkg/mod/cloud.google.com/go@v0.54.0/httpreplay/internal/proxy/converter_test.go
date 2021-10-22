// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package proxy

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"net/url"
	"testing"

	"cloud.google.com/go/internal/testutil"
	"github.com/google/go-cmp/cmp"
)

func TestConvertRequest(t *testing.T) {
	clone := func(h http.Header) http.Header {
		h2 := http.Header{}
		for k, v := range h {
			h2[k] = v
		}
		return h2
	}

	body := []byte("hello")

	conv := defaultConverter()
	conv.registerClearParams("secret")
	conv.registerRemoveParams("rm*")
	url, err := url.Parse("https://www.example.com?a=1&rmx=x&secret=2&c=3&rmy=4")
	if err != nil {
		t.Fatal(err)
	}
	in := &http.Request{
		Method: "GET",
		URL:    url,
		Body:   ioutil.NopCloser(bytes.NewReader(body)),
		Header: http.Header{
			"Content-Type":                      {"text/plain"},
			"Authorization":                     {"oauth2-token"},
			"X-Goog-Encryption-Key":             {"a-secret-key"},
			"X-Goog-Copy-Source-Encryption-Key": {"another-secret-key"},
		},
	}
	origHeader := clone(in.Header)

	got, err := conv.convertRequest(in)
	if err != nil {
		t.Fatal(err)
	}
	want := &Request{
		Method:    "GET",
		URL:       "https://www.example.com?a=1&secret=CLEARED&c=3",
		MediaType: "text/plain",
		BodyParts: [][]byte{body},
		Header: http.Header{
			"X-Goog-Encryption-Key":             {"CLEARED"},
			"X-Goog-Copy-Source-Encryption-Key": {"CLEARED"},
		},
		Trailer: http.Header{},
	}
	if diff := cmp.Diff(got, want); diff != "" {
		t.Error(diff)
	}
	// The original headers should be the same.
	if got, want := in.Header, origHeader; !testutil.Equal(got, want) {
		t.Errorf("got  %+v\nwant %+v", got, want)
	}
}

func TestPattern(t *testing.T) {
	for _, test := range []struct {
		in, want string
	}{
		{"", "^$"},
		{"abc", "^abc$"},
		{"*ab*", "^.*ab.*$"},
		{`a\*b`, `^a\\.*b$`},
		{"***", "^.*.*.*$"},
	} {
		got := pattern(test.in).String()
		if got != test.want {
			t.Errorf("%q: got %s, want %s", test.in, got, test.want)
		}
	}
}

func TestScrubQuery(t *testing.T) {
	clear := []tRegexp{pattern("c*")}
	remove := []tRegexp{pattern("r*")}
	for _, test := range []struct {
		in, want string
	}{
		{"", ""},
		{"a=1", "a=1"},
		{"a=1&b=2;g=3", "a=1&b=2;g=3"},
		{"a=1&r=2;c=3", "a=1&c=CLEARED"},
		{"ra=1&rb=2&rc=3", ""},
		{"a=1&%Z=2&r=3&c=4", "a=1&%Z=2&c=CLEARED"},
	} {
		got := scrubQuery(test.in, clear, remove)
		if got != test.want {
			t.Errorf("%s: got %q, want %q", test.in, got, test.want)
		}
	}
}
