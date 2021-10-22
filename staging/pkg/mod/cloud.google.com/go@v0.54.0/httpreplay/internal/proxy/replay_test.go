// Copyright 2018 Google LLC
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
	"net/http"
	"testing"

	"cloud.google.com/go/internal/testutil"
)

func TestParseRequestBody(t *testing.T) {
	wantMediaType := "multipart/mixed"
	wantParts := [][]byte{
		[]byte("A section"),
		[]byte("And another"),
	}
	for i, test := range []struct {
		contentType, body string
	}{
		{
			wantMediaType + "; boundary=foo",
			"--foo\r\nFoo: one\r\n\r\nA section\r\n" +
				"--foo\r\nFoo: two\r\n\r\nAnd another\r\n" +
				"--foo--\r\n",
		},
		// Same contents, different boundary.
		{
			wantMediaType + "; boundary=bar",
			"--bar\r\nFoo: one\r\n\r\nA section\r\n" +
				"--bar\r\nFoo: two\r\n\r\nAnd another\r\n" +
				"--bar--\r\n",
		},
	} {
		gotMediaType, gotParts, err := parseRequestBody(test.contentType, []byte(test.body))
		if err != nil {
			t.Fatalf("#%d: %v", i, err)
		}
		if gotMediaType != wantMediaType {
			t.Errorf("#%d: got %q, want %q", i, gotMediaType, wantMediaType)
		}
		if diff := testutil.Diff(gotParts, wantParts); diff != "" {
			t.Errorf("#%d: %s", i, diff)
		}
	}
}

func TestHeadersMatch(t *testing.T) {
	for _, test := range []struct {
		h1, h2 http.Header
		want   bool
	}{
		{
			http.Header{"A": {"x"}, "B": {"y", "z"}},
			http.Header{"A": {"x"}, "B": {"y", "z"}},
			true,
		},
		{
			http.Header{"A": {"x"}, "B": {"y", "z"}},
			http.Header{"A": {"x"}, "B": {"w"}},
			false,
		},
		{
			http.Header{"A": {"x"}, "B": {"y", "z"}, "I": {"foo"}},
			http.Header{"A": {"x"}, "B": {"y", "z"}, "I": {"bar"}},
			true,
		},
		{
			http.Header{"A": {"x"}, "B": {"y", "z"}},
			http.Header{"A": {"x"}, "B": {"y", "z"}, "I": {"bar"}},
			true,
		},
		{
			http.Header{"A": {"x"}, "B": {"y", "z"}, "I": {"foo"}},
			http.Header{"A": {"x"}, "I": {"bar"}},
			false,
		},
		{
			http.Header{"A": {"x"}, "I": {"foo"}},
			http.Header{"A": {"x"}, "B": {"y", "z"}, "I": {"bar"}},
			false,
		},
	} {
		got := headersMatch(test.h1, test.h2, map[string]bool{"I": true})
		if got != test.want {
			t.Errorf("%v, %v: got %t, want %t", test.h1, test.h2, got, test.want)
		}
	}
}
