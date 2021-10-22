// Copyright 2018 Google Inc. All Rights Reserved.
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
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"cloud.google.com/go/internal/testutil"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/martian"
)

func TestLogger(t *testing.T) {
	req := &http.Request{
		Method: "POST",
		URL: &url.URL{
			Scheme: "https",
			Host:   "example.com",
			Path:   "a/b/c",
		},
		Header:  http.Header{"H1": {"v1", "v2"}, "Content-Type": {"text/plain"}},
		Body:    ioutil.NopCloser(strings.NewReader("hello")),
		Trailer: http.Header{"T1": {"v3", "v4"}},
	}
	res := &http.Response{
		Request:    req,
		StatusCode: 204,
		Body:       ioutil.NopCloser(strings.NewReader("goodbye")),
		Header:     http.Header{"H2": {"v5"}},
		Trailer:    http.Header{"T2": {"v6", "v7"}},
	}
	l := newLogger()
	_, remove, err := martian.TestContext(req, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer remove()
	if err := l.ModifyRequest(req); err != nil {
		t.Fatal(err)
	}
	if err := l.ModifyResponse(res); err != nil {
		t.Fatal(err)
	}
	lg := l.Extract()
	want := []*Entry{
		{
			ID: lg.Entries[0].ID,
			Request: &Request{
				Method:    "POST",
				URL:       "https://example.com/a/b/c",
				Header:    http.Header{"H1": {"v1", "v2"}},
				MediaType: "text/plain",
				BodyParts: [][]byte{[]byte("hello")},
				Trailer:   http.Header{"T1": {"v3", "v4"}},
			},
			Response: &Response{
				StatusCode: 204,
				Body:       []byte("goodbye"),
				Header:     http.Header{"H2": {"v5"}},
				Trailer:    http.Header{"T2": {"v6", "v7"}},
			},
		},
	}
	if diff := testutil.Diff(lg.Entries, want); diff != "" {
		t.Error(diff)
	}
}

func TestToHTTPResponse(t *testing.T) {
	for _, test := range []struct {
		desc string
		lr   *Response
		req  *http.Request
		want *http.Response
	}{
		{
			desc: "GET request",
			lr: &Response{
				StatusCode: 201,
				Proto:      "1.1",
				Header:     http.Header{"h": {"v"}},
				Body:       []byte("text"),
			},
			req: &http.Request{Method: "GET"},
			want: &http.Response{
				Request:       &http.Request{Method: "GET"},
				StatusCode:    201,
				Proto:         "1.1",
				Header:        http.Header{"h": {"v"}},
				ContentLength: 4,
			},
		},
		{
			desc: "HEAD request with no Content-Length header",
			lr: &Response{
				StatusCode: 201,
				Proto:      "1.1",
				Header:     http.Header{"h": {"v"}},
				Body:       []byte("text"),
			},
			req: &http.Request{Method: "HEAD"},
			want: &http.Response{
				Request:       &http.Request{Method: "HEAD"},
				StatusCode:    201,
				Proto:         "1.1",
				Header:        http.Header{"h": {"v"}},
				ContentLength: -1,
			},
		},
		{
			desc: "HEAD request with Content-Length header",
			lr: &Response{
				StatusCode: 201,
				Proto:      "1.1",
				Header:     http.Header{"h": {"v"}, "Content-Length": {"17"}},
				Body:       []byte("text"),
			},
			req: &http.Request{Method: "HEAD"},
			want: &http.Response{
				Request:       &http.Request{Method: "HEAD"},
				StatusCode:    201,
				Proto:         "1.1",
				Header:        http.Header{"h": {"v"}, "Content-Length": {"17"}},
				ContentLength: 17,
			},
		},
	} {
		got := toHTTPResponse(test.lr, test.req)
		got.Body = nil
		if diff := testutil.Diff(got, test.want, cmpopts.IgnoreUnexported(http.Request{})); diff != "" {
			t.Errorf("%s: %s", test.desc, diff)
		}
	}
}

func TestEmptyBody(t *testing.T) {
	// Verify that a zero-length body is nil after logging.
	// That will ensure that net/http sends a "Content-Length: 0" header.
	req := &http.Request{
		Method: "POST",
		URL: &url.URL{
			Scheme: "https",
			Host:   "example.com",
			Path:   "a/b/c",
		},
		Body: ioutil.NopCloser(strings.NewReader("")),
	}
	l := newLogger()
	_, remove, err := martian.TestContext(req, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer remove()
	if err := l.ModifyRequest(req); err != nil {
		t.Fatal(err)
	}
	if req.Body != nil {
		t.Error("got non-nil req.Body, want nil")
	}
}
