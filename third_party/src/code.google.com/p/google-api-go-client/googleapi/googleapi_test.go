// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package googleapi

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"testing"
)

type SetOpaqueTest struct {
	in             *url.URL
	wantRequestURI string
}

var setOpaqueTests = []SetOpaqueTest{
	// no path
	{
		&url.URL{
			Scheme: "http",
			Host:   "www.golang.org",
		},
		"http://www.golang.org",
	},
	// path
	{
		&url.URL{
			Scheme: "http",
			Host:   "www.golang.org",
			Path:   "/",
		},
		"http://www.golang.org/",
	},
	// file with hex escaping
	{
		&url.URL{
			Scheme: "https",
			Host:   "www.golang.org",
			Path:   "/file%20one&two",
		},
		"https://www.golang.org/file%20one&two",
	},
	// query
	{
		&url.URL{
			Scheme:   "http",
			Host:     "www.golang.org",
			Path:     "/",
			RawQuery: "q=go+language",
		},
		"http://www.golang.org/?q=go+language",
	},
	// file with hex escaping in path plus query
	{
		&url.URL{
			Scheme:   "https",
			Host:     "www.golang.org",
			Path:     "/file%20one&two",
			RawQuery: "q=go+language",
		},
		"https://www.golang.org/file%20one&two?q=go+language",
	},
	// query with hex escaping
	{
		&url.URL{
			Scheme:   "http",
			Host:     "www.golang.org",
			Path:     "/",
			RawQuery: "q=go%20language",
		},
		"http://www.golang.org/?q=go%20language",
	},
}

// prefixTmpl is a template for the expected prefix of the output of writing
// an HTTP request.
const prefixTmpl = "GET %v HTTP/1.1\r\nHost: %v\r\n"

func TestSetOpaque(t *testing.T) {
	for _, test := range setOpaqueTests {
		u := *test.in
		SetOpaque(&u)

		w := &bytes.Buffer{}
		r := &http.Request{URL: &u}
		if err := r.Write(w); err != nil {
			t.Errorf("write request: %v", err)
			continue
		}

		prefix := fmt.Sprintf(prefixTmpl, test.wantRequestURI, test.in.Host)
		if got := string(w.Bytes()); !strings.HasPrefix(got, prefix) {
			t.Errorf("got %q expected prefix %q", got, prefix)
		}
	}
}

type CheckResponseTest struct {
	in       *http.Response
	bodyText string
	want     error
}

var checkResponseTests = []CheckResponseTest{
	{
		&http.Response{
			StatusCode: http.StatusOK,
		},
		"",
		nil,
	},
	{
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		`{"error":{"message":"Error message for StatusNotFound."}}`,
		&Error{
			Code:    http.StatusNotFound,
			Message: "Error message for StatusNotFound.",
			Body:    `{"error":{"message":"Error message for StatusNotFound."}}`,
		},
	},
	{
		&http.Response{
			StatusCode: http.StatusBadRequest,
		},
		`{"error":"invalid_token","error_description":"Invalid Value"}`,
		&Error{
			Code: http.StatusBadRequest,
			Body: `{"error":"invalid_token","error_description":"Invalid Value"}`,
		},
	},
}

func TestCheckResponse(t *testing.T) {
	for _, test := range checkResponseTests {
		res := test.in
		if test.bodyText != "" {
			res.Body = ioutil.NopCloser(strings.NewReader(test.bodyText))
		}
		if g := CheckResponse(res); !reflect.DeepEqual(g, test.want) {
			t.Errorf("CheckResponse: got %v, want %v", g, test.want)
		}
	}
}
