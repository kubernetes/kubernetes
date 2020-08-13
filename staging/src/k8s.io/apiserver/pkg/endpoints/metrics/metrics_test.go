/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package metrics

import (
	"fmt"
	"net/http"
	"net/url"
	"testing"
)

func TestCleanVerb(t *testing.T) {
	testCases := []struct {
		desc         string
		initialVerb  string
		request      *http.Request
		expectedVerb string
	}{
		{
			desc:         "An empty string should be designated as unknown",
			initialVerb:  "",
			request:      nil,
			expectedVerb: "other",
		},
		{
			desc:         "LIST should normally map to LIST",
			initialVerb:  "LIST",
			request:      nil,
			expectedVerb: "LIST",
		},
		{
			desc:        "LIST should be transformed to WATCH if we have the right query param on the request",
			initialVerb: "LIST",
			request: &http.Request{
				URL: &url.URL{
					RawQuery: "watch=true",
				},
			},
			expectedVerb: "WATCH",
		},
		{
			desc:        "LIST isn't transformed to WATCH if we have query params that do not include watch",
			initialVerb: "LIST",
			request: &http.Request{
				URL: &url.URL{
					RawQuery: "blah=asdf&something=else",
				},
			},
			expectedVerb: "LIST",
		},
		{
			desc:         "WATCHLIST should be transformed to WATCH",
			initialVerb:  "WATCHLIST",
			request:      nil,
			expectedVerb: "WATCH",
		},
		{
			desc:        "PATCH should be transformed to APPLY with the right content type",
			initialVerb: "PATCH",
			request: &http.Request{
				Header: http.Header{
					"Content-Type": []string{"application/apply-patch+yaml"},
				},
			},
			expectedVerb: "APPLY",
		},
		{
			desc:         "PATCH shouldn't be transformed to APPLY without the right content type",
			initialVerb:  "PATCH",
			request:      nil,
			expectedVerb: "PATCH",
		},
		{
			desc:         "WATCHLIST should be transformed to WATCH",
			initialVerb:  "WATCHLIST",
			request:      nil,
			expectedVerb: "WATCH",
		},
		{
			desc:         "unexpected verbs should be designated as unknown",
			initialVerb:  "notValid",
			request:      nil,
			expectedVerb: "other",
		},
	}
	for _, tt := range testCases {
		t.Run(tt.initialVerb, func(t *testing.T) {
			req := &http.Request{URL: &url.URL{}}
			if tt.request != nil {
				req = tt.request
			}
			cleansedVerb := cleanVerb(tt.initialVerb, req)
			if cleansedVerb != tt.expectedVerb {
				t.Errorf("Got %s, but expected %s", cleansedVerb, tt.expectedVerb)
			}
		})
	}
}

func TestContentType(t *testing.T) {
	testCases := []struct {
		rawContentType      string
		expectedContentType string
	}{
		{
			rawContentType:      "application/json",
			expectedContentType: "application/json",
		},
		{
			rawContentType:      "image/svg+xml",
			expectedContentType: "other",
		},
		{
			rawContentType:      "text/plain; charset=utf-8",
			expectedContentType: "text/plain;charset=utf-8",
		},
		{
			rawContentType:      "application/json;foo=bar",
			expectedContentType: "other",
		},
		{
			rawContentType:      "application/json;charset=hancoding",
			expectedContentType: "other",
		},
		{
			rawContentType:      "unknownbutvalidtype",
			expectedContentType: "other",
		},
	}

	for _, tt := range testCases {
		t.Run(fmt.Sprintf("parse %s", tt.rawContentType), func(t *testing.T) {
			cleansedContentType := cleanContentType(tt.rawContentType)
			if cleansedContentType != tt.expectedContentType {
				t.Errorf("Got %s, but expected %s", cleansedContentType, tt.expectedContentType)
			}
		})
	}
}
