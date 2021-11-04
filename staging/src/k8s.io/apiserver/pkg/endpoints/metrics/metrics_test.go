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
	"net/http"
	"net/url"
	"testing"

	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/endpoints/responsewriter"
)

func TestCleanVerb(t *testing.T) {
	testCases := []struct {
		desc          string
		initialVerb   string
		suggestedVerb string
		request       *http.Request
		expectedVerb  string
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
			desc:        "GET isn't be transformed to WATCH if we have the right query param on the request",
			initialVerb: "GET",
			request: &http.Request{
				URL: &url.URL{
					RawQuery: "watch=true",
				},
			},
			expectedVerb: "GET",
		},
		{
			desc:          "LIST is transformed to WATCH for the old pattern watch",
			initialVerb:   "LIST",
			suggestedVerb: "WATCH",
			request: &http.Request{
				URL: &url.URL{
					RawQuery: "/api/v1/watch/pods",
				},
			},
			expectedVerb: "WATCH",
		},
		{
			desc:          "LIST is transformed to WATCH for the old pattern watchlist",
			initialVerb:   "LIST",
			suggestedVerb: "WATCHLIST",
			request: &http.Request{
				URL: &url.URL{
					RawQuery: "/api/v1/watch/pods",
				},
			},
			expectedVerb: "WATCH",
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
			cleansedVerb := cleanVerb(tt.initialVerb, tt.suggestedVerb, req)
			if cleansedVerb != tt.expectedVerb {
				t.Errorf("Got %s, but expected %s", cleansedVerb, tt.expectedVerb)
			}
		})
	}
}

func TestCleanScope(t *testing.T) {
	testCases := []struct {
		name          string
		requestInfo   *request.RequestInfo
		expectedScope string
	}{
		{
			name:          "empty scope",
			requestInfo:   &request.RequestInfo{},
			expectedScope: "",
		},
		{
			name: "resource scope",
			requestInfo: &request.RequestInfo{
				Name:              "my-resource",
				Namespace:         "my-namespace",
				IsResourceRequest: false,
			},
			expectedScope: "resource",
		},
		{
			name: "namespace scope",
			requestInfo: &request.RequestInfo{
				Namespace:         "my-namespace",
				IsResourceRequest: false,
			},
			expectedScope: "namespace",
		},
		{
			name: "cluster scope",
			requestInfo: &request.RequestInfo{
				Namespace:         "",
				IsResourceRequest: true,
			},
			expectedScope: "cluster",
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			if CleanScope(test.requestInfo) != test.expectedScope {
				t.Errorf("failed to clean scope: %v", test.requestInfo)
			}
		})
	}
}

func TestResponseWriterDecorator(t *testing.T) {
	decorator := &ResponseWriterDelegator{
		ResponseWriter: &responsewriter.FakeResponseWriter{},
	}
	var w http.ResponseWriter = decorator

	if inner := w.(responsewriter.UserProvidedDecorator).Unwrap(); inner != decorator.ResponseWriter {
		t.Errorf("Expected the decorator to return the inner http.ResponseWriter object")
	}
}
