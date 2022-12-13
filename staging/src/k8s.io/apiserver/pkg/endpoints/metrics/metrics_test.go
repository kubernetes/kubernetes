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
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/endpoints/responsewriter"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
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
				Method: "GET",
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
				Method: "GET",
				URL: &url.URL{
					RawQuery: "blah=asdf&something=else",
				},
			},
			expectedVerb: "LIST",
		},
		{
			// The above may seem counter-intuitive, but it actually is needed for cases like
			// watching a single item, e.g.:
			//  /api/v1/namespaces/foo/pods/bar?fieldSelector=metadata.name=baz&watch=true
			desc:        "GET is transformed to WATCH if we have the right query param on the request",
			initialVerb: "GET",
			request: &http.Request{
				Method: "GET",
				URL: &url.URL{
					RawQuery: "watch=true",
				},
			},
			expectedVerb: "WATCH",
		},
		{
			desc:          "LIST is transformed to WATCH for the old pattern watch",
			initialVerb:   "LIST",
			suggestedVerb: "WATCH",
			request: &http.Request{
				Method: "GET",
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
				Method: "GET",
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
			name: "POST resource scope",
			requestInfo: &request.RequestInfo{
				Verb:              "create",
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

func TestCleanFieldValidation(t *testing.T) {
	testCases := []struct {
		name                    string
		url                     *url.URL
		expectedFieldValidation string
	}{
		{
			name:                    "empty field validation",
			url:                     &url.URL{},
			expectedFieldValidation: "",
		},
		{
			name: "ignore field validation",
			url: &url.URL{
				RawQuery: "fieldValidation=Ignore",
			},
			expectedFieldValidation: "Ignore",
		},
		{
			name: "warn field validation",
			url: &url.URL{
				RawQuery: "fieldValidation=Warn",
			},
			expectedFieldValidation: "Warn",
		},
		{
			name: "strict field validation",
			url: &url.URL{
				RawQuery: "fieldValidation=Strict",
			},
			expectedFieldValidation: "Strict",
		},
		{
			name: "invalid field validation",
			url: &url.URL{
				RawQuery: "fieldValidation=foo",
			},
			expectedFieldValidation: "invalid",
		},
		{
			name: "multiple field validation",
			url: &url.URL{
				RawQuery: "fieldValidation=Strict&fieldValidation=Ignore",
			},
			expectedFieldValidation: "invalid",
		},
	}
	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			if fieldValidation := cleanFieldValidation(test.url); fieldValidation != test.expectedFieldValidation {
				t.Errorf("failed to clean field validation, expected: %s, got: %s", test.expectedFieldValidation, fieldValidation)
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

func TestRecordDroppedRequests(t *testing.T) {
	testedMetrics := []string{
		"apiserver_request_total",
	}

	testCases := []struct {
		desc        string
		request     *http.Request
		requestInfo *request.RequestInfo
		isMutating  bool
		want        string
	}{
		{
			desc: "list pods",
			request: &http.Request{
				Method: "GET",
				URL: &url.URL{
					RawPath: "/api/v1/pods",
				},
			},
			requestInfo: &request.RequestInfo{
				Verb:              "list",
				APIGroup:          "",
				APIVersion:        "v1",
				Resource:          "pods",
				IsResourceRequest: true,
			},
			isMutating: false,
			want: `
			            # HELP apiserver_request_total [STABLE] Counter of apiserver requests broken out for each verb, dry run value, group, version, resource, scope, component, and HTTP response code.
			            # TYPE apiserver_request_total counter
			            apiserver_request_total{code="429",component="apiserver",dry_run="",group="",resource="pods",scope="cluster",subresource="",verb="LIST",version="v1"} 1
				`,
		},
		{
			desc: "post pods",
			request: &http.Request{
				Method: "POST",
				URL: &url.URL{
					RawPath: "/api/v1/namespaces/foo/pods",
				},
			},
			requestInfo: &request.RequestInfo{
				Verb:              "create",
				APIGroup:          "",
				APIVersion:        "v1",
				Resource:          "pods",
				IsResourceRequest: true,
			},
			isMutating: true,
			want: `
			            # HELP apiserver_request_total [STABLE] Counter of apiserver requests broken out for each verb, dry run value, group, version, resource, scope, component, and HTTP response code.
			            # TYPE apiserver_request_total counter
			            apiserver_request_total{code="429",component="apiserver",dry_run="",group="",resource="pods",scope="resource",subresource="",verb="POST",version="v1"} 1
				`,
		},
		{
			desc: "dry-run patch job status",
			request: &http.Request{
				Method: "PATCH",
				URL: &url.URL{
					RawPath:  "/apis/batch/v1/namespaces/foo/jobs/bar/status",
					RawQuery: "dryRun=All",
				},
			},
			requestInfo: &request.RequestInfo{
				Verb:              "patch",
				APIGroup:          "batch",
				APIVersion:        "v1",
				Resource:          "jobs",
				Name:              "bar",
				Subresource:       "status",
				IsResourceRequest: true,
			},
			isMutating: true,
			want: `
			            # HELP apiserver_request_total [STABLE] Counter of apiserver requests broken out for each verb, dry run value, group, version, resource, scope, component, and HTTP response code.
			            # TYPE apiserver_request_total counter
			            apiserver_request_total{code="429",component="apiserver",dry_run="All",group="batch",resource="jobs",scope="resource",subresource="status",verb="PATCH",version="v1"} 1
				`,
		},
	}

	// Since prometheus' gatherer is global, other tests may have updated metrics already, so
	// we need to reset them prior running this test.
	// This also implies that we can't run this test in parallel with other tests.
	Register()
	requestCounter.Reset()

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			defer requestCounter.Reset()

			RecordDroppedRequest(test.request, test.requestInfo, APIServerComponent, test.isMutating)

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(test.want), testedMetrics...); err != nil {
				t.Fatal(err)
			}

		})
	}
}
