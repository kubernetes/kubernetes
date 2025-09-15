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
	"context"
	"net/http"
	"net/url"
	"strings"
	"testing"

	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	"k8s.io/apimachinery/pkg/fields"
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
		requestInfo   *request.RequestInfo
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
			initialVerb:  request.MethodList,
			request:      nil,
			expectedVerb: request.MethodList,
		},
		{
			desc:        "LIST should be transformed to WATCH if we have the right query param on the request",
			initialVerb: request.MethodList,
			request: &http.Request{
				Method: request.MethodGet,
				URL: &url.URL{
					RawQuery: "watch=true",
				},
			},
			expectedVerb: request.MethodWatch,
		},
		{
			desc:        "LIST isn't transformed to WATCH if we have query params that do not include watch",
			initialVerb: request.MethodList,
			request: &http.Request{
				Method: request.MethodGet,
				URL: &url.URL{
					RawQuery: "blah=asdf&something=else",
				},
			},
			expectedVerb: request.MethodList,
		},
		{
			// The above may seem counter-intuitive, but it actually is needed for cases like
			// watching a single item, e.g.:
			//  /api/v1/namespaces/foo/pods/bar?fieldSelector=metadata.name=baz&watch=true
			desc:        "GET is transformed to WATCH if we have the right query param on the request",
			initialVerb: request.MethodGet,
			request: &http.Request{
				Method: request.MethodGet,
				URL: &url.URL{
					RawQuery: "watch=true",
				},
			},
			expectedVerb: request.MethodWatch,
		},
		{
			desc:          "LIST is transformed to WATCH for the old pattern watch",
			initialVerb:   request.MethodList,
			suggestedVerb: request.MethodWatch,
			request: &http.Request{
				Method: request.MethodGet,
				URL: &url.URL{
					RawQuery: "/api/v1/watch/pods",
				},
			},
			expectedVerb: request.MethodWatch,
		},
		{
			desc:          "LIST is transformed to WATCH for the old pattern watchlist",
			initialVerb:   request.MethodList,
			suggestedVerb: request.MethodWatchList,
			request: &http.Request{
				Method: request.MethodGet,
				URL: &url.URL{
					RawQuery: "/api/v1/watch/pods",
				},
			},
			expectedVerb: request.MethodWatch,
		},
		{
			desc:         "WATCHLIST should be transformed to WATCH",
			initialVerb:  request.MethodWatchList,
			request:      nil,
			expectedVerb: request.MethodWatch,
		},
		{
			desc:        "PATCH should be transformed to APPLY with the right content type",
			initialVerb: request.MethodPatch,
			request: &http.Request{
				Header: http.Header{
					"Content-Type": []string{"application/apply-patch+yaml"},
				},
			},
			expectedVerb: request.MethodApply,
		},
		{
			desc:         "PATCH shouldn't be transformed to APPLY without the right content type",
			initialVerb:  request.MethodPatch,
			request:      nil,
			expectedVerb: request.MethodPatch,
		},
		{
			desc:         "WATCHLIST should be transformed to WATCH",
			initialVerb:  request.MethodWatchList,
			request:      nil,
			expectedVerb: request.MethodWatch,
		},
		{
			desc:         "unexpected verbs should be designated as unknown",
			initialVerb:  "notValid",
			request:      nil,
			expectedVerb: "other",
		},
		{
			desc:        "Pod logs should be transformed to CONNECT",
			initialVerb: request.MethodGet,
			request: &http.Request{
				Method: request.MethodGet,
				URL: &url.URL{
					RawQuery: "/api/v1/namespaces/default/pods/test-pod/log",
				},
			},
			requestInfo: &request.RequestInfo{
				Verb:              request.MethodGet,
				Resource:          "pods",
				IsResourceRequest: true,
				Subresource:       "log",
			},
			expectedVerb: request.MethodConnect,
		},
		{
			desc:        "Pod exec should be transformed to CONNECT",
			initialVerb: request.MethodPost,
			request: &http.Request{
				Method: request.MethodPost,
				URL: &url.URL{
					RawQuery: "/api/v1/namespaces/default/pods/test-pod/exec?command=sh",
				},
				Header: map[string][]string{
					"Connection": {"Upgrade"},
					"Upgrade":    {"SPDY/3.1"},
					"X-Stream-Protocol-Version": {
						"v4.channel.k8s.io", "v3.channel.k8s.io", "v2.channel.k8s.io", "channel.k8s.io",
					},
				},
			},
			requestInfo: &request.RequestInfo{
				Verb:              request.MethodPost,
				Resource:          "pods",
				IsResourceRequest: true,
				Subresource:       "exec",
			},
			expectedVerb: request.MethodConnect,
		},
		{
			desc:        "Pod portforward should be transformed to CONNECT",
			initialVerb: request.MethodPost,
			request: &http.Request{
				Method: request.MethodPost,
				URL: &url.URL{
					RawQuery: "/api/v1/namespaces/default/pods/test-pod/portforward",
				},
				Header: map[string][]string{
					"Connection": {"Upgrade"},
					"Upgrade":    {"SPDY/3.1"},
					"X-Stream-Protocol-Version": {
						"v4.channel.k8s.io", "v3.channel.k8s.io", "v2.channel.k8s.io", "channel.k8s.io",
					},
				},
			},
			requestInfo: &request.RequestInfo{
				Verb:              request.MethodPost,
				Resource:          "pods",
				IsResourceRequest: true,
				Subresource:       "portforward",
			},
			expectedVerb: request.MethodConnect,
		},
		{
			desc:        "Deployment scale should not be transformed to CONNECT",
			initialVerb: request.MethodPut,
			request: &http.Request{
				Method: request.MethodPut,
				URL: &url.URL{
					RawQuery: "/apis/apps/v1/namespaces/default/deployments/test-1/scale",
				},
				Header: map[string][]string{},
			},
			requestInfo: &request.RequestInfo{
				Verb:              request.MethodPut,
				Resource:          "deployments",
				IsResourceRequest: true,
				Subresource:       "scale",
			},
			expectedVerb: request.MethodPut,
		},
	}
	for _, tt := range testCases {
		t.Run(tt.initialVerb, func(t *testing.T) {
			req := &http.Request{URL: &url.URL{}}
			if tt.request != nil {
				req = tt.request
			}
			cleansedVerb := cleanVerb(tt.initialVerb, tt.suggestedVerb, req, tt.requestInfo)
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
				Method: request.MethodGet,
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
				Method: request.MethodPost,
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
				Method: request.MethodPatch,
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

func TestCleanListScope(t *testing.T) {
	scenarios := []struct {
		name          string
		ctx           context.Context
		opts          *metainternalversion.ListOptions
		expectedScope string
	}{
		{
			name: "empty scope",
		},
		{
			name: "empty scope with empty request info",
			ctx:  request.WithRequestInfo(context.TODO(), &request.RequestInfo{}),
		},
		{
			name:          "namespace from ctx",
			ctx:           request.WithNamespace(context.TODO(), "foo"),
			expectedScope: "namespace",
		},
		{
			name: "namespace from field selector",
			opts: &metainternalversion.ListOptions{
				FieldSelector: fields.ParseSelectorOrDie("metadata.namespace=foo"),
			},
			expectedScope: "namespace",
		},
		{
			name:          "name from request info",
			ctx:           request.WithRequestInfo(context.TODO(), &request.RequestInfo{Name: "bar"}),
			expectedScope: "resource",
		},
		{
			name: "name from field selector",
			opts: &metainternalversion.ListOptions{
				FieldSelector: fields.ParseSelectorOrDie("metadata.name=bar"),
			},
			expectedScope: "resource",
		},
		{
			name:          "cluster scope request",
			ctx:           request.WithRequestInfo(context.TODO(), &request.RequestInfo{IsResourceRequest: true}),
			expectedScope: "cluster",
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			if scenario.ctx == nil {
				scenario.ctx = context.TODO()
			}
			actualScope := CleanListScope(scenario.ctx, scenario.opts)
			if actualScope != scenario.expectedScope {
				t.Errorf("unexpected scope = %s, expected = %s", actualScope, scenario.expectedScope)
			}
		})
	}
}
