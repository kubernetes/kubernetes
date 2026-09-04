/*
Copyright The Kubernetes Authors.

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

package service

import (
	"context"
	"net/http"
	"net/url"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/apiserver/pkg/endpoints/request"
	api "k8s.io/kubernetes/pkg/apis/core"
)

type fakeRedirector struct {
	location  *url.URL
	transport http.RoundTripper
	err       error
}

func (f *fakeRedirector) ResourceLocation(ctx context.Context, id string) (*url.URL, http.RoundTripper, error) {
	return f.location, f.transport, f.err
}

type fakeResponder struct {
	err error
}

func (f *fakeResponder) Error(err error) {
	f.err = err
}

func (f *fakeResponder) Object(statusCode int, obj runtime.Object) {}

func TestProxyConnectRawQuery(t *testing.T) {
	tests := []struct {
		name             string
		initialRawQuery  string
		requestRawQuery  string
		expectedRawQuery string
	}{
		{
			name:             "empty initial query, has request query",
			initialRawQuery:  "",
			requestRawQuery:  "foo=bar",
			expectedRawQuery: "foo=bar",
		},
		{
			name:             "has initial query, has request query",
			initialRawQuery:  "existing=param",
			requestRawQuery:  "foo=bar",
			expectedRawQuery: "existing=param",
		},
		{
			name:             "has initial query, empty request query",
			initialRawQuery:  "existing=param",
			requestRawQuery:  "",
			expectedRawQuery: "existing=param",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			redirector := &fakeRedirector{
				location: &url.URL{
					Scheme:   "http",
					Host:     "localhost:1234",
					RawQuery: tc.initialRawQuery,
				},
			}
			r := &ProxyREST{
				Redirector: redirector,
			}

			ctx := context.Background()
			if tc.requestRawQuery != "" || tc.initialRawQuery != "" {
				info := &request.RequestInfo{
					RawQuery: tc.requestRawQuery,
				}
				ctx = request.WithRequestInfo(ctx, info)
			}

			handler, err := r.Connect(ctx, "test-service", &api.ServiceProxyOptions{}, &fakeResponder{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if handler == nil {
				t.Fatal("handler should not be nil")
			}

			upgradeHandler, ok := handler.(*proxy.UpgradeAwareHandler)
			if !ok {
				t.Fatalf("expected *proxy.UpgradeAwareHandler, got %T", handler)
			}

			if upgradeHandler.Location.RawQuery != tc.expectedRawQuery {
				t.Errorf("expected RawQuery to be %q, got %q", tc.expectedRawQuery, upgradeHandler.Location.RawQuery)
			}
		})
	}
}
