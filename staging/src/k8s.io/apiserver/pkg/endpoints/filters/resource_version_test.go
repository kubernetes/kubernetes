/*
Copyright 2021 The Kubernetes Authors.

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

package filters

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/endpoints/request"
)

func TestHasInvalidResourceVersion(t *testing.T) {
	tests := []struct {
		name       string
		requestURI string
		expected   bool
	}{
		{
			name:       "the user does not specify a resource version",
			requestURI: "/api/v1/namespaces?resourceVersion=",
			expected:   true,
		},
		{
			name:       "the user specifies a valid resource version",
			requestURI: "/api/v1/namespaces?resourceVersion=100",
			expected:   true,
		},
		{
			name:       "the user specifies a resource version of zero",
			requestURI: "/api/v1/namespaces?resourceVersion=0",
			expected:   true,
		},
		{
			name:       "the user specifies an invalid resource version",
			requestURI: "/api/v1/namespaces?resourceVersion=foo",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request, err := http.NewRequest(http.MethodGet, test.requestURI, nil)
			if err != nil {
				t.Fatalf("failed to create new http request - %v", err)
			}

			resultGot := hasValidResourceVersion(request)

			if test.expected != resultGot {
				t.Errorf("expected: %t, but got: %t for request: %q", test.expected, resultGot, test.requestURI)
			}
		})
	}
}

func TestWithResourceVersionValidation(t *testing.T) {
	tests := []struct {
		name                        string
		requestURL                  string
		handlerInvokedCountExpected int
		statusCodeExpected          int
	}{
		{
			name:                        "user does not specify a resource version, request is served",
			requestURL:                  "/api/v1/namespaces?resourceVersion=",
			statusCodeExpected:          http.StatusOK,
			handlerInvokedCountExpected: 1,
		},
		{
			name:                        "user specifies a valid resource version, request is served",
			requestURL:                  "/api/v1/namespaces?resourceVersion=100",
			statusCodeExpected:          http.StatusOK,
			handlerInvokedCountExpected: 1,
		},
		{
			name:                        "user specifies a resource version of zero, request is served",
			requestURL:                  "/api/v1/namespaces?resourceVersion=0",
			statusCodeExpected:          http.StatusOK,
			handlerInvokedCountExpected: 1,
		},
		{
			name:               "user specifies an invalid resource version, request is rejected with HTTP 400",
			requestURL:         "/api/v1/namespaces?resourceVersion=foo",
			statusCodeExpected: http.StatusBadRequest,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var handlerInvokedCount int
			handler := http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
				handlerInvokedCount++
			})

			withResourceVersionValidation := WithResourceVersionValidation(handler, newSerializer())
			withResourceVersionValidation = WithRequestInfo(withResourceVersionValidation, &fakeRequestResolver{})

			testRequest := newRequest(t, test.requestURL)
			w := httptest.NewRecorder()
			withResourceVersionValidation.ServeHTTP(w, testRequest)

			if test.handlerInvokedCountExpected != handlerInvokedCount {
				t.Errorf("expected the request handler to be invoked %d times, but was actually invoked %d times", test.handlerInvokedCountExpected, handlerInvokedCount)
			}

			statusCodeGot := w.Result().StatusCode
			if test.statusCodeExpected != statusCodeGot {
				t.Errorf("expected status code %d but got: %d", test.statusCodeExpected, statusCodeGot)
			}
		})
	}
}

func newRequest(t *testing.T, requestURL string) *http.Request {
	req, err := http.NewRequest(http.MethodGet, requestURL, nil)
	if err != nil {
		t.Fatalf("failed to create new http request - %v", err)
	}

	return req
}

func newSerializer() runtime.NegotiatedSerializer {
	scheme := runtime.NewScheme()
	return serializer.NewCodecFactory(scheme).WithoutConversion()
}

type fakeRequestResolver struct{}

func (r fakeRequestResolver) NewRequestInfo(req *http.Request) (*request.RequestInfo, error) {
	return &request.RequestInfo{}, nil
}
