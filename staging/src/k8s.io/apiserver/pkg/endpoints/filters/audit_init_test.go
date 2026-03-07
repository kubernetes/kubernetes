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
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/uuid"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/audit"
)

func TestWithAuditID(t *testing.T) {
	largeAuditID := fmt.Sprintf("%s-%s", uuid.New().String(), uuid.New().String())
	tests := []struct {
		name             string
		newAuditIDFunc   func() string
		auditIDSpecified string
		auditIDExpected  string
	}{
		{
			name:             "user specifies a value for Audit-ID in the request header",
			auditIDSpecified: "foo-bar-baz",
			auditIDExpected:  "foo-bar-baz",
		},
		{
			name: "user does not specify a value for Audit-ID in the request header",
			newAuditIDFunc: func() string {
				return "foo-bar-baz"
			},
			auditIDExpected: "foo-bar-baz",
		},
		{
			name:             "the value in Audit-ID request header is too large, should not be truncated",
			auditIDSpecified: largeAuditID,
			auditIDExpected:  largeAuditID,
		},
		{
			name: "the generated Audit-ID is too large, should not be truncated",
			newAuditIDFunc: func() string {
				return largeAuditID
			},
			auditIDExpected: largeAuditID,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := t.Context()
			const auditKey = "Audit-ID"
			var (
				innerHandlerCallCount int
				auditIDGot            string
				found                 bool
			)
			handler := http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
				innerHandlerCallCount++

				// does the inner handler see the audit ID?
				v, ok := audit.AuditIDFrom(req.Context())

				found = ok
				auditIDGot = string(v)
			})

			wrapped := WithAuditInit(handler)
			if test.newAuditIDFunc != nil {
				wrapped = withAuditInit(handler, test.newAuditIDFunc, nil, nil)
			}

			testRequest, err := http.NewRequestWithContext(ctx, http.MethodGet, "/api/v1/namespaces", nil)
			if err != nil {
				t.Fatalf("failed to create new http request - %v", err)
			}
			if len(test.auditIDSpecified) > 0 {
				testRequest.Header.Set(auditKey, test.auditIDSpecified)
			}

			w := httptest.NewRecorder()
			wrapped.ServeHTTP(w, testRequest)

			if innerHandlerCallCount != 1 {
				t.Errorf("WithAuditID: expected the inner handler to be invoked once, but was invoked %d times", innerHandlerCallCount)
			}
			if !found {
				t.Error("WithAuditID: expected request.AuditIDFrom to return true, but got false")
			}
			if test.auditIDExpected != auditIDGot {
				t.Errorf("WithAuditID: expected the request context to have: %q, but got=%q", test.auditIDExpected, auditIDGot)
			}

			auditIDEchoed := w.Header().Get(auditKey)
			if test.auditIDExpected != auditIDEchoed {
				t.Errorf("WithAuditID: expected Audit-ID response header: %q, but got: %q", test.auditIDExpected, auditIDEchoed)
			}
		})
	}
}

func TestWithValidatingAuditID(t *testing.T) {
	largeAuditID := strings.Repeat("a", maxAuditIDLength+1)
	tests := []struct {
		name                 string
		newAuditIDFunc       func() string
		auditIDSpecified     string
		auditIDExpected      string
		innerHandlerExpected bool
		expectedStatusCode   int
	}{
		{
			name:                 "user specifies a value for Audit-ID in the request header",
			auditIDSpecified:     "foo-bar-baz",
			auditIDExpected:      "foo-bar-baz",
			innerHandlerExpected: true,
			expectedStatusCode:   http.StatusOK,
		},
		{
			name: "user does not specify a value for Audit-ID in the request header",
			newAuditIDFunc: func() string {
				return "foo-bar-baz"
			},
			auditIDExpected:      "foo-bar-baz",
			innerHandlerExpected: true,
			expectedStatusCode:   http.StatusOK,
		},
		{
			name: "the generated Audit-ID is too large, should fail validation",
			newAuditIDFunc: func() string {
				return largeAuditID
			},
			auditIDExpected:      largeAuditID,
			innerHandlerExpected: false,
			expectedStatusCode:   http.StatusBadRequest,
		},
		{
			name: "the generated Audit-ID has invalid characters, should fail validation",
			newAuditIDFunc: func() string {
				return "foo-$%!@#-baz"
			},
			auditIDExpected:      "foo-$%!@#-baz",
			innerHandlerExpected: false,
			expectedStatusCode:   http.StatusBadRequest,
		},
		{
			name:                 "the value in Audit-ID request header is too large, should return bad request",
			auditIDSpecified:     largeAuditID,
			auditIDExpected:      largeAuditID,
			innerHandlerExpected: false,
			expectedStatusCode:   http.StatusBadRequest,
		},
		{
			name:                 "user specifies Audit-ID with invalid characters, should return bad request",
			auditIDSpecified:     "foo-$%!@#-baz",
			auditIDExpected:      "foo-$%!@#-baz",
			innerHandlerExpected: false,
			expectedStatusCode:   http.StatusBadRequest,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			const auditKey = "Audit-ID"
			var (
				innerHandlerCallCount int
				auditIDGot            string
				found                 bool
			)
			handler := http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
				innerHandlerCallCount++

				// does the inner handler see the audit ID?
				v, ok := audit.AuditIDFrom(req.Context())

				found = ok
				auditIDGot = string(v)
			})

			negotiatedSerializer := serializer.NewCodecFactory(runtime.NewScheme())
			wrapped := WithValidatingAuditInit(handler, negotiatedSerializer)
			if test.newAuditIDFunc != nil {
				wrapped = withAuditInit(handler, test.newAuditIDFunc, validateAuditID, invalidAuditID(negotiatedSerializer))
			}

			testRequest, err := http.NewRequest(http.MethodGet, "/api/v1/namespaces", nil)
			if err != nil {
				t.Fatalf("failed to create new http request - %v", err)
			}
			if len(test.auditIDSpecified) > 0 {
				testRequest.Header.Set(auditKey, test.auditIDSpecified)
			}

			w := httptest.NewRecorder()
			wrapped.ServeHTTP(w, testRequest)

			// Only run these validations on the test case if the inner handler is expected to be called.
			if test.innerHandlerExpected {
				if innerHandlerCallCount != 1 {
					t.Errorf("WithValidatingAuditID: expected the inner handler to be invoked one time, but was invoked %d times", innerHandlerCallCount)
				}
				if !found {
					t.Error("WithValidatingAuditID: expected request.AuditIDFrom to return true, but got false")
				}
				if test.auditIDExpected != auditIDGot {
					t.Errorf("WithValidatingAuditID: expected the request context to have: %q, but got=%q", test.auditIDExpected, auditIDGot)
				}
			}

			if w.Code != test.expectedStatusCode {
				t.Errorf("WithValidatingAuditID: expected status code %v but got %v", test.expectedStatusCode, w.Code)
			}

			auditIDEchoed := w.Header().Get(auditKey)
			if test.auditIDExpected != auditIDEchoed {
				t.Errorf("WithValidatingAuditID: expected Audit-ID response header: %q, but got: %q", test.auditIDExpected, auditIDEchoed)
			}
		})
	}
}

func TestDefaultNewAuditIDAlwaysValid(t *testing.T) {
	// try 1000 generated audit id's to build reasonable confidence that the
	// defaultNewAuditID function always returns a valid audit id.
	for range 1000 {
		generatedAuditID := defaultNewAuditID()
		if !validateAuditID(generatedAuditID) {
			t.Errorf("defaultNewAuditID: generated audit id %q is not valid. length: %d, matches regex?: %v", generatedAuditID, len(generatedAuditID), auditIDPatternRegex.MatchString(generatedAuditID))
		}
	}
}
