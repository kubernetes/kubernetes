/*
Copyright 2014 The Kubernetes Authors.

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
	"context"
	"errors"
	"github.com/stretchr/testify/assert"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestAuthenticateRequestWithAud(t *testing.T) {
	success, failed := 0, 0
	testcases := []struct {
		name          string
		apiAuds       []string
		respAuds      []string
		expectSuccess bool
	}{
		{
			name:          "no api audience and no audience in response",
			apiAuds:       nil,
			respAuds:      nil,
			expectSuccess: true,
		},
		{
			name:          "audience in response",
			apiAuds:       nil,
			respAuds:      []string{"other"},
			expectSuccess: true,
		},
		{
			name:          "with api audience",
			apiAuds:       authenticator.Audiences([]string{"other"}),
			respAuds:      nil,
			expectSuccess: true,
		},
		{
			name:          "api audience matching response audience",
			apiAuds:       authenticator.Audiences([]string{"other"}),
			respAuds:      []string{"other"},
			expectSuccess: true,
		},
		{
			name:          "api audience non-matching response audience",
			apiAuds:       authenticator.Audiences([]string{"other"}),
			respAuds:      []string{"some"},
			expectSuccess: false,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			success, failed = 0, 0
			auth := WithAuthentication(
				http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
					if tc.expectSuccess {
						success = 1
					} else {
						t.Errorf("unexpected call to handler")
					}
				}),
				authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
					if req.Header.Get("Authorization") == "Something" {
						return &authenticator.Response{User: &user.DefaultInfo{Name: "user"}, Audiences: authenticator.Audiences(tc.respAuds)}, true, nil
					}
					return nil, false, errors.New("Authorization header is missing.")
				}),
				http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
					if tc.expectSuccess {
						t.Errorf("unexpected call to failed")
					} else {
						failed = 1
					}
				}),
				tc.apiAuds,
			)
			auth.ServeHTTP(httptest.NewRecorder(), &http.Request{Header: map[string][]string{"Authorization": {"Something"}}})
			if tc.expectSuccess {
				assert.Equal(t, 1, success)
				assert.Equal(t, 0, failed)
			} else {
				assert.Equal(t, 0, success)
				assert.Equal(t, 1, failed)
			}
		})
	}
}

func TestAuthenticateMetrics(t *testing.T) {
	testcases := []struct {
		name         string
		header       bool
		apiAuds      []string
		respAuds     []string
		expectMetric bool
		expectOk     bool
		expectError  bool
	}{
		{
			name:        "no api audience and no audience in response",
			header:      true,
			apiAuds:     nil,
			respAuds:    nil,
			expectOk:    true,
			expectError: false,
		},
		{
			name:        "api audience matching response audience",
			header:      true,
			apiAuds:     authenticator.Audiences([]string{"other"}),
			respAuds:    []string{"other"},
			expectOk:    true,
			expectError: false,
		},
		{
			name:        "no intersection results in error",
			header:      true,
			apiAuds:     authenticator.Audiences([]string{"other"}),
			respAuds:    []string{"some"},
			expectOk:    true,
			expectError: true,
		},
		{
			name:        "no header results in error",
			header:      false,
			apiAuds:     authenticator.Audiences([]string{"other"}),
			respAuds:    []string{"some"},
			expectOk:    false,
			expectError: true,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			called := 0
			auth := withAuthentication(
				http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
				}),
				authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
					if req.Header.Get("Authorization") == "Something" {
						return &authenticator.Response{User: &user.DefaultInfo{Name: "user"}, Audiences: authenticator.Audiences(tc.respAuds)}, true, nil
					}
					return nil, false, errors.New("Authorization header is missing.")
				}),
				http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
				}),
				tc.apiAuds,
				func(ctx context.Context, resp *authenticator.Response, ok bool, err error, apiAudiences authenticator.Audiences, authStart time.Time, authFinish time.Time) {
					called = 1
					if tc.expectOk != ok {
						t.Errorf("unexpected value of ok argument: %t", ok)
					}
					if tc.expectError {
						if err == nil {
							t.Errorf("unexpected value of err argument: %s", err)
						}
					} else {
						if err != nil {
							t.Errorf("unexpected value of err argument: %s", err)
						}
					}
				},
			)
			if tc.header {
				auth.ServeHTTP(httptest.NewRecorder(), &http.Request{Header: map[string][]string{"Authorization": {"Something"}}})
			} else {
				auth.ServeHTTP(httptest.NewRecorder(), &http.Request{})
			}
			assert.Equal(t, 1, called)
		})
	}
}

func TestAuthenticateRequest(t *testing.T) {
	success := make(chan struct{})
	auth := WithAuthentication(
		http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
			ctx := req.Context()
			user, ok := genericapirequest.UserFrom(ctx)
			if user == nil || !ok {
				t.Errorf("no user stored in context: %#v", ctx)
			}
			if req.Header.Get("Authorization") != "" {
				t.Errorf("Authorization header should be removed from request on success: %#v", req)
			}
			close(success)
		}),
		authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
			if req.Header.Get("Authorization") == "Something" {
				return &authenticator.Response{User: &user.DefaultInfo{Name: "user"}}, true, nil
			}
			return nil, false, errors.New("Authorization header is missing.")
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			t.Errorf("unexpected call to failed")
		}),
		nil,
	)

	auth.ServeHTTP(httptest.NewRecorder(), &http.Request{Header: map[string][]string{"Authorization": {"Something"}}})

	<-success
}

func TestAuthenticateRequestFailed(t *testing.T) {
	failed := make(chan struct{})
	auth := WithAuthentication(
		http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
			t.Errorf("unexpected call to handler")
		}),
		authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
			return nil, false, nil
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			close(failed)
		}),
		nil,
	)

	auth.ServeHTTP(httptest.NewRecorder(), &http.Request{})

	<-failed
}

func TestAuthenticateRequestError(t *testing.T) {
	failed := make(chan struct{})
	auth := WithAuthentication(
		http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
			t.Errorf("unexpected call to handler")
		}),
		authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
			return nil, false, errors.New("failure")
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			close(failed)
		}),
		nil,
	)

	auth.ServeHTTP(httptest.NewRecorder(), &http.Request{})

	<-failed
}
