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
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/authenticatorfactory"
	"k8s.io/apiserver/pkg/authentication/request/headerrequest"
	"k8s.io/apiserver/pkg/authentication/user"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
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
				nil,
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
				nil,
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
		nil,
	)

	auth.ServeHTTP(httptest.NewRecorder(), &http.Request{})

	<-failed
}

func TestAuthenticateRequestClearHeaders(t *testing.T) {
	testcases := map[string]struct {
		nameHeaders        []string
		groupHeaders       []string
		extraPrefixHeaders []string
		requestHeaders     http.Header
		finalHeaders       http.Header
	}{
		"user match": {
			nameHeaders:    []string{"X-Remote-User"},
			requestHeaders: http.Header{"X-Remote-User": {"Bob"}},
		},
		"user first match": {
			nameHeaders: []string{
				"X-Remote-User",
				"A-Second-X-Remote-User",
				"Another-X-Remote-User",
			},
			requestHeaders: http.Header{
				"X-Remote-User":          {"", "First header, second value"},
				"A-Second-X-Remote-User": {"Second header, first value", "Second header, second value"},
				"Another-X-Remote-User":  {"Third header, first value"}},
		},
		"user case-insensitive": {
			nameHeaders:    []string{"x-REMOTE-user"},             // configured headers can be case-insensitive
			requestHeaders: http.Header{"X-Remote-User": {"Bob"}}, // the parsed headers are normalized by the http package
		},

		"groups none": {
			nameHeaders:  []string{"X-Remote-User"},
			groupHeaders: []string{"X-Remote-Group"},
			requestHeaders: http.Header{
				"X-Remote-User": {"Bob"},
			},
		},
		"groups all matches": {
			nameHeaders:  []string{"X-Remote-User"},
			groupHeaders: []string{"X-Remote-Group-1", "X-Remote-Group-2"},
			requestHeaders: http.Header{
				"X-Remote-User":    {"Bob"},
				"X-Remote-Group-1": {"one-a", "one-b"},
				"X-Remote-Group-2": {"two-a", "two-b"},
			},
		},
		"groups case-insensitive": {
			nameHeaders:  []string{"X-REMOTE-User"},
			groupHeaders: []string{"X-REMOTE-Group"},
			requestHeaders: http.Header{
				"X-Remote-User":  {"Bob"},
				"X-Remote-Group": {"Users"},
			},
		},

		"extra prefix matches case-insensitive": {
			nameHeaders:        []string{"X-Remote-User"},
			groupHeaders:       []string{"X-Remote-Group-1", "X-Remote-Group-2"},
			extraPrefixHeaders: []string{"X-Remote-Extra-1-", "X-Remote-Extra-2-"},
			requestHeaders: http.Header{
				"X-Remote-User":         {"Bob"},
				"X-Remote-Group-1":      {"one-a", "one-b"},
				"X-Remote-Group-2":      {"two-a", "two-b"},
				"X-Remote-extra-1-key1": {"alfa", "bravo"},
				"X-Remote-Extra-1-Key2": {"charlie", "delta"},
				"X-Remote-Extra-1-":     {"india", "juliet"},
				"X-Remote-extra-2-":     {"kilo", "lima"},
				"X-Remote-extra-2-Key1": {"echo", "foxtrot"},
				"X-Remote-Extra-2-key2": {"golf", "hotel"},
			},
		},

		"extra prefix matches case-insensitive with unrelated headers": {
			nameHeaders:        []string{"X-Remote-User"},
			groupHeaders:       []string{"X-Remote-Group-1", "X-Remote-Group-2"},
			extraPrefixHeaders: []string{"X-Remote-Extra-1-", "X-Remote-Extra-2-"},
			requestHeaders: http.Header{
				"X-Group-Remote":        {"snorlax"}, // unrelated header
				"X-Group-Bear":          {"panda"},   // another unrelated header
				"X-Remote-User":         {"Bob"},
				"X-Remote-Group-1":      {"one-a", "one-b"},
				"X-Remote-Group-2":      {"two-a", "two-b"},
				"X-Remote-extra-1-key1": {"alfa", "bravo"},
				"X-Remote-Extra-1-Key2": {"charlie", "delta"},
				"X-Remote-Extra-1-":     {"india", "juliet"},
				"X-Remote-extra-2-":     {"kilo", "lima"},
				"X-Remote-extra-2-Key1": {"echo", "foxtrot"},
				"X-Remote-Extra-2-key2": {"golf", "hotel"},
			},
			finalHeaders: http.Header{
				"X-Group-Remote": {"snorlax"},
				"X-Group-Bear":   {"panda"},
			},
		},

		"custom config but request contains standard headers": {
			nameHeaders:        []string{"foo"},
			groupHeaders:       []string{"bar"},
			extraPrefixHeaders: []string{"baz"},
			requestHeaders: http.Header{
				"X-Remote-User":         {"Bob"},
				"X-Remote-Group-1":      {"one-a", "one-b"},
				"X-Remote-Group-2":      {"two-a", "two-b"},
				"X-Remote-extra-1-key1": {"alfa", "bravo"},
				"X-Remote-Extra-1-Key2": {"charlie", "delta"},
				"X-Remote-Extra-1-":     {"india", "juliet"},
				"X-Remote-extra-2-":     {"kilo", "lima"},
				"X-Remote-extra-2-Key1": {"echo", "foxtrot"},
				"X-Remote-Extra-2-key2": {"golf", "hotel"},
			},
			finalHeaders: http.Header{
				"X-Remote-Group-1": {"one-a", "one-b"},
				"X-Remote-Group-2": {"two-a", "two-b"},
			},
		},

		"custom config but request contains standard and custom headers": {
			nameHeaders:        []string{"one"},
			groupHeaders:       []string{"two"},
			extraPrefixHeaders: []string{"three-"},
			requestHeaders: http.Header{
				"X-Remote-User":         {"Bob"},
				"X-Remote-Group-3":      {"one-a", "one-b"},
				"X-Remote-Group-4":      {"two-a", "two-b"},
				"X-Remote-extra-1-key1": {"alfa", "bravo"},
				"X-Remote-Extra-1-Key2": {"charlie", "delta"},
				"X-Remote-Extra-1-":     {"india", "juliet"},
				"X-Remote-extra-2-":     {"kilo", "lima"},
				"X-Remote-extra-2-Key1": {"echo", "foxtrot"},
				"X-Remote-Extra-2-key2": {"golf", "hotel"},
				"One":                   {"echo", "foxtrot"},
				"Two":                   {"golf", "hotel"},
				"Three-Four":            {"india", "juliet"},
			},
			finalHeaders: http.Header{
				"X-Remote-Group-3": {"one-a", "one-b"},
				"X-Remote-Group-4": {"two-a", "two-b"},
			},
		},

		"escaped extra keys": {
			nameHeaders:        []string{"X-Remote-User"},
			groupHeaders:       []string{"X-Remote-Group"},
			extraPrefixHeaders: []string{"X-Remote-Extra-"},
			requestHeaders: http.Header{
				"X-Remote-User":                                            {"Bob"},
				"X-Remote-Group":                                           {"one-a", "one-b"},
				"X-Remote-Extra-Alpha":                                     {"alphabetical"},
				"X-Remote-Extra-Alph4num3r1c":                              {"alphanumeric"},
				"X-Remote-Extra-Percent%20encoded":                         {"percent encoded"},
				"X-Remote-Extra-Almost%zzpercent%xxencoded":                {"not quite percent encoded"},
				"X-Remote-Extra-Example.com%2fpercent%2520encoded":         {"url with double percent encoding"},
				"X-Remote-Extra-Example.com%2F%E4%BB%8A%E6%97%A5%E3%81%AF": {"url with unicode"},
				"X-Remote-Extra-Abc123!#$+.-_*\\^`~|'":                     {"header key legal characters"},
			},
		},
	}

	for k, testcase := range testcases {
		t.Run(k, func(t *testing.T) {
			var handlerCalls, authnCalls int
			auth := WithAuthentication(
				http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
					handlerCalls++
				}),
				authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
					authnCalls++
					return &authenticator.Response{User: &user.DefaultInfo{Name: "panda"}}, true, nil
				}),
				http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
					t.Errorf("unexpected call to handler")
				}),
				nil,
				&authenticatorfactory.RequestHeaderConfig{
					UsernameHeaders:     headerrequest.StaticStringSlice(testcase.nameHeaders),
					GroupHeaders:        headerrequest.StaticStringSlice(testcase.groupHeaders),
					ExtraHeaderPrefixes: headerrequest.StaticStringSlice(testcase.extraPrefixHeaders),
				},
			)

			auth.ServeHTTP(httptest.NewRecorder(), &http.Request{Header: testcase.requestHeaders})

			if handlerCalls != 1 || authnCalls != 1 {
				t.Errorf("unexpected calls: handlerCalls=%d, authnCalls=%d", handlerCalls, authnCalls)
			}

			want := testcase.finalHeaders
			if want == nil && testcase.requestHeaders != nil {
				want = http.Header{}
			}
			if diff := cmp.Diff(want, testcase.requestHeaders); len(diff) > 0 {
				t.Errorf("unexpected final headers (-want +got):\n%s", diff)
			}
		})
	}
}
