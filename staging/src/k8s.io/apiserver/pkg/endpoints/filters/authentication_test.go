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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"golang.org/x/net/http2"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/authenticatorfactory"
	"k8s.io/apiserver/pkg/authentication/request/anonymous"
	"k8s.io/apiserver/pkg/authentication/request/headerrequest"
	"k8s.io/apiserver/pkg/authentication/request/union"
	"k8s.io/apiserver/pkg/authentication/user"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/scheme"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
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
		uidHeaders         []string
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
		"uid none": {
			nameHeaders: []string{"X-Remote-User"},
			uidHeaders:  []string{"X-Remote-Uid"},
			requestHeaders: http.Header{
				"X-Remote-User": {"Alice"},
			},
		},
		"uid all matches": {
			nameHeaders: []string{"X-Remote-User"},
			uidHeaders:  []string{"X-Remote-Uid-1", "X-Remote-Uid-2"},
			requestHeaders: http.Header{
				"X-Remote-User":  {"Alice"},
				"X-Remote-Uid-1": {"one"},
				"X-Remote-Uid-2": {"two", "three"},
			},
		},
		"uid case-insensitive": {
			nameHeaders: []string{"X-Remote-USER"},
			uidHeaders:  []string{"X-REMOTE-UID-1"},
			requestHeaders: http.Header{
				"X-Remote-User":  {"Alice"},
				"X-Remote-Uid-1": {"one"},
			},
		},

		"extra prefix matches case-insensitive": {
			nameHeaders:        []string{"X-Remote-User"},
			uidHeaders:         []string{"X-Remote-Uid-1"},
			groupHeaders:       []string{"X-Remote-Group-1", "X-Remote-Group-2"},
			extraPrefixHeaders: []string{"X-Remote-Extra-1-", "X-Remote-Extra-2-"},
			requestHeaders: http.Header{
				"X-Remote-User":         {"Bob"},
				"X-Remote-Uid-1":        {"bobs-uid"},
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
			uidHeaders:         []string{"X-Remote-Uid"},
			groupHeaders:       []string{"X-Remote-Group-1", "X-Remote-Group-2"},
			extraPrefixHeaders: []string{"X-Remote-Extra-1-", "X-Remote-Extra-2-"},
			requestHeaders: http.Header{
				"X-Group-Remote":        {"snorlax"}, // unrelated header
				"X-Group-Bear":          {"panda"},   // another unrelated header
				"X-Uid-Remote":          {"bobs-unrelated-uid"},
				"X-Remote-User":         {"Bob"},
				"X-Remote-Uid":          {"bobs-uid"},
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
				"X-Uid-Remote":   {"bobs-unrelated-uid"},
			},
		},

		"custom config but request contains standard headers": {
			nameHeaders:        []string{"foo"},
			uidHeaders:         []string{"footoo"},
			groupHeaders:       []string{"bar"},
			extraPrefixHeaders: []string{"baz"},
			requestHeaders: http.Header{
				"X-Remote-User":         {"Bob"},
				"X-Remote-Uid":          {"bobs-uid"},
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
			uidHeaders:         []string{"onetoo"},
			groupHeaders:       []string{"two"},
			extraPrefixHeaders: []string{"three-"},
			requestHeaders: http.Header{
				"X-Remote-User":         {"Bob"},
				"X-Remote-Uid":          {"bobs-uid"},
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
			uidHeaders:         []string{"X-Remote-Uid"},
			groupHeaders:       []string{"X-Remote-Group"},
			extraPrefixHeaders: []string{"X-Remote-Extra-"},
			requestHeaders: http.Header{
				"X-Remote-User":                                            {"Bob"},
				"X-Remote-Uid":                                             {"bobs-uid"},
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
					Config: headerrequest.NewStaticRequestHeaderConfig(testcase.nameHeaders, testcase.uidHeaders, testcase.groupHeaders, testcase.extraPrefixHeaders, nil),
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

func TestAuthenticateRequestClearsCustomRequestHeaders(t *testing.T) {
	snapshot := &headerrequest.RequestHeaderConfig{UsernameHeaders: []string{"X-Custom-User"}}
	snapshotCalls := 0
	auth := WithAuthentication(
		http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
			if req.Header.Get("X-Custom-User") != "" {
				t.Errorf("expected the custom request header to be cleared")
			}
		}),
		authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
			return &authenticator.Response{User: &user.DefaultInfo{Name: "panda"}}, true, nil
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			t.Errorf("unexpected authentication failure")
		}),
		nil,
		&authenticatorfactory.RequestHeaderConfig{
			Config: headerrequest.RequestHeaderConfigProviderFunc(func() *headerrequest.RequestHeaderConfig {
				snapshotCalls++
				return snapshot
			}),
		},
	)

	auth.ServeHTTP(httptest.NewRecorder(), &http.Request{Header: http.Header{"X-Custom-User": {"alice"}}})
	if snapshotCalls != 1 {
		t.Fatalf("expected one request header snapshot for header clearing, got %d", snapshotCalls)
	}
}

// TestWithAuthenticationRequestHeaderProvidersAreIsolated runs the filter around a union of two
// request header authenticators with different CA roots and different configurations: A accepts
// any common name signed by its CA, B only accepts "proxy-b" signed by its own CA. Each
// authenticator must decide using its own configuration, so neither A's allowlist, A's header
// names nor A's unavailability may leak into B's decision.
func TestWithAuthenticationRequestHeaderProvidersAreIsolated(t *testing.T) {
	caA, certsA := newRequestHeaderTestCA(t, "ca-a", "ordinary-client")
	caB, certsB := newRequestHeaderTestCA(t, "ca-b", "ordinary-client", "proxy-b")

	tests := []struct {
		name            string
		certificate     *x509.Certificate
		unavailableA    bool
		unavailableB    bool
		restrictedA     bool
		distinctHeaders bool
		wantStatus      int
		wantUser        *user.DefaultInfo
	}{
		{
			name:        "disallowed CN signed by CA B is rejected by B's allowlist",
			certificate: certsB[0],
			wantStatus:  http.StatusUnauthorized,
		},
		{
			name:        "allowed CN signed by CA B is accepted",
			certificate: certsB[1],
			wantStatus:  http.StatusOK,
			wantUser:    &user.DefaultInfo{Name: "admin", Groups: []string{}, Extra: map[string][]string{}},
		},
		{
			name:         "unavailable A does not disable B",
			certificate:  certsB[1],
			unavailableA: true,
			wantStatus:   http.StatusOK,
			wantUser:     &user.DefaultInfo{Name: "admin", Groups: []string{}, Extra: map[string][]string{}},
		},
		{
			name:            "B extracts and clears its own header names",
			certificate:     certsB[1],
			distinctHeaders: true,
			wantStatus:      http.StatusOK,
			wantUser:        &user.DefaultInfo{Name: "b-user", UID: "b-uid", Groups: []string{"b-group"}, Extra: map[string][]string{"scope": {"b-scope"}}},
		},
		{
			name:        "B is not restricted by A's allowlist",
			certificate: certsB[1],
			restrictedA: true,
			wantStatus:  http.StatusOK,
			wantUser:    &user.DefaultInfo{Name: "admin", Groups: []string{}, Extra: map[string][]string{}},
		},
		{
			name:         "unavailable B cannot borrow A's configuration",
			certificate:  certsB[1],
			unavailableB: true,
			wantStatus:   http.StatusUnauthorized,
		},
		{
			name:        "unrestricted A still accepts any CN signed by CA A",
			certificate: certsA[0],
			wantStatus:  http.StatusOK,
			wantUser:    &user.DefaultInfo{Name: "admin", Groups: []string{}, Extra: map[string][]string{}},
		},
		{
			name:       "no client certificate never authenticates request headers",
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:         "both unavailable fails closed",
			certificate:  certsB[1],
			unavailableA: true,
			unavailableB: true,
			wantStatus:   http.StatusUnauthorized,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			configA := headerrequest.NewStaticRequestHeaderConfig([]string{"X-Remote-User"}, nil, nil, nil, nil)
			configB := headerrequest.NewStaticRequestHeaderConfig([]string{"X-Remote-User"}, nil, nil, nil, []string{"proxy-b"})
			if tt.distinctHeaders {
				configA = headerrequest.NewStaticRequestHeaderConfig([]string{"X-A-User"}, nil, nil, nil, nil)
				configB = headerrequest.NewStaticRequestHeaderConfig([]string{"X-B-User"}, []string{"X-B-Uid"}, []string{"X-B-Group"}, []string{"X-B-Extra-"}, []string{"proxy-b"})
			}
			if tt.restrictedA {
				configA = headerrequest.NewStaticRequestHeaderConfig([]string{"X-Remote-User"}, nil, nil, nil, []string{"proxy-a"})
			}
			if tt.unavailableA {
				configA = nil
			}
			if tt.unavailableB {
				configB = nil
			}
			// function providers are not comparable, so a fix relying on provider identity
			// must not compare interface values
			providerA := headerrequest.RequestHeaderConfigProviderFunc(func() *headerrequest.RequestHeaderConfig { return configA })
			providerB := headerrequest.RequestHeaderConfigProviderFunc(func() *headerrequest.RequestHeaderConfig { return configB })

			var gotUser user.Info
			var remaining http.Header
			handler := WithAuthentication(
				http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
					gotUser, _ = genericapirequest.UserFrom(req.Context())
					remaining = req.Header.Clone()
					w.WriteHeader(http.StatusOK)
				}),
				union.New(headerrequest.NewSecure(providerA, caA), headerrequest.NewSecure(providerB, caB)),
				http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusUnauthorized) }),
				nil,
				&authenticatorfactory.RequestHeaderConfig{Config: providerA, CAContentProvider: caA},
			)

			req := httptest.NewRequest(http.MethodGet, "https://extension.example.test/", nil)
			if tt.certificate != nil {
				req.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{tt.certificate}}
			}
			req.Header.Set("X-Remote-User", "admin")
			if tt.distinctHeaders {
				req.Header.Set("X-A-User", "a-admin")
				req.Header.Set("X-B-User", "b-user")
				req.Header.Set("X-B-Uid", "b-uid")
				req.Header.Set("X-B-Group", "b-group")
				req.Header.Set("X-B-Extra-Scope", "b-scope")
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, req)

			if response.Code != tt.wantStatus {
				t.Errorf("HTTP status = %d, want %d", response.Code, tt.wantStatus)
			}
			if tt.wantUser == nil {
				if gotUser != nil {
					t.Errorf("expected no authenticated user, got %#v", gotUser)
				}
				return
			}
			if diff := cmp.Diff(tt.wantUser, gotUser); len(diff) > 0 {
				t.Errorf("unexpected user (-want +got):\n%s", diff)
			}
			for _, name := range []string{"X-Remote-User", "X-A-User", "X-B-User", "X-B-Uid", "X-B-Group", "X-B-Extra-Scope"} {
				if remaining.Get(name) != "" {
					t.Errorf("authentication header %s leaked to the handler", name)
				}
			}
		})
	}
}

// newRequestHeaderTestCA returns a CA content provider and one client certificate per common
// name, all signed by that CA.
func newRequestHeaderTestCA(t *testing.T, name string, commonNames ...string) (dynamiccertificates.CAContentProvider, []*x509.Certificate) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	ca := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: name},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign,
	}
	der, err := x509.CreateCertificate(rand.Reader, ca, ca, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	parsedCA, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}
	provider, err := dynamiccertificates.NewStaticCAContent(name, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}))
	if err != nil {
		t.Fatal(err)
	}
	certs := make([]*x509.Certificate, 0, len(commonNames))
	for i, cn := range commonNames {
		leaf := &x509.Certificate{
			SerialNumber: big.NewInt(int64(i + 2)),
			Subject:      pkix.Name{CommonName: cn},
			NotBefore:    ca.NotBefore,
			NotAfter:     ca.NotAfter,
			KeyUsage:     x509.KeyUsageDigitalSignature,
			ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		}
		der, err := x509.CreateCertificate(rand.Reader, leaf, parsedCA, &key.PublicKey, key)
		if err != nil {
			t.Fatal(err)
		}
		cert, err := x509.ParseCertificate(der)
		if err != nil {
			t.Fatal(err)
		}
		certs = append(certs, cert)
	}
	return provider, certs
}
func TestUnauthenticatedHTTP2ClientConnectionClose(t *testing.T) {
	s := httptest.NewUnstartedServer(WithAuthentication(
		http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok")) }),
		authenticator.RequestFunc(func(r *http.Request) (*authenticator.Response, bool, error) {
			switch r.Header.Get("Authorization") {
			case "known":
				return &authenticator.Response{User: &user.DefaultInfo{Name: "panda"}}, true, nil
			case "error":
				return nil, false, errors.New("authn err")
			case "anonymous":
				return anonymous.NewAuthenticator(nil).AuthenticateRequest(r)
			case "anonymous_group":
				return &authenticator.Response{User: &user.DefaultInfo{Groups: []string{user.AllUnauthenticated}}}, true, nil
			default:
				return nil, false, nil
			}
		}),
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			r = r.WithContext(genericapirequest.WithRequestInfo(r.Context(), &genericapirequest.RequestInfo{}))
			Unauthorized(scheme.Codecs).ServeHTTP(w, r)
		}),
		nil,
		nil,
	))

	http2Options := &http2.Server{}

	if err := http2.ConfigureServer(s.Config, http2Options); err != nil {
		t.Fatal(err)
	}

	s.TLS = s.Config.TLSConfig

	s.StartTLS()
	t.Cleanup(s.Close)

	const reqs = 4

	cases := []struct {
		name                   string
		authorizationHeader    string
		skipHTTP2DOSMitigation bool
		expectConnections      uint64
	}{
		{
			name:                   "known",
			authorizationHeader:    "known",
			skipHTTP2DOSMitigation: false,
			expectConnections:      1,
		},
		{
			name:                   "error",
			authorizationHeader:    "error",
			skipHTTP2DOSMitigation: false,
			expectConnections:      reqs,
		},
		{
			name:                   "anonymous",
			authorizationHeader:    "anonymous",
			skipHTTP2DOSMitigation: false,
			expectConnections:      reqs,
		},
		{
			name:                   "anonymous_group",
			authorizationHeader:    "anonymous_group",
			skipHTTP2DOSMitigation: false,
			expectConnections:      reqs,
		},
		{
			name:                   "other",
			authorizationHeader:    "other",
			skipHTTP2DOSMitigation: false,
			expectConnections:      reqs,
		},

		{
			name:                   "known skip=true",
			authorizationHeader:    "known",
			skipHTTP2DOSMitigation: true,
			expectConnections:      1,
		},
		{
			name:                   "error skip=true",
			authorizationHeader:    "error",
			skipHTTP2DOSMitigation: true,
			expectConnections:      1,
		},
		{
			name:                   "anonymous skip=true",
			authorizationHeader:    "anonymous",
			skipHTTP2DOSMitigation: true,
			expectConnections:      1,
		},
		{
			name:                   "anonymous_group skip=true",
			authorizationHeader:    "anonymous_group",
			skipHTTP2DOSMitigation: true,
			expectConnections:      1,
		},
		{
			name:                   "other skip=true",
			authorizationHeader:    "other",
			skipHTTP2DOSMitigation: true,
			expectConnections:      1,
		},
	}

	rootCAs := x509.NewCertPool()
	rootCAs.AddCert(s.Certificate())

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := func(t *testing.T, nextProto string, expectConnections uint64) {
				ctx := t.Context()
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.UnauthenticatedHTTP2DOSMitigation, !tc.skipHTTP2DOSMitigation)

				var localAddrs atomic.Uint64 // indicates how many TCP connection set up

				tlsConfig := &tls.Config{
					RootCAs:    rootCAs,
					NextProtos: []string{nextProto},
				}

				dailer := tls.Dialer{
					Config: tlsConfig,
				}

				tr := &http.Transport{
					TLSHandshakeTimeout: 10 * time.Second,
					TLSClientConfig:     tlsConfig,
					DialTLSContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
						conn, err := dailer.DialContext(ctx, network, addr)
						if err != nil {
							return nil, err
						}

						localAddrs.Add(1)

						return conn, nil
					},
				}

				tr.MaxIdleConnsPerHost = 1 // allow http1 to have keep alive connections open
				if nextProto == http2.NextProtoTLS {
					// Disable connection pooling to avoid additional connections
					// that cause the test to flake
					tr.MaxIdleConnsPerHost = -1
					if err := http2.ConfigureTransport(tr); err != nil {
						t.Fatal(err)
					}
				}

				client := &http.Client{
					Transport: tr,
				}

				for i := 0; i < reqs; i++ {
					req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.URL, nil)
					if err != nil {
						t.Fatal(err)
					}
					if len(tc.authorizationHeader) > 0 {
						req.Header.Set("Authorization", tc.authorizationHeader)
					}

					resp, err := client.Do(req)
					if err != nil {
						t.Fatal(err)
					}
					_, _ = io.Copy(io.Discard, resp.Body)
					_ = resp.Body.Close()
				}

				if expectConnections != localAddrs.Load() {
					t.Fatalf("expect TCP connection: %d, actual: %d", expectConnections, localAddrs.Load())
				}
			}

			t.Run(http2.NextProtoTLS, func(t *testing.T) {
				f(t, http2.NextProtoTLS, tc.expectConnections)
			})

			// http1 connection reuse occasionally flakes on CI, skipping for now
			// t.Run("http/1.1", func(t *testing.T) {
			// 	f(t, "http/1.1", 1)
			// })
		})
	}
}

func TestAuthenticationLatencyTracked(t *testing.T) {
	successHandlerCalled := false
	handler := WithAuthentication(
		http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
			// latency annotations should be added by the time the success handler is called
			annotations := genericapirequest.AuditAnnotationsFromLatencyTrackers(req.Context())
			if annotations["apiserver.latency.k8s.io/authentication"] == "" {
				t.Errorf("missing authentication latency annotation: %v", annotations)
			}
			successHandlerCalled = true
		}),
		authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
			return &authenticator.Response{User: &user.DefaultInfo{Name: "user"}}, true, nil
		}),
		http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			t.Error("unexpected call to error handler")
		}),
		nil,
		nil,
	)
	handler = WithLatencyTrackers(handler)
	handler = WithRequestInfo(handler, &fakeRequestResolver{})
	req, err := http.NewRequest(http.MethodGet, "/", nil)
	if err != nil {
		t.Fatal(err)
	}
	handler.ServeHTTP(httptest.NewRecorder(), req)
	if !successHandlerCalled {
		t.Fatal("no call to success handler")
	}
}
