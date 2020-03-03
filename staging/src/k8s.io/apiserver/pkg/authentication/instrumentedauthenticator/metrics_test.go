/*
Copyright 2020 The Kubernetes Authors.

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

package instrumentedauthenticator

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/request/union"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/component-base/metrics/testutil"
)

func TestMetrics(t *testing.T) {
	// Excluding authentication_duration_seconds since it is difficult to predict its values.
	metrics := []string{
		"authenticated_user_requests",
		"authentication_attempts",
	}

	testCases := []struct {
		desc        string
		want        string
		request     authenticator.Request
		apiAudience authenticator.Audiences
	}{
		{
			desc: "auth ok",
			request: WrapRequest(
				authenticator.RequestFunc(
					func(_ *http.Request) (*authenticator.Response, bool, error) {
						return &authenticator.Response{User: &user.DefaultInfo{Name: "admin"}}, true, nil
					}),
				"instrumented-auth"),
			want: `
				# HELP authenticated_user_requests [ALPHA] Counter of authenticated requests broken out by username.
				# TYPE authenticated_user_requests counter
				authenticated_user_requests{authenticator="instrumented-auth", username="admin"} 1
		    # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
		    # TYPE authentication_attempts counter
		    authentication_attempts{authenticator="instrumented-auth", result="success"} 1
				`,
		},
		{
			desc: "auth failed with error",
			request: WrapRequest(
				authenticator.RequestFunc(
					func(_ *http.Request) (*authenticator.Response, bool, error) {
						return nil, false, errors.New("some error")
					}),
				"instrumented-auth"),
			want: `
		    # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
		    # TYPE authentication_attempts counter
		    authentication_attempts{authenticator="instrumented-auth", result="error",} 1
				`,
		},
		{
			desc: "auth failed with status false",
			request: WrapRequest(
				authenticator.RequestFunc(
					func(_ *http.Request) (*authenticator.Response, bool, error) {
						return &authenticator.Response{User: &user.DefaultInfo{Name: "admin"}}, false, nil
					}),
				"instrumented-auth"),
			want: `
		    # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
		    # TYPE authentication_attempts counter
		    authentication_attempts{authenticator="instrumented-auth", result="failure"} 1
				`,
		},
		{
			desc: "auth failed due to audiences not intersecting",
			request: WrapRequest(
				authenticator.RequestFunc(
					func(_ *http.Request) (*authenticator.Response, bool, error) {
						return &authenticator.Response{
								User:      &user.DefaultInfo{Name: "admin"},
								Audiences: authenticator.Audiences{"audience-x"}},
							true,
							nil
					}),
				"instrumented-auth"),
			apiAudience: authenticator.Audiences{"audience-y"},
			want: `
		    # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
		    # TYPE authentication_attempts counter
		    authentication_attempts{authenticator="instrumented-auth", result="error"} 1
				`,
		},
		{
			desc: "audiences not supplied in the response",
			request: WrapRequest(
				authenticator.RequestFunc(
					func(_ *http.Request) (*authenticator.Response, bool, error) {
						return &authenticator.Response{
								User: &user.DefaultInfo{Name: "admin"},
							},
							true,
							nil
					}),
				"instrumented-auth"),
			apiAudience: authenticator.Audiences{"audience-y"},
			want: `
		    # HELP authenticated_user_requests [ALPHA] Counter of authenticated requests broken out by username.
				# TYPE authenticated_user_requests counter
				authenticated_user_requests{authenticator="instrumented-auth", username="admin"} 1
		    # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
		    # TYPE authentication_attempts counter
		    authentication_attempts{authenticator="instrumented-auth", result="success"} 1
				`,
		},
		{
			desc: "audiences not supplied to the handler",
			request: WrapRequest(
				authenticator.RequestFunc(
					func(_ *http.Request) (*authenticator.Response, bool, error) {
						return &authenticator.Response{
								User:      &user.DefaultInfo{Name: "admin"},
								Audiences: authenticator.Audiences{"audience-x"},
							},
							true,
							nil
					}),
				"instrumented-auth"),
			want: `
		    # HELP authenticated_user_requests [ALPHA] Counter of authenticated requests broken out by username.
				# TYPE authenticated_user_requests counter
				authenticated_user_requests{authenticator="instrumented-auth", username="admin"} 1
		    # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
		    # TYPE authentication_attempts counter
		    authentication_attempts{authenticator="instrumented-auth", result="success"} 1
				`,
		},
	}

	registry := testutil.NewFakeKubeRegistry("1.18.0")
	registry.MustRegister(authenticatedAttemptsCounter)
	registry.MustRegister(authenticatedUserCounter)

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			defer authenticatedUserCounter.Reset()
			defer authenticatedAttemptsCounter.Reset()

			request, _ := http.NewRequestWithContext(
				authenticator.WithAudiences(
					context.Background(),
					tt.apiAudience),
				http.MethodPost,
				"http://foo",
				nil)
			tt.request.AuthenticateRequest(request)
			if err := testutil.GatherAndCompare(registry, strings.NewReader(tt.want), metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestUnionAuthMetrics(t *testing.T) {
	metrics := []string{
		"authenticated_user_requests",
		"authentication_attempts",
	}

	testCases := []struct {
		desc        string
		want        string
		request     authenticator.Request
		apiAudience authenticator.Audiences
	}{
		{
			desc: "union auth - all authenticators fail",
			request: union.New(
				WrapRequest(
					authenticator.RequestFunc(
						func(_ *http.Request) (*authenticator.Response, bool, error) {
							return &authenticator.Response{
									User: &user.DefaultInfo{Name: "admin"},
								},
								false,
								nil
						}),
					"instrumented-auth-01"),
				WrapRequest(
					authenticator.RequestFunc(
						func(_ *http.Request) (*authenticator.Response, bool, error) {
							return &authenticator.Response{
									User: &user.DefaultInfo{Name: "admin"},
								},
								false,
								nil
						}),
					"instrumented-auth-02"),
			),
			want: `
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{authenticator="instrumented-auth-01",result="failure"} 1
        authentication_attempts{authenticator="instrumented-auth-02",result="failure"} 1
				`,
		},
		{
			desc: "union auth - first authenticator fails - second succeeds",
			request: union.New(
				WrapRequest(
					authenticator.RequestFunc(
						func(_ *http.Request) (*authenticator.Response, bool, error) {
							return &authenticator.Response{
									User: &user.DefaultInfo{Name: "admin"},
								},
								false,
								nil
						}),
					"instrumented-auth-01"),
				WrapRequest(
					authenticator.RequestFunc(
						func(_ *http.Request) (*authenticator.Response, bool, error) {
							return &authenticator.Response{
									User: &user.DefaultInfo{Name: "admin"},
								},
								true,
								nil
						}),
					"instrumented-auth-02"),
			),
			want: `
       # HELP authenticated_user_requests [ALPHA] Counter of authenticated requests broken out by username.
        # TYPE authenticated_user_requests counter
        authenticated_user_requests{authenticator="instrumented-auth-02",username="admin"} 1
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{authenticator="instrumented-auth-01",result="failure"} 1
        authentication_attempts{authenticator="instrumented-auth-02",result="success"} 1
				`,
		},
		{
			desc: "union auth - first authenticator succeeds",
			request: union.New(
				WrapRequest(
					authenticator.RequestFunc(
						func(_ *http.Request) (*authenticator.Response, bool, error) {
							return &authenticator.Response{
									User: &user.DefaultInfo{Name: "admin"},
								},
								true,
								nil
						}),
					"instrumented-auth-01"),
				WrapRequest(
					authenticator.RequestFunc(
						func(_ *http.Request) (*authenticator.Response, bool, error) {
							return &authenticator.Response{
									User: &user.DefaultInfo{Name: "admin"},
								},
								true,
								nil
						}),
					"instrumented-auth-02"),
			),
			want: `
        # HELP authenticated_user_requests [ALPHA] Counter of authenticated requests broken out by username.
        # TYPE authenticated_user_requests counter
        authenticated_user_requests{authenticator="instrumented-auth-01",username="admin"} 1
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{authenticator="instrumented-auth-01",result="success"} 1
				`,
		},
		{
			desc: "union auth - request is not compatible with any of the authenticators within a union",
			request: union.New(
				WrapRequest(
					authenticator.RequestFunc(
						func(_ *http.Request) (*authenticator.Response, bool, error) {
							return nil,
								false,
								nil
						}),
					"instrumented-auth-01"),
				WrapRequest(
					authenticator.RequestFunc(
						func(_ *http.Request) (*authenticator.Response, bool, error) {
							return nil,
								false,
								nil
						}),
					"instrumented-auth-02"),
			),
		},
	}

	registry := testutil.NewFakeKubeRegistry("1.18.0")
	registry.MustRegister(authenticatedAttemptsCounter)
	registry.MustRegister(authenticatedUserCounter)

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			defer authenticatedUserCounter.Reset()
			defer authenticatedAttemptsCounter.Reset()

			tt.request.AuthenticateRequest(&http.Request{})
			if err := testutil.GatherAndCompare(registry, strings.NewReader(tt.want), metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
