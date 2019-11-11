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

package filters

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/component-base/metrics/legacyregistry"
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
		response    *authenticator.Response
		status      bool
		err         error
		apiAudience authenticator.Audiences
		want        string
	}{
		{
			desc: "auth ok",
			response: &authenticator.Response{
				User:              &user.DefaultInfo{Name: "admin"},
				AuthMethod:        "foo",
				AuthenticatorName: "bar",
			},
			status: true,
			want: `
				# HELP authenticated_user_requests [ALPHA] Counter of authenticated requests broken out by username.
				# TYPE authenticated_user_requests counter
				authenticated_user_requests{authenticator="bar",method="foo",username="admin"} 1
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{authenticator="bar",method="foo",result="success"} 1
				`,
		},
		{
			desc: "auth failed with error",
			err:  authenticator.Errorf("foo", "bar", "some error"),
			want: `
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{authenticator="bar",method="foo",result="error"} 1
				`,
		},
		{
			desc: "auth failed with status false",
			want: `
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{authenticator="unknown",method="unknown",result="failure"} 1
				`,
		},
		{
			desc: "auth failed due to audiences not intersecting",
			response: &authenticator.Response{
				User:              &user.DefaultInfo{Name: "admin"},
				AuthMethod:        "foo",
				AuthenticatorName: "bar",
				Audiences:         authenticator.Audiences{"audience-x"},
			},
			status:      true,
			apiAudience: authenticator.Audiences{"audience-y"},
			want: `
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{authenticator="bar",method="foo",result="non-intersecting-audiences"} 1
				`,
		},
	}

	// Since prometheus' gatherer is global, other tests may have updated metrics already, so
	// we need to reset them prior running this test.
	// This also implies that we can't run this test in parallel with other auth tests.
	authenticatedUserCounter.Reset()
	authenticatedAttemptsCounter.Reset()

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			defer authenticatedUserCounter.Reset()
			defer authenticatedAttemptsCounter.Reset()
			done := make(chan struct{})
			auth := WithAuthentication(
				http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
					close(done)
				}),
				authenticator.RequestFunc(func(_ *http.Request) (*authenticator.Response, bool, error) {
					return tt.response, tt.status, tt.err
				}),
				http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
					close(done)
				}),
				tt.apiAudience,
			)

			auth.ServeHTTP(httptest.NewRecorder(), &http.Request{})
			<-done

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestAuthenticatorNameFromError(t *testing.T) {
	testCases := []struct {
		desc string
		in   error
		want string
	}{
		{
			desc: "nil error",
			in:   nil,
			want: unknownAuthenticator,
		},
		{
			desc: "auth error",
			in:   authenticator.NewError("foo", "bar", errors.New("bar")),
			want: "bar",
		},
		{
			desc: "auth error with empty authenticator",
			in:   authenticator.NewError("foo", "", errors.New("bar")),
			want: unknownAuthenticator,
		},
		{
			desc: "non-auth error",
			in:   errors.New("bar"),
			want: unknownAuthenticator,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			if got := authenticatorNameFromError(tc.in); got != tc.want {
				t.Fatalf("Got %v, want %v for authentictor's name", got, tc.want)
			}
		})
	}
}

func TestAuthenticatorNameFromResponse(t *testing.T) {
	testCases := []struct {
		desc string
		in   *authenticator.Response
		want string
	}{
		{
			desc: "nil response",
			in:   nil,
			want: unknownAuthenticator,
		},
		{
			desc: "response with authenticator's name",
			in:   &authenticator.Response{AuthenticatorName: "foo"},
			want: "foo",
		},
		{
			desc: "response without authenticator's name",
			in:   &authenticator.Response{},
			want: unknownAuthenticator,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			if got := authenticatorNameFromResponse(tc.in); got != tc.want {
				t.Fatalf("Got %v, want %v for authentictor's name", got, tc.want)
			}
		})
	}
}
