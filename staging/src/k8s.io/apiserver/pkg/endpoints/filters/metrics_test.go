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
				User: &user.DefaultInfo{Name: "admin"},
			},
			status: true,
			want: `
				# HELP authenticated_user_requests [ALPHA] Counter of authenticated requests broken out by username.
				# TYPE authenticated_user_requests counter
				authenticated_user_requests{username="admin"} 1
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{result="success"} 1
				`,
		},
		{
			desc: "auth failed with error",
			err:  errors.New("some error"),
			want: `
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{result="error"} 1
				`,
		},
		{
			desc: "auth failed with status false",
			want: `
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{result="failure"} 1
				`,
		},
		{
			desc: "auth failed due to audiences not intersecting",
			response: &authenticator.Response{
				User:      &user.DefaultInfo{Name: "admin"},
				Audiences: authenticator.Audiences{"audience-x"},
			},
			status:      true,
			apiAudience: authenticator.Audiences{"audience-y"},
			want: `
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{result="error"} 1
				`,
		},
		{
			desc: "audiences not supplied in the response",
			response: &authenticator.Response{
				User: &user.DefaultInfo{Name: "admin"},
			},
			status:      true,
			apiAudience: authenticator.Audiences{"audience-y"},
			want: `
        # HELP authenticated_user_requests [ALPHA] Counter of authenticated requests broken out by username.
				# TYPE authenticated_user_requests counter
				authenticated_user_requests{username="admin"} 1
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{result="success"} 1
				`,
		},
		{
			desc: "audiences not supplied to the handler",
			response: &authenticator.Response{
				User:      &user.DefaultInfo{Name: "admin"},
				Audiences: authenticator.Audiences{"audience-x"},
			},
			status: true,
			want: `
        # HELP authenticated_user_requests [ALPHA] Counter of authenticated requests broken out by username.
				# TYPE authenticated_user_requests counter
				authenticated_user_requests{username="admin"} 1
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{result="success"} 1
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
