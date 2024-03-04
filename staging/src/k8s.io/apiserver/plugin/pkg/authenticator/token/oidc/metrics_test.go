/*
Copyright 2024 The Kubernetes Authors.

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

package oidc

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

const (
	testIssuer = "testIssuer"
)

func TestRecordAuthenticationLatency(t *testing.T) {
	tests := []struct {
		name            string
		authenticator   authenticator.Token
		generateMetrics func()
		expectedValue   string
	}{
		{
			name:          "success",
			authenticator: &dummyAuthenticator{response: &authenticator.Response{}, ok: true},
			expectedValue: `
        # HELP apiserver_authentication_jwt_authenticator_latency_seconds [ALPHA] Latency of jwt authentication operations in seconds. This is the time spent authenticating a token for cache miss only (i.e. when the token is not found in the cache).
        # TYPE apiserver_authentication_jwt_authenticator_latency_seconds histogram
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success",le="0.001"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success",le="0.005"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success",le="0.01"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success",le="0.025"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success",le="0.05"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success",le="0.1"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success",le="0.25"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success",le="0.5"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success",le="1"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success",le="2.5"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success",le="5"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success",le="10"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success",le="+Inf"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_sum{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success"} 1e-09
        apiserver_authentication_jwt_authenticator_latency_seconds_count{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="success"} 1
		`,
		},
		{
			name:          "error",
			authenticator: &dummyAuthenticator{response: &authenticator.Response{}, ok: false, err: fmt.Errorf("error")},
			expectedValue: `
        # HELP apiserver_authentication_jwt_authenticator_latency_seconds [ALPHA] Latency of jwt authentication operations in seconds. This is the time spent authenticating a token for cache miss only (i.e. when the token is not found in the cache).
        # TYPE apiserver_authentication_jwt_authenticator_latency_seconds histogram
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure",le="0.001"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure",le="0.005"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure",le="0.01"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure",le="0.025"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure",le="0.05"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure",le="0.1"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure",le="0.25"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure",le="0.5"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure",le="1"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure",le="2.5"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure",le="5"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure",le="10"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_bucket{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure",le="+Inf"} 1
        apiserver_authentication_jwt_authenticator_latency_seconds_sum{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure"} 1e-09
        apiserver_authentication_jwt_authenticator_latency_seconds_count{jwt_issuer_hash="sha256:29b34beedc55b972f2428f21bc588f9d38e5e8f7a7af825486e7bb4fd9caa2ad",result="failure"} 1
		`,
		},
		{
			name:          "no metrics when issuer doesn't match",
			authenticator: &dummyAuthenticator{response: &authenticator.Response{}, ok: false, err: nil},
			expectedValue: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			jwtAuthenticatorLatencyMetric.Reset()
			RegisterMetrics()

			a := newInstrumentedAuthenticatorWithClock(testIssuer, tt.authenticator, dummyClock{})
			_, _, _ = a.AuthenticateToken(context.Background(), "token")

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.expectedValue), "apiserver_authentication_jwt_authenticator_latency_seconds"); err != nil {
				t.Fatal(err)
			}
		})
	}
}

type dummyAuthenticator struct {
	response *authenticator.Response
	ok       bool
	err      error
}

func (a *dummyAuthenticator) AuthenticateToken(ctx context.Context, token string) (*authenticator.Response, bool, error) {
	return a.response, a.ok, a.err
}

type dummyClock struct {
}

func (d dummyClock) Now() time.Time {
	return time.Now()
}

func (d dummyClock) Since(t time.Time) time.Duration {
	return time.Duration(1)
}
