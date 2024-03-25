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

package webhook

import (
	"context"
	"net/http"
	"testing"

	authorizationv1 "k8s.io/api/authorization/v1"
	"k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/cel"
)

func TestAuthorizerMetrics(t *testing.T) {
	scenarios := []struct {
		name                            string
		canceledRequest                 bool
		clientCert, clientKey, clientCA []byte
		serverCert, serverKey, serverCA []byte
		authzFakeServiceStatusCode      int
		authFakeServiceDeny             bool
		expectedRegisteredStatusCode    string
		expectEvalutionResult           string
		expectDurationResult            string
		expectFailOpenResult            string
		wantErr                         bool
	}{
		{
			name:       "happy path",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: caCert,
			expectedRegisteredStatusCode: "200",
			expectEvalutionResult:        "success",
			expectDurationResult:         "success",
			expectFailOpenResult:         "",
		},

		{
			name:       "timed out request",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: caCert,
			authzFakeServiceStatusCode:   http.StatusGatewayTimeout,
			expectedRegisteredStatusCode: "504",
			expectEvalutionResult:        "timeout",
			expectDurationResult:         "timeout",
			expectFailOpenResult:         "timeout",
		},

		{
			name:       "canceled request",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: caCert,
			canceledRequest:              true,
			expectedRegisteredStatusCode: "<error>",
			expectEvalutionResult:        "canceled",
			expectDurationResult:         "canceled",
			expectFailOpenResult:         "",
		},

		{
			name:       "an internal error returned from the webhook",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: caCert,
			authzFakeServiceStatusCode:   500,
			expectedRegisteredStatusCode: "500",
			expectEvalutionResult:        "error",
			expectDurationResult:         "error",
			expectFailOpenResult:         "error",
		},

		{
			name:       "incorrect client certificate used, the webhook not called, an error is recorded",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: badCACert,
			expectedRegisteredStatusCode: "<error>",
			expectEvalutionResult:        "error",
			expectDurationResult:         "error",
			expectFailOpenResult:         "error",
			wantErr:                      true,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			service := new(mockV1Service)
			service.statusCode = scenario.authzFakeServiceStatusCode
			if service.statusCode == 0 {
				service.statusCode = 200
			}
			service.reviewHook = func(*authorizationv1.SubjectAccessReview) {
				if scenario.canceledRequest {
					cancel()
				}
			}
			service.allow = !scenario.authFakeServiceDeny

			server, err := NewV1TestServer(service, scenario.serverCert, scenario.serverKey, scenario.serverCA)
			if err != nil {
				t.Errorf("%s: failed to create server: %v", scenario.name, err)
				return
			}
			defer server.Close()

			fakeAuthzMetrics := &fakeAuthorizerMetrics{}
			wh, err := newV1Authorizer(server.URL, scenario.clientCert, scenario.clientKey, scenario.clientCA, 0, fakeAuthzMetrics, []apiserver.WebhookMatchCondition{}, "")
			if err != nil {
				t.Error("failed to create client")
				return
			}

			attr := authorizer.AttributesRecord{User: &user.DefaultInfo{}}
			_, _, err = wh.Authorize(ctx, attr)
			if scenario.wantErr {
				if err == nil {
					t.Errorf("expected error making authorization request: %v", err)
				}
			}

			if fakeAuthzMetrics.totalCode != scenario.expectedRegisteredStatusCode {
				t.Errorf("incorrect status code recorded for RecordRequestTotal method, expected = %v, got %v", scenario.expectedRegisteredStatusCode, fakeAuthzMetrics.totalCode)
			}

			if fakeAuthzMetrics.latencyCode != scenario.expectedRegisteredStatusCode {
				t.Errorf("incorrect status code recorded for RecordRequestLatency method, expected = %v, got %v", scenario.expectedRegisteredStatusCode, fakeAuthzMetrics.latencyCode)
			}

			if fakeAuthzMetrics.evaluationsResult != scenario.expectEvalutionResult {
				t.Errorf("expected evaluationsResult %q, got %q", scenario.expectEvalutionResult, fakeAuthzMetrics.evaluationsResult)
			}
			if fakeAuthzMetrics.durationResult != scenario.expectDurationResult {
				t.Errorf("expected durationResult %q, got %q", scenario.expectDurationResult, fakeAuthzMetrics.durationResult)
			}
			if fakeAuthzMetrics.failOpenResult != scenario.expectFailOpenResult {
				t.Errorf("expected failOpenResult %q, got %q", scenario.expectFailOpenResult, fakeAuthzMetrics.failOpenResult)
			}
		})
	}
}

type fakeAuthorizerMetrics struct {
	totalCode string

	latency     float64
	latencyCode string

	evaluations       int
	evaluationsResult string

	duration       float64
	durationResult string

	failOpen       int
	failOpenResult string

	cel.NoopMatcherMetrics
}

func (f *fakeAuthorizerMetrics) RecordRequestTotal(_ context.Context, code string) {
	f.totalCode = code
}

func (f *fakeAuthorizerMetrics) RecordRequestLatency(_ context.Context, code string, latency float64) {
	f.latency = latency
	f.latencyCode = code
}

func (f *fakeAuthorizerMetrics) RecordWebhookEvaluation(ctx context.Context, name, result string) {
	f.evaluations += 1
	f.evaluationsResult = result
}
func (f *fakeAuthorizerMetrics) RecordWebhookDuration(ctx context.Context, name, result string, duration float64) {
	f.duration = duration
	f.durationResult = result
}
func (f *fakeAuthorizerMetrics) RecordWebhookFailOpen(ctx context.Context, name, result string) {
	f.failOpen += 1
	f.failOpenResult = result
}
