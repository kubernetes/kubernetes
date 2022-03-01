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
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func TestAuthorizerMetrics(t *testing.T) {
	scenarios := []struct {
		name                            string
		clientCert, clientKey, clientCA []byte
		serverCert, serverKey, serverCA []byte
		authzFakeServiceStatusCode      int
		authFakeServiceDeny             bool
		expectedRegisteredStatusCode    string
		wantErr                         bool
	}{
		{
			name:       "happy path",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: caCert,
			expectedRegisteredStatusCode: "200",
		},

		{
			name:       "an internal error returned from the webhook",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: caCert,
			authzFakeServiceStatusCode:   500,
			expectedRegisteredStatusCode: "500",
		},

		{
			name:       "incorrect client certificate used, the webhook not called, an error is recorded",
			clientCert: clientCert, clientKey: clientKey, clientCA: caCert,
			serverCert: serverCert, serverKey: serverKey, serverCA: badCACert,
			expectedRegisteredStatusCode: "<error>",
			wantErr:                      true,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			service := new(mockV1Service)
			service.statusCode = scenario.authzFakeServiceStatusCode
			if service.statusCode == 0 {
				service.statusCode = 200
			}
			service.allow = !scenario.authFakeServiceDeny

			server, err := NewV1TestServer(service, scenario.serverCert, scenario.serverKey, scenario.serverCA)
			if err != nil {
				t.Errorf("%s: failed to create server: %v", scenario.name, err)
				return
			}
			defer server.Close()

			fakeAuthzMetrics := &fakeAuthorizerMetrics{}
			authzMetrics := AuthorizerMetrics{
				RecordRequestTotal:   fakeAuthzMetrics.RequestTotal,
				RecordRequestLatency: fakeAuthzMetrics.RequestLatency,
			}
			wh, err := newV1Authorizer(server.URL, scenario.clientCert, scenario.clientKey, scenario.clientCA, 0, authzMetrics)
			if err != nil {
				t.Error("failed to create client")
				return
			}

			attr := authorizer.AttributesRecord{User: &user.DefaultInfo{}}
			_, _, err = wh.Authorize(context.Background(), attr)
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
		})
	}
}

type fakeAuthorizerMetrics struct {
	totalCode string

	latency     float64
	latencyCode string
}

func (f *fakeAuthorizerMetrics) RequestTotal(_ context.Context, code string) {
	f.totalCode = code
}

func (f *fakeAuthorizerMetrics) RequestLatency(_ context.Context, code string, latency float64) {
	f.latency = latency
	f.latencyCode = code
}
