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
)

func TestAuthenticatorMetrics(t *testing.T) {
	scenarios := []struct {
		name                            string
		clientCert, clientKey, clientCA []byte
		serverCert, serverKey, serverCA []byte
		authnFakeServiceStatusCode      int
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
			authnFakeServiceStatusCode:   500,
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
			service.statusCode = scenario.authnFakeServiceStatusCode
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

			fakeAuthnMetrics := &fakeAuthenticatorMetrics{}
			authnMetrics := AuthenticatorMetrics{
				RecordRequestTotal:   fakeAuthnMetrics.RequestTotal,
				RecordRequestLatency: fakeAuthnMetrics.RequestLatency,
			}
			wh, err := newV1TokenAuthenticator(server.URL, scenario.clientCert, scenario.clientKey, scenario.clientCA, 0, nil, authnMetrics)
			if err != nil {
				t.Error("failed to create client")
				return
			}

			_, _, err = wh.AuthenticateToken(context.Background(), "t0k3n")
			if scenario.wantErr {
				if err == nil {
					t.Errorf("expected error making authorization request: %v", err)
				}
			}

			if fakeAuthnMetrics.totalCode != scenario.expectedRegisteredStatusCode {
				t.Errorf("incorrect status code recorded for RecordRequestTotal method, expected = %v, got %v", scenario.expectedRegisteredStatusCode, fakeAuthnMetrics.totalCode)
			}

			if fakeAuthnMetrics.latencyCode != scenario.expectedRegisteredStatusCode {
				t.Errorf("incorrect status code recorded for RecordRequestLatency method, expected = %v, got %v", scenario.expectedRegisteredStatusCode, fakeAuthnMetrics.latencyCode)
			}
		})
	}
}

type fakeAuthenticatorMetrics struct {
	totalCode string

	latency     float64
	latencyCode string
}

func (f *fakeAuthenticatorMetrics) RequestTotal(_ context.Context, code string) {
	f.totalCode = code
}

func (f *fakeAuthenticatorMetrics) RequestLatency(_ context.Context, code string, latency float64) {
	f.latency = latency
	f.latencyCode = code
}
