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

package webhook

import (
	"context"
	"encoding/pem"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"

	"golang.org/x/net/http2"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestWebhookClientConfig(t *testing.T) {
	cm, _ := NewClientManager([]schema.GroupVersion{})
	authInfoResolver, err := NewDefaultAuthenticationInfoResolver("")
	if err != nil {
		t.Fatal(err)
	}
	cm.SetAuthenticationInfoResolver(authInfoResolver)
	cm.SetServiceResolver(NewDefaultServiceResolver())

	tests := []struct {
		name             string
		url              string
		expectAllowHTTP2 bool
	}{
		{
			name:             "force http1",
			url:              "https://webhook.example.com",
			expectAllowHTTP2: false,
		},
		{
			name:             "allow http2 for localhost",
			url:              "https://localhost",
			expectAllowHTTP2: true,
		},
		{
			name:             "allow http2 for 127.0.0.1",
			url:              "https://127.0.0.1",
			expectAllowHTTP2: true,
		},
		{
			name:             "allow http2 for [::1]:0",
			url:              "https://[::1]",
			expectAllowHTTP2: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			cc := ClientConfig{
				URL: tc.url,
			}
			cfg, err := cm.hookClientConfig(cc)
			if err != nil {
				t.Fatal(err)
			}
			if tc.expectAllowHTTP2 && !allowHTTP2(cfg.NextProtos) {
				t.Errorf("expected allow http/2, got: %v", cfg.NextProtos)
			}
		})
	}
}

func allowHTTP2(nextProtos []string) bool {
	if len(nextProtos) == 0 {
		// the transport expressed no NextProto preference, allow
		return true
	}
	for _, p := range nextProtos {
		if p == http2.NextProtoTLS {
			// the transport explicitly allowed http/2
			return true
		}
	}
	// the transport explicitly set NextProtos and excluded http/2
	return false
}

type fakeAuthInfoResolver struct{}

func (f *fakeAuthInfoResolver) ClientConfigFor(server string) (*rest.Config, error) {
	return &rest.Config{
		TLSClientConfig: rest.TLSClientConfig{
			ServerName: "example.com",
		},
	}, nil
}

func (f *fakeAuthInfoResolver) ClientConfigForService(serviceName, namespace string, port int) (*rest.Config, error) {
	return &rest.Config{
		TLSClientConfig: rest.TLSClientConfig{
			ServerName: "example.com",
		},
	}, nil
}

// fakeDynamicServiceResolver returns the next endpoint in the list for each request.
type fakeDynamicServiceResolver struct {
	endpoints []*url.URL
	counter   int32
}

func (f *fakeDynamicServiceResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	val := atomic.AddInt32(&f.counter, 1) - 1
	if val >= int32(len(f.endpoints)) {
		val = int32(len(f.endpoints)) - 1
	}
	return f.endpoints[val], nil
}

// TestWebhookClientIdleConnectionIPReuse tests that the webhook client follow the resolver
// endpoint instead of reusing the previous endpoint when there are IP address changes.
func TestWebhookClientIdleConnectionIPReuse(t *testing.T) {
	tests := []struct {
		name                   string
		enableFeatureGate      bool
		expectedServerACalls   int32
		expectedServerBCalls   int32
		expectedSecondResponse string
	}{
		{
			name:                   "feature gate enabled - round-trip load balancing routes to Server B",
			enableFeatureGate:      true,
			expectedServerACalls:   1,
			expectedServerBCalls:   1,
			expectedSecondResponse: "ServerB",
		},
		{
			name:                   "feature gate disabled - dialer resolution caches to Server A",
			enableFeatureGate:      false,
			expectedServerACalls:   2,
			expectedServerBCalls:   0,
			expectedSecondResponse: "ServerA",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WebhookRoundTripLoadBalancing, tc.enableFeatureGate)

			var serverACalls int32
			serverA := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				atomic.AddInt32(&serverACalls, 1)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte("ServerA"))
			}))
			defer serverA.Close()

			var serverBCalls int32
			serverB := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				atomic.AddInt32(&serverBCalls, 1)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte("ServerB"))
			}))
			defer serverB.Close()

			urlA, err := url.Parse(serverA.URL)
			if err != nil {
				t.Fatal(err)
			}
			urlB, err := url.Parse(serverB.URL)
			if err != nil {
				t.Fatal(err)
			}

			// Combine CAs from both test servers
			var caBundle []byte
			for _, cert := range serverA.TLS.Certificates[0].Certificate {
				caBundle = append(caBundle, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert})...)
			}
			for _, cert := range serverB.TLS.Certificates[0].Certificate {
				caBundle = append(caBundle, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert})...)
			}

			resolver := &fakeDynamicServiceResolver{
				endpoints: []*url.URL{urlA, urlB},
			}

			cm, err := NewClientManager([]schema.GroupVersion{})
			if err != nil {
				t.Fatal(err)
			}
			cm.SetAuthenticationInfoResolver(&fakeAuthInfoResolver{})
			cm.SetServiceResolver(resolver)

			cc := ClientConfig{
				Name:     "test-webhook",
				CABundle: caBundle,
				Service: &ClientConfigService{
					Name:      "test-service",
					Namespace: "default",
					Port:      443,
				},
			}

			client, err := cm.HookClient(cc)
			if err != nil {
				t.Fatal(err)
			}

			// Request 1: Resolves to Server A
			req1 := client.Post().Body([]byte("test"))
			res1, err := req1.DoRaw(context.Background())
			if err != nil {
				t.Fatalf("First request failed: %v", err)
			}
			if string(res1) != "ServerA" {
				t.Errorf("Expected Response ServerA, got %s", string(res1))
			}

			// Request 2: Resolves to Server B if feature gate enabled, Server A if disabled
			req2 := client.Post().Body([]byte("test"))
			res2, err := req2.DoRaw(context.Background())
			if err != nil {
				t.Fatalf("Second request failed: %v", err)
			}
			if string(res2) != tc.expectedSecondResponse {
				t.Errorf("Expected Response %s, got %s", tc.expectedSecondResponse, string(res2))
			}

			if callsA := atomic.LoadInt32(&serverACalls); callsA != tc.expectedServerACalls {
				t.Errorf("Expected %d calls to Server A, got %d", tc.expectedServerACalls, callsA)
			}
			if callsB := atomic.LoadInt32(&serverBCalls); callsB != tc.expectedServerBCalls {
				t.Errorf("Expected %d calls to Server B, got %d", tc.expectedServerBCalls, callsB)
			}
		})
	}
}
