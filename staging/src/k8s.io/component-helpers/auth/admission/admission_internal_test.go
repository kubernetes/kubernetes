/*
Copyright The Kubernetes Authors.

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

package admission

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
)

func TestHandlerHealthCheckDelegates(t *testing.T) {
	var status atomic.Int64
	status.Store(http.StatusServiceUnavailable)
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != localJWKSPath {
			t.Errorf("probe path = %q, want %q", r.URL.Path, localJWKSPath)
		}
		w.WriteHeader(int(status.Load()))
		_, _ = w.Write([]byte("not a JWKS"))
	}))
	defer server.Close()
	handler := &Handler{verifier: &Verifier{client: server.Client(), jwksURL: server.URL + localJWKSPath}}
	if err := handler.HealthCheck(); err == nil {
		t.Fatal("HealthCheck succeeded for an unavailable endpoint")
	}
	status.Store(http.StatusOK)
	if err := handler.HealthCheck(); err != nil {
		t.Fatalf("HealthCheck after endpoint recovery: %v", err)
	}
}

func TestWithTokenVerificationConstruction(t *testing.T) {
	var requests atomic.Int32
	var server *httptest.Server
	server = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/.well-known/openid-configuration" {
			_ = json.NewEncoder(w).Encode(map[string]any{"issuer": server.URL, "jwks_uri": server.URL + localJWKSPath})
			return
		}
		if r.URL.Path != localJWKSPath {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"keys":[]}`))
	}))
	defer server.Close()
	var optionsApplied int
	option := func(b *builder) {
		optionsApplied++
		b.remoteSet = true
		b.issuer = server.URL
		b.audience = "https://webhook.example/validate"
		b.httpClient = server.Client()
	}
	if h, err := WithTokenVerification(context.Background(), nil, option); h != nil || err == nil {
		t.Fatalf("nil callback = (%v, %v), want (nil, error)", h, err)
	}
	if optionsApplied != 0 || requests.Load() != 0 {
		t.Fatal("nil callback applied options or performed I/O")
	}
	h, err := WithTokenVerification(context.Background(), func(context.Context, *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		return &admissionv1.AdmissionResponse{Allowed: true}
	}, option)
	if err != nil || h == nil {
		t.Fatalf("construction = (%v, %v)", h, err)
	}
	if optionsApplied != 1 || requests.Load() != 2 {
		t.Fatalf("options applied %d times, endpoint requests %d; want 1 and 2", optionsApplied, requests.Load())
	}
}
