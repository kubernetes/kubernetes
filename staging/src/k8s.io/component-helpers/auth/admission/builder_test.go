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

package admission_test

import (
	"context"
	"encoding/json"
	"encoding/pem"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/client-go/rest"
	"k8s.io/component-helpers/auth/admission"
)

func TestWithTokenVerificationConstruction(t *testing.T) {
	var requests atomic.Int32
	var server *httptest.Server
	server = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/.well-known/openid-configuration" {
			_ = json.NewEncoder(w).Encode(map[string]any{"issuer": server.URL, "jwks_uri": server.URL + "/openid/v1/jwks"})
			return
		}
		if r.URL.Path != "/openid/v1/jwks" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"keys":[]}`))
	}))
	defer server.Close()
	var optionsApplied int
	option := admission.OptionForTest(func(issuer, audience *string, client **http.Client) {
		optionsApplied++
		*issuer = server.URL
		*audience = "https://webhook.example/validate"
		*client = server.Client()
	})
	if h, err := admission.WithTokenVerification(context.Background(), nil, option); h != nil || err == nil {
		t.Fatalf("nil callback = (%v, %v), want (nil, error)", h, err)
	}
	if optionsApplied != 0 || requests.Load() != 0 {
		t.Fatal("nil callback applied options or performed I/O")
	}
	h, err := admission.WithTokenVerification(context.Background(), func(context.Context, *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		return &admissionv1.AdmissionResponse{Allowed: true}
	}, option)
	if err != nil || h == nil {
		t.Fatalf("construction = (%v, %v)", h, err)
	}
	if optionsApplied != 1 || requests.Load() != 1 {
		t.Fatalf("options applied %d times, discovery requests %d; want 1 and 1", optionsApplied, requests.Load())
	}
}

func TestBuildInClusterServiceAudience(t *testing.T) {
	namespaceFile := filepath.Join(t.TempDir(), "namespace")
	if err := os.WriteFile(namespaceFile, []byte("webhooks\n"), 0600); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name    string
		svc     string
		path    string
		port    uint16
		want    string
		wantErr bool
	}{
		{name: "default Service port and path written explicitly", svc: "hook", path: "/", port: 443, want: "https://hook.webhooks.svc:443/"},
		{name: "explicit Service port and path", svc: "hook", path: "/validate", port: 9443, want: "https://hook.webhooks.svc:9443/validate"},
		{name: "trailing slash retained", svc: "hook", path: "/validate/", port: 443, want: "https://hook.webhooks.svc:443/validate/"},
		{name: "other Service in same namespace is syntactically valid", svc: "dinner", path: "/validate", port: 8443, want: "https://dinner.webhooks.svc:8443/validate"},
		{name: "empty Service name", svc: "", path: "/validate", port: 443, wantErr: true},
		{name: "invalid Service name", svc: "bad_name", path: "/validate", port: 443, wantErr: true},
		{name: "dotted Service name", svc: "hook.webhooks.svc", path: "/validate", port: 443, wantErr: true},
		{name: "port zero", svc: "hook", path: "/validate", port: 0, wantErr: true},
		{name: "empty path", svc: "hook", path: "", port: 443, wantErr: true},
		{name: "relative path without leading slash", svc: "hook", path: "validate", port: 443, wantErr: true},
		{name: "scheme-relative path", svc: "hook", path: "//evil.example/validate", port: 443, wantErr: true},
		{name: "query", svc: "hook", path: "/validate?probe=true", port: 443, wantErr: true},
		{name: "empty query", svc: "hook", path: "/validate?", port: 443, wantErr: true},
		{name: "fragment", svc: "hook", path: "/validate#probe", port: 443, wantErr: true},
		{name: "empty fragment", svc: "hook", path: "/validate#", port: 443, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := admission.BuildInClusterServiceAudienceForTest(tc.svc, tc.path, tc.port, namespaceFile)
			if tc.wantErr {
				if err == nil || !strings.Contains(err.Error(), "WithServiceAudience") {
					t.Fatalf("validation = (%q, %v), want a helpful WithServiceAudience error", got, err)
				}
				return
			}
			if err != nil || got != tc.want {
				t.Fatalf("buildInClusterServiceAudience = (%q, %v), want (%q, nil)", got, err, tc.want)
			}
		})
	}

	for _, tc := range []struct {
		name    string
		content *string
	}{
		{name: "missing namespace file"},
		{name: "empty projected namespace", content: new("")},
		{name: "invalid projected namespace", content: new("bad_namespace")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "namespace")
			if tc.content != nil {
				if err := os.WriteFile(path, []byte(*tc.content), 0600); err != nil {
					t.Fatal(err)
				}
			}
			// A missing projection must not silently become the default namespace.
			if _, err := admission.BuildInClusterServiceAudienceForTest("hook", "/validate", 443, path); err == nil || !strings.Contains(err.Error(), "projected Pod namespace") {
				t.Fatalf("namespace validation = %v, want projected namespace error", err)
			}
		})
	}
}

func TestInClusterAudienceValidationPrecedesBootstrap(t *testing.T) {
	v, err := admission.NewVerifierForTest(context.Background(), admission.WithServiceAudience("hook", "/validate", 0))
	if v != nil || err == nil || !strings.Contains(err.Error(), "WithServiceAudience") {
		t.Fatalf("newVerifier invalid audience = (%v, %v), want an audience error", v, err)
	}
	h, err := admission.WithTokenVerification(context.Background(), func(context.Context, *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		return &admissionv1.AdmissionResponse{Allowed: true}
	}, admission.WithServiceAudience("hook", "/validate", 0))
	if h != nil || err == nil || !strings.Contains(err.Error(), "WithServiceAudience") {
		t.Fatalf("WithTokenVerification invalid audience = (%v, %v), want an audience error", h, err)
	}
}

// TestWithServiceAudience_ConflictsWithIssuer proves WithServiceAudience and
// WithIssuer cannot be combined: the audience would be set two ways, so the
// builder fails closed at construction rather than applying a silent precedence.
func TestWithServiceAudience_ConflictsWithIssuer(t *testing.T) {
	ts := newOIDCTestServer(t)

	v, err := admission.NewVerifierForTest(context.Background(),
		admission.WithIssuer(ts.issuer, testAudience, ts.client()),
		admission.WithServiceAudience("hook", "/validate", 443))
	if err == nil {
		t.Fatalf("expected a construction error combining WithServiceAudience and WithIssuer, got verifier %v", v)
	}
	if v != nil {
		t.Errorf("expected a nil verifier on a construction error, got %v", v)
	}
	if !strings.Contains(err.Error(), "WithServiceAudience cannot be combined with WithIssuer") {
		t.Errorf("error = %q, want it to mention the WithServiceAudience/WithIssuer conflict", err)
	}
}

// TestWithServiceAudience_EmptyIsError proves an explicit but incomplete Service
// audience is a hard construction error, never a verifier built with an
// audience-skipping audience.
func TestWithServiceAudience_EmptyIsError(t *testing.T) {
	v, err := admission.NewVerifierForTest(context.Background(), admission.WithServiceAudience("", "", 0))
	if err == nil || v != nil {
		t.Fatalf("NewVerifier with empty Service audience = (%v, %v), want (nil, error)", v, err)
	}
}

func TestInClusterRequiresTrustedAudienceBeforeBootstrap(t *testing.T) {
	admit := func(context.Context, *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		return &admissionv1.AdmissionResponse{Allowed: true}
	}
	for _, tc := range []struct {
		name string
		opts []admission.Option
	}{
		{name: "no options"},
		{name: "empty audience", opts: []admission.Option{admission.WithServiceAudience("", "", 0)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, err := admission.WithTokenVerification(context.Background(), admit, tc.opts...)
			if h != nil || err == nil || !strings.Contains(err.Error(), "WithServiceAudience") {
				t.Fatalf("WithTokenVerification = (%v, %v), want audience error", h, err)
			}
			v, err := admission.NewVerifierForTest(context.Background(), tc.opts...)
			if v != nil || err == nil || !strings.Contains(err.Error(), "WithServiceAudience") {
				t.Fatalf("NewVerifier = (%v, %v), want audience error", v, err)
			}
		})
	}
}

func TestLocalVerificationUsesProjectedIssuerAndTransport(t *testing.T) {
	ts := newOIDCTestServer(t)
	const projectedIssuer = "https://projected-issuer.example"
	projectedClaims := ts.baseClaims()
	projectedClaims["iss"] = projectedIssuer
	projectedClaims["aud"] = []string{"https://kubernetes.default.svc"}
	delete(projectedClaims, "kubernetes.io")
	projected := ts.sign(t, projectedClaims)
	tokenFile := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(tokenFile, []byte(projected), 0600); err != nil {
		t.Fatal(err)
	}
	var requests atomic.Int32
	config := &rest.Config{
		Host:            ts.issuer + "/",
		BearerTokenFile: tokenFile,
		TLSClientConfig: rest.TLSClientConfig{
			CAData: pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: ts.server.Certificate().Raw}),
		},
		WrapTransport: func(next http.RoundTripper) http.RoundTripper {
			return requestTransportFunc(func(r *http.Request) (*http.Response, error) {
				requests.Add(1)
				if r.URL.Path != "/openid/v1/jwks" {
					t.Errorf("local verification requested %q", r.URL.Path)
				}
				if r.Header.Get("Authorization") != "Bearer "+projected {
					t.Error("local JWKS request did not use the projected credential")
				}
				return next.RoundTrip(r)
			})
		},
	}
	v, err := admission.NewLocalVerifierForTest(context.Background(), config, testAudience)
	if err != nil {
		t.Fatal(err)
	}
	claims := ts.baseClaims()
	claims["iss"] = projectedIssuer
	if err := v.Verify(context.Background(), admissionReview(testGroup).Request, ts.sign(t, claims)); err != nil {
		t.Fatalf("projected issuer: %v", err)
	}
	if err := v.Verify(context.Background(), admissionReview(testGroup).Request, ts.sign(t, ts.baseClaims())); err == nil {
		t.Fatal("incoming token selected the expected issuer")
	}
	if requests.Load() == 0 {
		t.Fatal("local verification did not fetch signing keys")
	}
	if config.BearerTokenFile != tokenFile || config.BearerToken != "" {
		t.Fatal("constructor mutated the supplied REST configuration")
	}
	t.Run("static BearerToken without BearerTokenFile is rejected", func(t *testing.T) {
		staticOnly := rest.CopyConfig(config)
		staticOnly.BearerToken = projected
		staticOnly.BearerTokenFile = ""
		before := requests.Load()
		if verifier, err := admission.NewLocalVerifierForTest(context.Background(), staticOnly, testAudience); err == nil || verifier != nil {
			t.Fatalf("static BearerToken bootstrap = (%v, %v), want error requiring BearerTokenFile", verifier, err)
		}
		if requests.Load() != before {
			t.Fatal("invalid local bootstrap performed network I/O")
		}
	})
	t.Run("malformed projected token", func(t *testing.T) {
		badFile := filepath.Join(t.TempDir(), "bad-token")
		if err := os.WriteFile(badFile, []byte("opaque"), 0600); err != nil {
			t.Fatal(err)
		}
		bad := rest.CopyConfig(config)
		bad.BearerTokenFile = badFile
		before := requests.Load()
		if verifier, err := admission.NewLocalVerifierForTest(context.Background(), bad, testAudience); err == nil || verifier != nil {
			t.Fatalf("local bootstrap = (%v, %v)", verifier, err)
		}
		if requests.Load() != before {
			t.Fatal("invalid local bootstrap performed network I/O")
		}
	})
	badCA := rest.CopyConfig(config)
	badCA.CAData = []byte("not a certificate")
	if verifier, err := admission.NewLocalVerifierForTest(context.Background(), badCA, testAudience); err == nil || verifier != nil {
		t.Fatalf("invalid client-go CA configuration = (%v, %v)", verifier, err)
	}
}

func TestRemoteIssuerConfiguration(t *testing.T) {
	for _, issuer := range []string{
		"http://issuer.example", "https://user:password@issuer.example", "https://issuer.example?query=value", "https://issuer.example#fragment",
	} {
		t.Run(issuer, func(t *testing.T) {
			if verifier, err := admission.NewVerifierForTest(context.Background(), admission.WithIssuer(issuer, testAudience, nil)); err == nil || verifier != nil {
				t.Fatalf("invalid remote issuer = (%v, %v)", verifier, err)
			}
		})
	}
	var keysRequested atomic.Int32
	var server *httptest.Server
	server = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/.well-known/openid-configuration" {
			_ = json.NewEncoder(w).Encode(map[string]any{"issuer": "https://other.example", "jwks_uri": server.URL + "/keys"})
			return
		}
		keysRequested.Add(1)
		_, _ = w.Write([]byte(`{"keys":[]}`))
	}))
	defer server.Close()
	if verifier, err := admission.NewVerifierForTest(context.Background(), admission.WithIssuer(server.URL, testAudience, server.Client())); err == nil || verifier != nil {
		t.Fatalf("mismatched discovery issuer = (%v, %v)", verifier, err)
	}
	if keysRequested.Load() != 0 {
		t.Fatal("mismatched issuer proceeded to the signing-key endpoint")
	}
}

func TestRemoteVerificationBoundsBackgroundFetch(t *testing.T) {
	issuer := newOIDCTestServer(t)
	keys := testJWKS(t, issuer.server, "/keys")
	var block atomic.Bool
	started := make(chan struct{})
	finished := make(chan struct{})
	var server *httptest.Server
	server = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/.well-known/openid-configuration" {
			_ = json.NewEncoder(w).Encode(map[string]any{"issuer": server.URL, "jwks_uri": server.URL + "/keys"})
			return
		}
		if block.Load() {
			close(started)
			<-r.Context().Done()
			close(finished)
			return
		}
		_, _ = w.Write(keys)
	}))
	defer server.Close()
	client := server.Client()
	client.Timeout = 200 * time.Millisecond
	v, err := admission.NewVerifierForTest(context.Background(), admission.WithIssuer(server.URL, testAudience, client))
	if err != nil {
		t.Fatal(err)
	}
	if client.CheckRedirect != nil || client.Timeout != 200*time.Millisecond {
		t.Fatal("constructor mutated the caller's HTTP client")
	}
	claims := issuer.baseClaims()
	claims["iss"] = server.URL
	token := issuer.sign(t, claims)
	block.Store(true)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- v.Verify(ctx, admissionReview(testGroup).Request, token) }()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("verification did not fetch signing keys lazily")
	}
	cancel()
	select {
	case err := <-done:
		if err == nil || err.Error() != "webhook token verification failed" {
			t.Fatalf("cancelled verification = %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("verification did not cancel its wait")
	}
	select {
	case <-finished:
	case <-time.After(5 * time.Second):
		t.Fatal("background key request outlived the HTTP client's timeout")
	}
}
