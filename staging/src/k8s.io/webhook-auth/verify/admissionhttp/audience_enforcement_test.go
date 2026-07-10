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

package admissionhttp_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/webhook-auth/verify"
	"k8s.io/webhook-auth/verify/admissionhttp"
	"k8s.io/webhook-auth/verify/oidc"
)

// TestAudienceFromServiceURL_ParseError covers the parse-failure branch: a
// service URL that url.Parse rejects surfaces an error rather than a bogus
// audience, so a misconfigured webhook URL fails fast at construction.
func TestAudienceFromServiceURL_ParseError(t *testing.T) {
	// "%zz" is an invalid percent-escape, which url.Parse rejects.
	if _, err := admissionhttp.AudienceFromServiceURL("https://host/%zz"); err == nil {
		t.Error("expected a parse error for an invalid URL escape")
	}
}

// TestWithTokenVerificationDerivedAudience_MismatchEnforced proves the derived
// audience is actually ENFORCED, not merely plumbed: with the expected audience
// derived from the request (https://<host>/validate), a token carrying a
// DIFFERENT aud is rejected with a generic 401 and the downstream handler is
// never reached. This distinguishes "the derived value is used by the verifier"
// from "the verifier was built but ignores it".
func TestWithTokenVerificationDerivedAudience_MismatchEnforced(t *testing.T) {
	ts := newOIDCTestServer(t)

	newVerifier := func(auds []string) (*verify.Verifier, error) {
		return oidc.NewRemoteVerifier(context.Background(), ts.issuer, auds[0], oidc.WithHTTPClient(ts.client()))
	}

	spy := newSpyHandler()
	h := admissionhttp.WithTokenVerificationDerivedAudience(newVerifier, spy.serve)
	server := httptest.NewServer(h)
	defer server.Close()

	// Token asserts an audience that is NOT the request-derived one.
	claims := map[string]any{
		"iss": ts.issuer,
		"aud": []string{"https://not-the-derived-audience.example/validate"},
		"exp": time.Now().Add(time.Hour).Unix(),
		"nbf": time.Now().Add(-time.Minute).Unix(),
		"kubernetes.io": map[string]any{
			"validatingWebhookConfiguration": map[string]any{"name": "vwc", "uid": "vwc-uid"},
			"attestationClaims":              map[string]any{allowedAPIGroupClaimKey: []string{wildcardAPIGroup}},
		},
	}
	token := ts.sign(t, claims)

	body, err := json.Marshal(&admissionv1.AdmissionReview{
		Request: &admissionv1.AdmissionRequest{
			Resource: metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
		},
	})
	if err != nil {
		t.Fatalf("marshal review: %v", err)
	}

	req, err := http.NewRequest(http.MethodPost, server.URL+"/validate", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := server.Client().Do(req)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d (a wrong-audience token must be rejected)", resp.StatusCode, http.StatusUnauthorized)
	}
	if spy.wasReached() {
		t.Error("downstream must not be reached for a wrong-audience token")
	}
}

// TestWithTokenVerificationDerivedAudience_CacheReuse proves the per-audience
// verifier cache: two requests with the SAME derived audience (same Host+path)
// build the verifier ONCE, while a request with a DIFFERENT derived audience
// (different Host) builds a second one. The verifier is constructed on cache
// miss regardless of token validity, so the requests need no valid token — only
// the construction count is asserted.
func TestWithTokenVerificationDerivedAudience_CacheReuse(t *testing.T) {
	ts := newOIDCTestServer(t)

	var mu sync.Mutex
	buildCount := 0
	newVerifier := func(auds []string) (*verify.Verifier, error) {
		mu.Lock()
		buildCount++
		mu.Unlock()
		return oidc.NewRemoteVerifier(context.Background(), ts.issuer, auds[0], oidc.WithHTTPClient(ts.client()))
	}

	h := admissionhttp.WithTokenVerificationDerivedAudience(newVerifier, nil)

	// Drive ServeHTTP directly so r.Host controls the derived audience without a
	// real network hop. The body is a valid AdmissionReview; the token is absent,
	// so each request denies AFTER the verifier is (or isn't) constructed.
	body := admissionReviewBody(t, testGroup)
	serve := func(host string) {
		r := httptest.NewRequest(http.MethodPost, "http://"+host+"/validate", bytes.NewReader(body))
		r.Host = host
		h.ServeHTTP(httptest.NewRecorder(), r)
	}

	serve("host-a.example")
	serve("host-a.example") // same derived audience -> cache hit
	serve("host-b.example") // different derived audience -> second build

	mu.Lock()
	got := buildCount
	mu.Unlock()
	if got != 2 {
		t.Errorf("verifier constructed %d times, want 2 (one per distinct derived audience)", got)
	}
}

// TestWithTokenVerificationDerivedAudience_HappyPathEnforcedIdentity confirms
// the derived-audience path is not just a pass/deny gate: when the token's
// audience matches the request-derived audience, verification succeeds and the
// bound identity is available to a downstream handler. It complements the
// mismatch test by proving the correctly-audienced token is accepted end to end.
func TestWithTokenVerificationDerivedAudience_HappyPathEnforcedIdentity(t *testing.T) {
	ts := newOIDCTestServer(t)

	newVerifier := func(auds []string) (*verify.Verifier, error) {
		return oidc.NewRemoteVerifier(context.Background(), ts.issuer, auds[0], oidc.WithHTTPClient(ts.client()))
	}

	var gotGroup string
	reached := false
	next := func(w http.ResponseWriter, _ *http.Request, review *admissionv1.AdmissionReview) {
		reached = true
		gotGroup = review.Request.Resource.Group
		w.WriteHeader(http.StatusOK)
	}
	h := admissionhttp.WithTokenVerificationDerivedAudience(newVerifier, next)
	server := httptest.NewServer(h)
	defer server.Close()

	host := server.URL[len("http://"):]
	expectedAud := "https://" + host + "/validate"

	claims := map[string]any{
		"iss": ts.issuer,
		"aud": []string{expectedAud},
		"exp": time.Now().Add(time.Hour).Unix(),
		"nbf": time.Now().Add(-time.Minute).Unix(),
		"kubernetes.io": map[string]any{
			"validatingWebhookConfiguration": map[string]any{"name": "vwc", "uid": "vwc-uid"},
			"attestationClaims":              map[string]any{allowedAPIGroupClaimKey: []string{wildcardAPIGroup}},
		},
	}
	token := ts.sign(t, claims)

	body, err := json.Marshal(&admissionv1.AdmissionReview{
		Request: &admissionv1.AdmissionRequest{
			Resource: metav1.GroupVersionResource{Group: "batch", Version: "v1", Resource: "jobs"},
		},
	})
	if err != nil {
		t.Fatalf("marshal review: %v", err)
	}
	req, err := http.NewRequest(http.MethodPost, server.URL+"/validate", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := server.Client().Do(req)
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}
	if !reached {
		t.Error("downstream should be reached for a correctly-audienced token")
	}
	if gotGroup != "batch" {
		t.Errorf("downstream review group = %q, want batch", gotGroup)
	}
}
