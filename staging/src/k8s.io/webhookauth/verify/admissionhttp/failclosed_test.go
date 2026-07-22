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
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/webhookauth/verify"
	"k8s.io/webhookauth/verify/admissionhttp"
)

// TestBearerToken_Table exercises the header parser directly across the scheme
// and token shapes it must accept or reject. The scheme match is
// case-insensitive (RFC 7235); an empty or whitespace-only token is rejected so
// a bare "Bearer" header cannot slip through as a present-but-empty credential.
func TestBearerToken_Table(t *testing.T) {
	tests := []struct {
		name      string
		header    string // "" means no Authorization header at all
		wantToken string
		wantOK    bool
	}{
		{name: "valid bearer", header: "Bearer abc.def.ghi", wantToken: "abc.def.ghi", wantOK: true},
		{name: "lowercase scheme is accepted", header: "bearer abc", wantToken: "abc", wantOK: true},
		{name: "mixed-case scheme is accepted", header: "BeArEr abc", wantToken: "abc", wantOK: true},
		{name: "trailing header whitespace is trimmed", header: "Bearer abc   ", wantToken: "abc", wantOK: true},
		{name: "extra space before the token is rejected (matches apiserver)", header: "Bearer    abc", wantOK: false},
		{name: "no header", header: "", wantOK: false},
		{name: "non-bearer scheme", header: "Basic dXNlcjpwYXNz", wantOK: false},
		{name: "bearer with empty token", header: "Bearer ", wantOK: false},
		{name: "bearer with whitespace-only token", header: "Bearer      ", wantOK: false},
		{name: "shorter than the scheme prefix", header: "Bear", wantOK: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "/validate", nil)
			if tc.header != "" {
				r.Header.Set("Authorization", tc.header)
			}
			token, ok := admissionhttp.BearerToken(r)
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if ok && token != tc.wantToken {
				t.Errorf("token = %q, want %q", token, tc.wantToken)
			}
		})
	}
}

// TestWithTokenVerification_NilRequestBody proves fail-closed on a bodyless
// request (r.Body == nil): a generic 401, downstream not reached, rather than
// treating an absent body as an empty/core-group review.
func TestWithTokenVerification_NilRequestBody(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	adapter := admissionhttp.NewHandlerForTest(v, spy.serve, nil)

	r := httptest.NewRequest(http.MethodPost, "/validate", nil)
	r.Body = nil // an explicitly bodyless request
	r.Header.Set("Authorization", "Bearer "+ts.sign(t, ts.baseClaims()))
	rec := httptest.NewRecorder()

	adapter.ServeHTTP(rec, r)

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
	if spy.wasReached() {
		t.Error("downstream must not be reached when the request has no body")
	}
	assertNoLeak(t, rec.Body.String())
}

// TestWithTokenVerification_ValidJSONNoRequest proves fail-closed on valid JSON
// that decodes to an AdmissionReview with no Request: a nil Request (ambiguous
// resource group) must not proceed — generic 401, downstream skipped, even with
// a valid core-group-authorized token.
func TestWithTokenVerification_ValidJSONNoRequest(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	adapter := admissionhttp.NewHandlerForTest(v, spy.serve, nil)

	// A syntactically valid AdmissionReview envelope with Request omitted.
	body := []byte(`{"apiVersion":"admission.k8s.io/v1","kind":"AdmissionReview"}`)
	token := ts.sign(t, ts.coreGroupClaims())

	rec := httptest.NewRecorder()
	adapter.ServeHTTP(rec, newRequest(t, body, token))

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
	if spy.wasReached() {
		t.Error("downstream must not be reached for a review with no Request")
	}
	assertNoLeak(t, rec.Body.String())
}

// TestWithTokenVerification_HugeBodyDefaultLimit proves the DEFAULT body guard
// (not just a tiny custom limit) rejects an oversized body with a generic 401,
// never decoding or forwarding it — guarding against unbounded reads out of the
// box.
func TestWithTokenVerification_HugeBodyDefaultLimit(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	adapter := admissionhttp.NewHandlerForTest(v, spy.serve, nil) // default limit

	// Comfortably larger than the adapter's default maximum body size.
	huge := bytes.Repeat([]byte("a"), 8<<20) // 8 MiB
	token := ts.sign(t, ts.baseClaims())

	rec := httptest.NewRecorder()
	adapter.ServeHTTP(rec, newRequest(t, huge, token))

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
	if spy.wasReached() {
		t.Error("downstream must not be reached on an over-default-limit body")
	}
	assertNoLeak(t, rec.Body.String())
}

// TestWithTokenVerification_EmptyBearerRejected proves an Authorization header
// present but carrying an empty token is treated as no credential: generic 401,
// downstream not reached, even though the body decodes cleanly.
func TestWithTokenVerification_EmptyBearerRejected(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	adapter := admissionhttp.NewHandlerForTest(v, spy.serve, nil)

	r := newRequest(t, admissionReviewBody(t, testGroup), "")
	r.Header.Set("Authorization", "Bearer ") // present but empty
	rec := httptest.NewRecorder()

	adapter.ServeHTTP(rec, r)

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
	if spy.wasReached() {
		t.Error("downstream must not be reached for an empty bearer token")
	}
}

// TODO(kep-6060): the nil-next "terminal 200" mode was removed — next is now
// required (WithTokenVerification panics on nil). If a standalone auth-gate use
// case is ever needed, reintroduce it explicitly rather than via a nil next.
// (Was TestWithTokenVerification_NilNextTerminal.)

// TestWithTokenVerification_RemoteConfigValidation proves the fail-closed
// construction contract for remote mode (Edie M1/M3/M8). WithRemoteConfig commits
// the handler to remote, so a present-but-incomplete config — missing audience,
// missing issuer, or both — is a HARD construction error: a nil handler and a
// non-nil error, never a silent fallback to in-cluster and never a verifier built
// with an empty (audience-skipping) audience. The positive controls prove a
// complete remote config builds, and that the zero-remote-option in-cluster path
// is NOT accidentally gated by the remote check.
func TestWithTokenVerification_RemoteConfigValidation(t *testing.T) {
	admit := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
	}

	// Partial remote configs fail closed BEFORE any verifier is constructed: the
	// joint issuer+audience check short-circuits, so no network/discovery happens.
	partials := []struct {
		name string
		cfg  admissionhttp.RemoteConfig
	}{
		{name: "issuer set, audience empty", cfg: admissionhttp.RemoteConfig{Issuer: "https://issuer.example.com"}},
		{name: "audience set, issuer empty", cfg: admissionhttp.RemoteConfig{Audience: testAudience}},
		{name: "both empty", cfg: admissionhttp.RemoteConfig{}},
	}
	for _, tc := range partials {
		t.Run(tc.name, func(t *testing.T) {
			h, err := admissionhttp.WithTokenVerification(context.Background(), admit, admissionhttp.WithRemoteConfig(tc.cfg))
			if err == nil {
				t.Fatalf("expected a construction error for a partial remote config, got nil error (handler=%v)", h)
			}
			if h != nil {
				t.Errorf("expected a nil handler on a construction error, got %v", h)
			}
			// Robust to wording: assert only the stable substring, not the exact message.
			if !strings.Contains(err.Error(), "issuer and an audience") {
				t.Errorf("error = %q, want it to mention the joint issuer/audience requirement", err)
			}
		})
	}

	// Positive control: a fully-populated remote config against a throwaway issuer
	// builds a handler with no error (mirrors the Example_rawHTTPWebhook setup).
	t.Run("complete remote config builds", func(t *testing.T) {
		ts := newOIDCTestServer(t)
		h, err := admissionhttp.WithTokenVerification(context.Background(), admit, admissionhttp.WithRemoteConfig(admissionhttp.RemoteConfig{
			Issuer:     ts.issuer,
			Audience:   testAudience,
			HTTPClient: ts.client(),
		}))
		if err != nil {
			t.Fatalf("complete remote config: unexpected error: %v", err)
		}
		if h == nil {
			t.Error("expected a non-nil handler for a complete remote config")
		}
	})

	// Positive control: the zero-remote-option in-cluster path must NOT be gated by
	// the remote validation. With no WithRemoteConfig, WithTokenVerification builds
	// the deferred in-cluster verifier (redirected offline via the test seam)
	// without a validation error.
	t.Run("in-cluster path is not gated by remote validation", func(t *testing.T) {
		ts := newOIDCTestServer(t)
		h, err := admissionhttp.WithTokenVerification(context.Background(), admit,
			admissionhttp.WithInClusterEndpointForTest(ts.issuer, ts.client()))
		if err != nil {
			t.Fatalf("in-cluster path: unexpected error: %v", err)
		}
		if h == nil {
			t.Error("expected a non-nil handler for the in-cluster path")
		}
	})
}

// erroringBody is a request body whose Read returns a non-EOF error, simulating
// a client connection that drops mid-transfer. It lets the test drive the
// body-read-failure branch of the adapter's single-pass decode.
type erroringBody struct{}

func (erroringBody) Read([]byte) (int, error) { return 0, errors.New("connection reset") }
func (erroringBody) Close() error             { return nil }

// TestWithTokenVerification_BodyReadErrorFailsClosed proves fail-closed on a
// body that cannot be read to completion (mid-transfer I/O error): generic 401,
// downstream not reached, never proceeding on a partial or empty review.
func TestWithTokenVerification_BodyReadErrorFailsClosed(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	adapter := admissionhttp.NewHandlerForTest(v, spy.serve, nil)

	r := httptest.NewRequest(http.MethodPost, "/validate", http.NoBody)
	r.Body = erroringBody{}
	r.Header.Set("Authorization", "Bearer "+ts.sign(t, ts.baseClaims()))
	rec := httptest.NewRecorder()

	adapter.ServeHTTP(rec, r)

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
	if spy.wasReached() {
		t.Error("downstream must not be reached when the body read fails")
	}
	assertNoLeak(t, rec.Body.String())
}

// closeOnceBody proves single consumption: it counts Close calls and refuses any
// Read after Close. A buffer-and-reset double decode would either read after
// close or close twice — both recorded here.
type closeOnceBody struct {
	mu       sync.Mutex
	r        *bytes.Reader
	closes   int
	closed   bool
	readErr  error
	readAfor bool // a read occurred after Close
}

func newCloseOnceBody(b []byte) *closeOnceBody {
	return &closeOnceBody{r: bytes.NewReader(b)}
}

func (c *closeOnceBody) Read(p []byte) (int, error) {
	c.mu.Lock()
	if c.closed {
		c.readAfor = true
		c.mu.Unlock()
		return 0, errors.New("read after close")
	}
	c.mu.Unlock()
	return c.r.Read(p)
}

func (c *closeOnceBody) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closes++
	c.closed = true
	return nil
}

func (c *closeOnceBody) stats() (closes int, readAfterClose bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closes, c.readAfor
}

// TestWithTokenVerification_DecodeOnce_BodyConsumedExactlyOnce hardens the
// decode-once contract at the body level: the adapter reads the body in a single
// pass, Closes it exactly once, and never reads after Close (which a
// buffer-and-reset double decode would require). The downstream still receives
// the correctly decoded review.
func TestWithTokenVerification_DecodeOnce_BodyConsumedExactlyOnce(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)

	// A downstream that records the decoded request. It cannot touch r.Body (the
	// seam receives the decoded AdmissionRequest, not the raw request), so the
	// poisoned body reflects ONLY the adapter's own reads: any read-after-close
	// would indicate a second decode pass in the adapter.
	var gotRequest *admissionv1.AdmissionRequest
	reached := false
	next := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		reached = true
		gotRequest = req
		return &admissionv1.AdmissionResponse{Allowed: true}
	}
	adapter := admissionhttp.NewHandlerForTest(v, next, nil)

	body := admissionReviewBody(t, testGroup)
	cb := newCloseOnceBody(body)

	r := httptest.NewRequest(http.MethodPost, "/validate", http.NoBody)
	r.Body = cb
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer "+ts.sign(t, ts.baseClaims()))
	rec := httptest.NewRecorder()

	adapter.ServeHTTP(rec, r)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d (body %q)", rec.Code, http.StatusOK, rec.Body.String())
	}
	if !reached {
		t.Fatal("downstream was not reached on a valid request")
	}
	closes, readAfterClose := cb.stats()
	if closes != 1 {
		t.Errorf("body Close called %d times, want exactly 1", closes)
	}
	if readAfterClose {
		t.Error("adapter read the body after Close — indicates a second decode pass")
	}
	// The downstream received the decoded request, not a re-read of the body.
	if gotRequest == nil || gotRequest.Resource.Group != testGroup {
		t.Errorf("downstream did not receive the single decoded request: %+v", gotRequest)
	}
}

// TestVerifyAdmissionRequest_ZeroDecode asserts the already-decoded entry point's
// structural guarantee: VerifyAdmissionRequest operates on the caller's decoded
// *AdmissionRequest and performs NO body read or JSON decode (no io.Reader in its
// signature), so the controller-runtime path decodes exactly once, in the
// framework.
func TestVerifyAdmissionRequest_ZeroDecode(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	token := ts.sign(t, ts.baseClaims())

	req := &admissionv1.AdmissionRequest{
		Resource: metav1.GroupVersionResource{Group: testGroup, Version: "v1", Resource: "deployments"},
	}
	if err := admissionhttp.VerifyAdmissionRequest(context.Background(), v, req, token); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// A nil request fails closed with the generic error, still without any decode.
	err := admissionhttp.VerifyAdmissionRequest(context.Background(), v, nil, token)
	if err == nil || !errors.Is(err, verify.ErrVerificationFailed) {
		t.Fatalf("nil request: want generic ErrVerificationFailed, got %v", err)
	}
}
