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

// TestWithTokenVerification_NilRequestBody proves fail-closed on a request with
// no body at all (r.Body == nil): the adapter denies with a generic 401 and
// never reaches the downstream handler, rather than treating an absent body as
// an empty/core-group review.
func TestWithTokenVerification_NilRequestBody(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	adapter := admissionhttp.WithTokenVerification(v, spy.serve)

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

// TestWithTokenVerification_ValidJSONNoRequest proves fail-closed on a body that
// is valid JSON and decodes into an AdmissionReview but carries no Request. The
// adapter must not proceed with a nil Request (which would make the review's
// resource group ambiguous); it denies with a generic 401 and skips downstream,
// even with an otherwise-valid token authorized for the core group.
func TestWithTokenVerification_ValidJSONNoRequest(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	adapter := admissionhttp.WithTokenVerification(v, spy.serve)

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
// (not just a tiny custom limit) rejects an oversized body: a body larger than
// the adapter's built-in maximum is denied with a generic 401 and never decoded
// or forwarded, guarding against unbounded reads on the default configuration.
func TestWithTokenVerification_HugeBodyDefaultLimit(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	adapter := admissionhttp.WithTokenVerification(v, spy.serve) // default limit

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
	adapter := admissionhttp.WithTokenVerification(v, spy.serve)

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

// erroringBody is a request body whose Read returns a non-EOF error, simulating
// a client connection that drops mid-transfer. It lets the test drive the
// body-read-failure branch of the adapter's single-pass decode.
type erroringBody struct{}

func (erroringBody) Read([]byte) (int, error) { return 0, errors.New("connection reset") }
func (erroringBody) Close() error             { return nil }

// TestWithTokenVerification_BodyReadErrorFailsClosed proves the adapter fails
// closed when the request body cannot be read to completion (a mid-transfer I/O
// error): it denies with a generic 401 and never reaches the downstream handler,
// rather than proceeding with a partially read or empty review.
func TestWithTokenVerification_BodyReadErrorFailsClosed(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	adapter := admissionhttp.WithTokenVerification(v, spy.serve)

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

// closeOnceBody wraps a body reader to prove single consumption: it counts Close
// calls and refuses any Read after Close. If the adapter buffered-and-reset the
// body to decode it twice, it would either read after close (error) or close
// more than once — both of which this body records.
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
// decode-once contract at the body level: the adapter must read the body a
// single pass and Close it exactly once, and it must NOT read the body again
// after closing it (which a buffer-and-reset double decode would require). A
// counting/poisoned body records both, and the downstream still receives the
// correctly decoded review.
func TestWithTokenVerification_DecodeOnce_BodyConsumedExactlyOnce(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)

	// A downstream that records the decoded review but deliberately does NOT
	// touch r.Body, so the poisoned body reflects ONLY the adapter's own reads:
	// any read-after-close would then indicate a second decode pass in the
	// adapter, not routine downstream draining.
	var gotReview *admissionv1.AdmissionReview
	reached := false
	next := func(w http.ResponseWriter, _ *http.Request, review *admissionv1.AdmissionReview) {
		reached = true
		gotReview = review
		w.WriteHeader(http.StatusOK)
	}
	adapter := admissionhttp.WithTokenVerification(v, next)

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
	// The downstream received the decoded review, not a re-read of the body.
	if gotReview == nil || gotReview.Request == nil || gotReview.Request.Resource.Group != testGroup {
		t.Errorf("downstream did not receive the single decoded review: %+v", gotReview)
	}
}

// TestVerifyAdmissionReview_ZeroDecode documents and asserts the structural
// guarantee of the already-decoded entry point: VerifyAdmissionReview operates
// on the caller's decoded *AdmissionReview and performs NO body read or JSON
// decode of its own (there is no io.Reader in its signature). The controller-
// runtime path therefore decodes exactly once — in the framework, never here.
// The test drives a valid review and a nil-Request review through the entry
// point using the same in-memory object, with no HTTP body involved.
func TestVerifyAdmissionReview_ZeroDecode(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	token := ts.sign(t, ts.baseClaims())

	review := &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
		Request: &admissionv1.AdmissionRequest{
			Resource: metav1.GroupVersionResource{Group: testGroup, Version: "v1", Resource: "deployments"},
		},
	}
	// The same pointer is used before and after; a decode would have to replace
	// or mutate it. We assert the call succeeds and the caller's object is
	// unchanged (Request identity preserved).
	before := review.Request
	if err := admissionhttp.VerifyAdmissionReview(context.Background(), v, review, token); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if review.Request != before {
		t.Error("VerifyAdmissionReview replaced the caller's Request — it must not decode")
	}

	// A nil-Request review fails closed with the generic error, still without any
	// decode.
	err := admissionhttp.VerifyAdmissionReview(context.Background(), v, &admissionv1.AdmissionReview{}, token)
	if err == nil || !errors.Is(err, verify.ErrVerificationFailed) {
		t.Fatalf("nil-Request review: want generic ErrVerificationFailed, got %v", err)
	}
}
