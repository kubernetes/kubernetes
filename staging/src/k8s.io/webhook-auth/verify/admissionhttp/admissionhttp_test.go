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

// The tests below exercise the admissionhttp adapter's BEHAVIOR — HTTP status
// codes, whether the downstream handler is reached, deny reasons, body limits,
// and fail-closed decoding — using a stdlib fake KeySet. Real end-to-end
// SIGNATURE verification is not exercised here; it is covered against the
// go-oidc-backed KeySet in a later step. A token here is simply the JSON claims
// payload the fake KeySet returns verbatim as "already verified".
package admissionhttp_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/webhook-auth/verify"
	"k8s.io/webhook-auth/verify/admissionhttp"
)

const (
	testAudience = "webhook.example.com"
	testGroup    = "apps"
	testIssuer   = "https://issuer.example.com"
	reviewUID    = "req-uid-1"
)

// fakeKeySet is a stand-in verify.KeySet that performs no real crypto: it treats
// the raw token string as the already-verified JSON claims payload. This mirrors
// the fake used by the core verify package tests and keeps the adapter tests
// dependency-minimal — signing is just JSON marshaling and the module needs only
// k8s.io/api. Real signature verification is covered separately (see the
// file-level comment above).
type fakeKeySet struct{}

func (fakeKeySet) VerifySignature(_ context.Context, rawToken string) ([]byte, error) {
	return []byte(rawToken), nil
}

// baseClaims returns a valid validating-webhook-bound token claim set for
// testGroup with a comfortably future expiry. Tests mutate the returned map to
// build sad-path variants.
func baseClaims() map[string]interface{} {
	return map[string]interface{}{
		"iss": "https://issuer.example.com",
		"sub": "system:serviceaccount:kube-system:webhook-auth",
		"aud": []string{testAudience},
		"exp": time.Now().Add(5 * time.Minute).Unix(),
		"kubernetes.io": map[string]interface{}{
			"validatingWebhookConfiguration": map[string]string{"name": "vwc", "uid": "vwc-uid"},
			"attestationClaims": map[string][]string{
				verify.AllowedAPIGroupClaimKey: {testGroup},
			},
		},
	}
}

// mintToken serializes claims into the token string the fake KeySet returns
// verbatim as the verified payload. It replaces real JWS signing: because the
// fake KeySet passes the raw token through untouched, "signing" is just JSON
// marshaling of the claim set.
func mintToken(t *testing.T, claims map[string]interface{}) string {
	t.Helper()
	payload, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshal claims: %v", err)
	}
	return string(payload)
}

func newVerifier(t *testing.T) *verify.Verifier {
	t.Helper()
	v, err := verify.NewVerifier(fakeKeySet{}, testIssuer, []string{testAudience})
	if err != nil {
		t.Fatalf("NewVerifier: %v", err)
	}
	return v
}

// admissionReview builds an AdmissionReview whose resource belongs to apiGroup.
func admissionReview(apiGroup string) *admissionv1.AdmissionReview {
	return &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
		Request: &admissionv1.AdmissionRequest{
			UID:      types.UID(reviewUID),
			Resource: metav1.GroupVersionResource{Group: apiGroup, Version: "v1", Resource: "deployments"},
			Name:     "my-deploy",
		},
	}
}

// admissionReviewBody builds a JSON AdmissionReview request body whose resource
// belongs to apiGroup. It is the payload the adapter decodes.
func admissionReviewBody(t *testing.T, apiGroup string) []byte {
	t.Helper()
	body, err := json.Marshal(admissionReview(apiGroup))
	if err != nil {
		t.Fatalf("marshal AdmissionReview: %v", err)
	}
	return body
}

// spyHandler is the downstream admission webhook the adapter forwards to. It is
// an admissionhttp.ReviewHandler: it receives the AdmissionReview the adapter
// already decoded, records what it observed, and returns a canned allow
// response. It also drains r.Body to prove the adapter consumed it once and did
// not buffer-and-reset it for a second decode.
type spyHandler struct {
	mu        sync.Mutex
	reached   bool
	gotReview *admissionv1.AdmissionReview
	leftover  []byte
	respUID   types.UID
	allowVal  bool
}

func newSpyHandler() *spyHandler {
	return &spyHandler{respUID: types.UID(reviewUID), allowVal: true}
}

// serve matches admissionhttp.ReviewHandler.
func (s *spyHandler) serve(w http.ResponseWriter, r *http.Request, review *admissionv1.AdmissionReview) {
	leftover, _ := io.ReadAll(r.Body)
	s.mu.Lock()
	s.reached = true
	s.gotReview = review
	s.leftover = leftover
	s.mu.Unlock()

	resp := admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
		Response: &admissionv1.AdmissionResponse{UID: s.respUID, Allowed: s.allowVal},
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *spyHandler) wasReached() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.reached
}

func (s *spyHandler) review() *admissionv1.AdmissionReview {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.gotReview
}

func (s *spyHandler) leftoverBody() []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.leftover
}

// newRequest builds a POST carrying the AdmissionReview body and, when token is
// non-empty, an Authorization: Bearer header.
func newRequest(t *testing.T, body []byte, token string) *http.Request {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, "/validate", bytes.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	if token != "" {
		r.Header.Set("Authorization", "Bearer "+token)
	}
	return r
}

// TestWithTokenVerification_EndToEnd exercises the adapter's full behavior: a
// fake-KeySet-backed verifier, the enforce-only adapter, and a spy downstream
// ReviewHandler. Each case asserts the HTTP status, whether the downstream was
// reached, the observed reason string, and that no response leaks claim
// identifiers. There is no permissive mode: every failure is a uniform 401 and
// the downstream is never reached. Signature verification itself is faked here
// (see the file-level comment); these cases cover the contract and adapter
// checks layered on top of it.
func TestWithTokenVerification_EndToEnd(t *testing.T) {
	tests := []struct {
		name string
		// token builds the presented token; nil omits the header.
		token func(t *testing.T) string
		// reviewGroup is the API group of the AdmissionReview resource.
		reviewGroup string
		// wantStatus is the expected HTTP status code.
		wantStatus int
		// wantNextReached asserts whether the downstream handler ran.
		wantNextReached bool
		// wantReason, when non-empty, is a substring the observed deny reason must contain.
		wantReason string
	}{
		{
			name:            "valid token, group matches -> next reached, 200 allow",
			token:           func(t *testing.T) string { return mintToken(t, baseClaims()) },
			reviewGroup:     testGroup,
			wantStatus:      http.StatusOK,
			wantNextReached: true,
		},
		{
			name:            "missing Authorization header -> 401, next not reached",
			token:           nil,
			reviewGroup:     testGroup,
			wantStatus:      http.StatusUnauthorized,
			wantNextReached: false,
			wantReason:      "bearer",
		},
		{
			name: "expired token -> 401, next not reached",
			token: func(t *testing.T) string {
				c := baseClaims()
				c["exp"] = time.Now().Add(-1 * time.Minute).Unix()
				return mintToken(t, c)
			},
			reviewGroup:     testGroup,
			wantStatus:      http.StatusUnauthorized,
			wantNextReached: false,
			wantReason:      "expired",
		},
		{
			name: "wrong audience -> 401, next not reached",
			token: func(t *testing.T) string {
				c := baseClaims()
				c["aud"] = []string{"someone.else.example.com"}
				return mintToken(t, c)
			},
			reviewGroup:     testGroup,
			wantStatus:      http.StatusUnauthorized,
			wantNextReached: false,
			wantReason:      "audience",
		},
		{
			name: "allowedAPIGroup mismatch vs review group -> 401, next not reached",
			token: func(t *testing.T) string {
				// Token authorizes testGroup, but the review is for another group.
				return mintToken(t, baseClaims())
			},
			reviewGroup:     "batch",
			wantStatus:      http.StatusUnauthorized,
			wantNextReached: false,
			wantReason:      "authorize",
		},
		{
			name: "wildcard allowedAPIGroup -> allowed for any review group",
			token: func(t *testing.T) string {
				c := baseClaims()
				k8s := c["kubernetes.io"].(map[string]interface{})
				k8s["attestationClaims"] = map[string][]string{
					verify.AllowedAPIGroupClaimKey: {verify.WildcardAPIGroup},
				}
				return mintToken(t, c)
			},
			reviewGroup:     "any.group.example.com",
			wantStatus:      http.StatusOK,
			wantNextReached: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v := newVerifier(t)
			spy := newSpyHandler()

			var (
				hookCalled bool
				hookRes    *verify.Result
				hookReason string
			)
			adapter := admissionhttp.WithTokenVerification(v, spy.serve,
				admissionhttp.WithObserver(func(res *verify.Result, reason string) {
					hookCalled = true
					hookRes = res
					hookReason = reason
				}),
			)

			var token string
			if tc.token != nil {
				token = tc.token(t)
			}
			body := admissionReviewBody(t, tc.reviewGroup)
			req := newRequest(t, body, token)
			rec := httptest.NewRecorder()

			adapter.ServeHTTP(rec, req)

			if rec.Code != tc.wantStatus {
				t.Errorf("status = %d, want %d (body: %q)", rec.Code, tc.wantStatus, rec.Body.String())
			}
			if got := spy.wasReached(); got != tc.wantNextReached {
				t.Errorf("next reached = %v, want %v", got, tc.wantNextReached)
			}
			if !hookCalled {
				t.Error("observer was not called")
			}
			if tc.wantNextReached {
				// Success: observer sees the identity and no reason.
				if hookRes == nil {
					t.Error("expected observer to see a verified result, got nil")
				}
				if hookReason != "" {
					t.Errorf("expected empty reason on success, got %q", hookReason)
				}
			} else {
				// Denial: observer sees no result and the expected reason.
				if hookRes != nil {
					t.Errorf("expected nil result on denial, got %+v", hookRes)
				}
				if !strings.Contains(hookReason, tc.wantReason) {
					t.Errorf("deny reason = %q, want it to contain %q", hookReason, tc.wantReason)
				}
			}

			// No response must leak webhook identifiers regardless of outcome.
			assertNoLeak(t, rec.Body.String())
		})
	}
}

// TestWithTokenVerification_DecodesOnce proves the decode-once design: the
// adapter decodes the AdmissionReview a single time and hands the decoded object
// to the downstream ReviewHandler, and it does NOT buffer-and-reset r.Body — so
// the downstream both receives the correct decoded review and finds the body
// already fully consumed (nothing left for a second decode).
func TestWithTokenVerification_DecodesOnce(t *testing.T) {
	v := newVerifier(t)
	spy := newSpyHandler()
	adapter := admissionhttp.WithTokenVerification(v, spy.serve)

	body := admissionReviewBody(t, testGroup)
	token := mintToken(t, baseClaims())
	req := newRequest(t, body, token)
	rec := httptest.NewRecorder()

	adapter.ServeHTTP(rec, req)

	if !spy.wasReached() {
		t.Fatal("downstream handler was not reached on a valid token")
	}
	got := spy.review()
	if got == nil || got.Request == nil {
		t.Fatal("downstream did not receive a decoded AdmissionReview")
	}
	if got.Request.Resource.Group != testGroup {
		t.Errorf("decoded review group = %q, want %q", got.Request.Resource.Group, testGroup)
	}
	if string(got.Request.UID) != reviewUID {
		t.Errorf("decoded review UID = %q, want %q", got.Request.UID, reviewUID)
	}
	if left := spy.leftoverBody(); len(left) != 0 {
		t.Errorf("expected r.Body fully consumed by a single decode, got %d leftover bytes: %q", len(left), string(left))
	}
}

// TestWithTokenVerification_OverHTTPServer runs the adapter behind a real
// httptest.Server to demonstrate the end-to-end wire path, including the
// Authorization header travelling over an actual HTTP round trip.
func TestWithTokenVerification_OverHTTPServer(t *testing.T) {
	v := newVerifier(t)
	spy := newSpyHandler()
	srv := httptest.NewServer(admissionhttp.WithTokenVerification(v, spy.serve))
	defer srv.Close()

	body := admissionReviewBody(t, testGroup)
	token := mintToken(t, baseClaims())

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, srv.URL, bytes.NewReader(body))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)

	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}
	if !spy.wasReached() {
		t.Error("downstream handler was not reached over the HTTP server")
	}

	respBody, _ := io.ReadAll(resp.Body)
	var review admissionv1.AdmissionReview
	if err := json.Unmarshal(respBody, &review); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if review.Response == nil || !review.Response.Allowed {
		t.Errorf("expected allow response, got %+v", review.Response)
	}
}

// coreGroupClaims returns an otherwise-valid token claim set authorized for the
// core API group (""). Before the fail-closed behavior, an undecodable
// AdmissionReview body defaulted the review group to "", which such a token
// would match.
func coreGroupClaims() map[string]interface{} {
	c := baseClaims()
	k8s := c["kubernetes.io"].(map[string]interface{})
	k8s["attestationClaims"] = map[string][]string{
		verify.AllowedAPIGroupClaimKey: {""},
	}
	return c
}

// TestWithTokenVerification_UndecodableBodyFailsClosed proves the adapter fails
// closed: an undecodable AdmissionReview body does not default the review group
// to the core group (""). The request is denied with a generic 401 and the
// downstream handler is never reached, even when the presented token is
// otherwise valid and authorized for the core group.
func TestWithTokenVerification_UndecodableBodyFailsClosed(t *testing.T) {
	token := mintToken(t, coreGroupClaims())
	undecodable := []byte("{ this is not a valid AdmissionReview")

	v := newVerifier(t)
	spy := newSpyHandler()
	adapter := admissionhttp.WithTokenVerification(v, spy.serve)

	rec := httptest.NewRecorder()
	adapter.ServeHTTP(rec, newRequest(t, undecodable, token))

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
	if spy.wasReached() {
		t.Error("downstream handler was reached on an undecodable body")
	}
	assertNoLeak(t, rec.Body.String())
}

// TestWithTokenVerification_OverLimitBodyRejected proves the adapter keeps the
// over-limit body guard: a body larger than the configured limit is rejected
// with a generic 401 instead of decoding truncated bytes or reaching the
// downstream handler.
func TestWithTokenVerification_OverLimitBodyRejected(t *testing.T) {
	v := newVerifier(t)
	spy := newSpyHandler()
	// A tiny limit guarantees the AdmissionReview body exceeds it.
	adapter := admissionhttp.WithTokenVerification(v, spy.serve, admissionhttp.WithMaxBodyBytes(16))

	body := admissionReviewBody(t, testGroup)
	if int64(len(body)) <= 16 {
		t.Fatalf("test body must exceed the limit, got %d bytes", len(body))
	}
	token := mintToken(t, baseClaims())

	rec := httptest.NewRecorder()
	adapter.ServeHTTP(rec, newRequest(t, body, token))

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusUnauthorized)
	}
	if spy.wasReached() {
		t.Error("downstream handler was reached with an over-limit body")
	}
	assertNoLeak(t, rec.Body.String())
}

// TestVerifyAdmissionReview exercises the primary decoded-input entry point that
// a caller (for example controller-runtime) uses after decoding the body once.
// It proves a correct token verifies, an unauthorized group produces the single
// generic failure with a useful log reason, and a review with no Request fails
// closed.
func TestVerifyAdmissionReview(t *testing.T) {
	v := newVerifier(t)

	t.Run("valid token and matching group -> nil", func(t *testing.T) {
		token := mintToken(t, baseClaims())
		if err := admissionhttp.VerifyAdmissionReview(context.Background(), v, admissionReview(testGroup), token); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("group not authorized -> generic failure", func(t *testing.T) {
		token := mintToken(t, baseClaims())
		err := admissionhttp.VerifyAdmissionReview(context.Background(), v, admissionReview("batch"), token)
		if err == nil || !errors.Is(err, verify.ErrVerificationFailed) {
			t.Fatalf("want generic verification failure, got %v", err)
		}
		if !strings.Contains(verify.Reason(err), "authorize") {
			t.Errorf("reason = %q, want it to mention authorization", verify.Reason(err))
		}
	})

	t.Run("nil request -> fails closed", func(t *testing.T) {
		token := mintToken(t, baseClaims())
		err := admissionhttp.VerifyAdmissionReview(context.Background(), v, &admissionv1.AdmissionReview{}, token)
		if err == nil || !errors.Is(err, verify.ErrVerificationFailed) {
			t.Fatalf("want generic verification failure for a review with no Request, got %v", err)
		}
	})
}

// assertNoLeak fails if the response body contains any webhook identifier the
// verifier is trusted to keep private (name, uid, subject, or group). This
// guards the anti-enumeration contract at the HTTP boundary.
func assertNoLeak(t *testing.T, body string) {
	t.Helper()
	for _, secret := range []string{"vwc", "vwc-uid", "webhook-auth", testGroup} {
		if strings.Contains(body, secret) {
			t.Errorf("response body leaked %q: %q", secret, body)
		}
	}
}
