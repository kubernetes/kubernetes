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
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	jose "gopkg.in/go-jose/go-jose.v2"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/webhook/authentication/verify"
	"k8s.io/client-go/webhook/authentication/verify/admissionhttp"
	"k8s.io/client-go/webhook/authentication/verify/josekeyset"
)

const (
	testAudience = "webhook.example.com"
	testGroup    = "apps"
	testIssuer   = "https://issuer.example.com"
	testKeyID    = "admissionhttp-test-key"
	reviewUID    = "req-uid-1"
)

// signingHarness mints RS256-signed KEP-6060 tokens and exposes the matching
// JWKS so a StaticKeySet can verify them. It mirrors the harness used by the
// josekeyset tests but is local so this end-to-end demo stays self-contained.
type signingHarness struct {
	signer jose.Signer
	jwks   jose.JSONWebKeySet
}

func newSigningHarness(t *testing.T) *signingHarness {
	t.Helper()
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	signer, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.RS256, Key: jose.JSONWebKey{Key: priv, KeyID: testKeyID}},
		(&jose.SignerOptions{}).WithType("JWT"),
	)
	if err != nil {
		t.Fatalf("new signer: %v", err)
	}
	jwks := jose.JSONWebKeySet{Keys: []jose.JSONWebKey{
		{Key: priv.Public(), KeyID: testKeyID, Algorithm: string(jose.RS256), Use: "sig"},
	}}
	return &signingHarness{signer: signer, jwks: jwks}
}

// mint signs claims into a compact JWS.
func (h *signingHarness) mint(t *testing.T, claims map[string]interface{}) string {
	t.Helper()
	payload, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshal claims: %v", err)
	}
	jws, err := h.signer.Sign(payload)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	compact, err := jws.CompactSerialize()
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	return compact
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

func newVerifier(t *testing.T, h *signingHarness) *verify.Verifier {
	t.Helper()
	v, err := verify.NewVerifier(josekeyset.NewStaticKeySet(h.jwks), testIssuer, []string{testAudience})
	if err != nil {
		t.Fatalf("NewVerifier: %v", err)
	}
	return v
}

// admissionReviewBody builds a JSON AdmissionReview request body whose resource
// belongs to apiGroup. It is the payload the wrapped handler expects to read.
func admissionReviewBody(t *testing.T, apiGroup string) []byte {
	t.Helper()
	review := admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
		Request: &admissionv1.AdmissionRequest{
			UID:      types.UID(reviewUID),
			Resource: metav1.GroupVersionResource{Group: apiGroup, Version: "v1", Resource: "deployments"},
			Name:     "my-deploy",
		},
	}
	body, err := json.Marshal(review)
	if err != nil {
		t.Fatalf("marshal AdmissionReview: %v", err)
	}
	return body
}

// spyHandler is the "real" admission webhook that the adapter wraps. It records
// whether it was reached and the exact body it observed, then returns a canned
// allow AdmissionReview so a successful round-trip is visible in the response.
type spyHandler struct {
	mu       sync.Mutex
	reached  bool
	gotBody  []byte
	respUID  types.UID
	allowVal bool
}

func newSpyHandler() *spyHandler {
	return &spyHandler{respUID: types.UID(reviewUID), allowVal: true}
}

func (s *spyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	s.mu.Lock()
	s.reached = true
	s.gotBody = body
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

func (s *spyHandler) observedBody() []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.gotBody
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

// TestWithTokenVerification_EndToEnd exercises the full offline flow: a real
// signed token, a StaticKeySet-backed verifier, the adapter, and a spy "real"
// handler standing in for the webhook. Each case asserts the HTTP status, that
// next was (or was not) reached, and that no response leaks claim identifiers.
func TestWithTokenVerification_EndToEnd(t *testing.T) {
	tests := []struct {
		name string
		// permissive selects permissive mode; false means enforce (default).
		permissive bool
		// token builds the presented token from the harness; "" omits the header.
		token func(t *testing.T, h *signingHarness) string
		// reviewGroup is the API group of the AdmissionReview resource.
		reviewGroup string
		// wantStatus is the expected HTTP status code.
		wantStatus int
		// wantNextReached asserts whether the wrapped handler ran.
		wantNextReached bool
		// wantHookErr asserts whether the result hook observed a failure.
		wantHookErr bool
	}{
		{
			name:            "valid token, group matches -> next reached, 200 allow",
			token:           func(t *testing.T, h *signingHarness) string { return h.mint(t, baseClaims()) },
			reviewGroup:     testGroup,
			wantStatus:      http.StatusOK,
			wantNextReached: true,
			wantHookErr:     false,
		},
		{
			name:            "missing Authorization header, enforce -> 401, next not reached",
			token:           nil,
			reviewGroup:     testGroup,
			wantStatus:      http.StatusUnauthorized,
			wantNextReached: false,
			wantHookErr:     true,
		},
		{
			name:            "missing Authorization header, permissive -> next reached",
			permissive:      true,
			token:           nil,
			reviewGroup:     testGroup,
			wantStatus:      http.StatusOK,
			wantNextReached: true,
			wantHookErr:     true,
		},
		{
			name: "expired token, enforce -> 403, next not reached",
			token: func(t *testing.T, h *signingHarness) string {
				c := baseClaims()
				c["exp"] = time.Now().Add(-1 * time.Minute).Unix()
				return h.mint(t, c)
			},
			reviewGroup:     testGroup,
			wantStatus:      http.StatusForbidden,
			wantNextReached: false,
			wantHookErr:     true,
		},
		{
			name:       "expired token, permissive -> next reached, hook observes failure",
			permissive: true,
			token: func(t *testing.T, h *signingHarness) string {
				c := baseClaims()
				c["exp"] = time.Now().Add(-1 * time.Minute).Unix()
				return h.mint(t, c)
			},
			reviewGroup:     testGroup,
			wantStatus:      http.StatusOK,
			wantNextReached: true,
			wantHookErr:     true,
		},
		{
			name: "wrong audience, enforce -> 403, next not reached",
			token: func(t *testing.T, h *signingHarness) string {
				c := baseClaims()
				c["aud"] = []string{"someone.else.example.com"}
				return h.mint(t, c)
			},
			reviewGroup:     testGroup,
			wantStatus:      http.StatusForbidden,
			wantNextReached: false,
			wantHookErr:     true,
		},
		{
			name: "allowedAPIGroup mismatch vs review group, enforce -> 403",
			token: func(t *testing.T, h *signingHarness) string {
				// Token authorizes testGroup, but the review is for another group.
				return h.mint(t, baseClaims())
			},
			reviewGroup:     "batch",
			wantStatus:      http.StatusForbidden,
			wantNextReached: false,
			wantHookErr:     true,
		},
		{
			name: "wildcard allowedAPIGroup -> allowed for any review group",
			token: func(t *testing.T, h *signingHarness) string {
				c := baseClaims()
				k8s := c["kubernetes.io"].(map[string]interface{})
				k8s["attestationClaims"] = map[string][]string{
					verify.AllowedAPIGroupClaimKey: {verify.WildcardAPIGroup},
				}
				return h.mint(t, c)
			},
			reviewGroup:     "any.group.example.com",
			wantStatus:      http.StatusOK,
			wantNextReached: true,
			wantHookErr:     false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h := newSigningHarness(t)
			v := newVerifier(t, h)
			spy := newSpyHandler()

			var (
				hookCalled bool
				hookErr    error
			)
			opts := []admissionhttp.Option{
				admissionhttp.WithResultHook(func(_ *verify.Result, err error) {
					hookCalled = true
					hookErr = err
				}),
			}
			if tc.permissive {
				opts = append(opts, admissionhttp.WithPermissiveMode())
			} else {
				opts = append(opts, admissionhttp.WithEnforceMode())
			}

			adapter := admissionhttp.WithTokenVerification(v, spy, opts...)

			var token string
			if tc.token != nil {
				token = tc.token(t, h)
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
				t.Error("result hook was not called")
			}
			if tc.wantHookErr && hookErr == nil {
				t.Error("expected result hook to observe an error, got nil")
			}
			if !tc.wantHookErr && hookErr != nil {
				t.Errorf("unexpected result hook error: %v", hookErr)
			}

			// No response must leak webhook identifiers regardless of outcome.
			assertNoLeak(t, rec.Body.String())
		})
	}
}

// TestWithTokenVerification_BodyReadableByNext proves the adapter's body reset:
// after the adapter buffers and decodes the AdmissionReview, the wrapped handler
// must still observe the full, byte-identical request body.
func TestWithTokenVerification_BodyReadableByNext(t *testing.T) {
	h := newSigningHarness(t)
	v := newVerifier(t, h)
	spy := newSpyHandler()
	adapter := admissionhttp.WithTokenVerification(v, spy)

	body := admissionReviewBody(t, testGroup)
	token := h.mint(t, baseClaims())
	req := newRequest(t, body, token)
	rec := httptest.NewRecorder()

	adapter.ServeHTTP(rec, req)

	if !spy.wasReached() {
		t.Fatal("wrapped handler was not reached on a valid token")
	}
	if got := spy.observedBody(); !bytes.Equal(got, body) {
		t.Errorf("wrapped handler saw body %q, want %q", string(got), string(body))
	}
}

// TestWithTokenVerification_OverHTTPServer runs the adapter behind a real
// httptest.Server to demonstrate the end-to-end wire path, including the
// Authorization header travelling over an actual HTTP round trip.
func TestWithTokenVerification_OverHTTPServer(t *testing.T) {
	h := newSigningHarness(t)
	v := newVerifier(t, h)
	spy := newSpyHandler()
	srv := httptest.NewServer(admissionhttp.WithTokenVerification(v, spy))
	defer srv.Close()

	body := admissionReviewBody(t, testGroup)
	token := h.mint(t, baseClaims())

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
		t.Error("wrapped handler was not reached over the HTTP server")
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
// core API group (""). Before the fail-closed fix, an undecodable AdmissionReview
// body defaulted the review group to "", which such a token would match.
func coreGroupClaims() map[string]interface{} {
	c := baseClaims()
	k8s := c["kubernetes.io"].(map[string]interface{})
	k8s["attestationClaims"] = map[string][]string{
		verify.AllowedAPIGroupClaimKey: {""},
	}
	return c
}

// TestWithTokenVerification_UndecodableBodyFailsClosed proves the adapter fails
// closed: an undecodable AdmissionReview body no longer defaults the review
// group to the core group (""). In enforce mode the request is denied and the
// wrapped handler is never reached, even when the presented token is otherwise
// valid and authorized for the core group.
func TestWithTokenVerification_UndecodableBodyFailsClosed(t *testing.T) {
	h := newSigningHarness(t)
	token := h.mint(t, coreGroupClaims())
	undecodable := []byte("{ this is not a valid AdmissionReview")

	t.Run("enforce -> denied, next not reached", func(t *testing.T) {
		v := newVerifier(t, h)
		spy := newSpyHandler()
		adapter := admissionhttp.WithTokenVerification(v, spy, admissionhttp.WithEnforceMode())

		rec := httptest.NewRecorder()
		adapter.ServeHTTP(rec, newRequest(t, undecodable, token))

		if rec.Code != http.StatusBadRequest {
			t.Errorf("status = %d, want %d", rec.Code, http.StatusBadRequest)
		}
		if spy.wasReached() {
			t.Error("wrapped handler was reached on an undecodable body")
		}
		assertNoLeak(t, rec.Body.String())
	})

	t.Run("permissive -> next reached", func(t *testing.T) {
		v := newVerifier(t, h)
		spy := newSpyHandler()
		adapter := admissionhttp.WithTokenVerification(v, spy, admissionhttp.WithPermissiveMode())

		rec := httptest.NewRecorder()
		adapter.ServeHTTP(rec, newRequest(t, undecodable, token))

		if !spy.wasReached() {
			t.Error("permissive mode should still reach the wrapped handler")
		}
	})
}

// TestWithTokenVerification_OverLimitBodyRejected proves the adapter rejects a
// body larger than the configured limit with a generic 400 instead of
// forwarding truncated bytes to the wrapped handler.
func TestWithTokenVerification_OverLimitBodyRejected(t *testing.T) {
	h := newSigningHarness(t)
	v := newVerifier(t, h)
	spy := newSpyHandler()
	// A tiny limit guarantees the AdmissionReview body exceeds it.
	adapter := admissionhttp.WithTokenVerification(v, spy, admissionhttp.WithMaxBodyBytes(16))

	body := admissionReviewBody(t, testGroup)
	if int64(len(body)) <= 16 {
		t.Fatalf("test body must exceed the limit, got %d bytes", len(body))
	}
	token := h.mint(t, baseClaims())

	rec := httptest.NewRecorder()
	adapter.ServeHTTP(rec, newRequest(t, body, token))

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
	if spy.wasReached() {
		t.Error("wrapped handler was reached with an over-limit body")
	}
	assertNoLeak(t, rec.Body.String())
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
