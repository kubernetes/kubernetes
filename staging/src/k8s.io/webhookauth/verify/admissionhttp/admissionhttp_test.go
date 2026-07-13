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
// codes, whether the downstream handler is reached, body limits, and
// fail-closed decoding — against REAL token verification. Verifiers are built
// with oidc.NewRemoteVerifier over a httptest.NewTLSServer that serves an
// OIDC discovery document and JWKS; tokens are real RS256 JWTs signed by that
// server's key. There is no insecure path: the client reaches the server only
// through its own cert pool. The secure production wiring is exercised verbatim.
package admissionhttp_test

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"errors"
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
	"k8s.io/webhookauth/verify"
	"k8s.io/webhookauth/verify/admissionhttp"
	"k8s.io/webhookauth/verify/oidc"
)

const (
	testKeyID    = "test-signing-key"
	testAudience = "webhook.example.com"
	testGroup    = "apps"
	testSubject  = "system:serviceaccount:kube-system:webhook-auth"
	reviewUID    = "req-uid-1"

	// allowedAPIGroupClaimKey is the fully-namespaced KEP-6060 attestation claim
	// key. The core package's constant is unexported, so the tests hard-code the
	// wire string a spec-compliant issuer must emit; the bare "allowedAPIGroup"
	// form is a known issuer bug and MUST NOT be used.
	allowedAPIGroupClaimKey = "webhook-authentication.k8s.io/allowedAPIGroup"
	wildcardAPIGroup        = "*"
)

// oidcTestServer is a httptest TLS server that serves an OIDC discovery document
// and a JWKS holding a single locally generated RSA signing key. It lets the
// adapter tests exercise the real production path (oidc.NewRemoteVerifier
// → go-oidc discovery + signature/iss/aud/exp checks) with no insecure or
// skip-verify option: callers reach it only through the server's own cert pool.
type oidcTestServer struct {
	server *httptest.Server
	issuer string
	signer jose.Signer
}

// startOIDCServer stands up the TLS discovery + JWKS endpoint for a fresh RSA
// key. It returns an error rather than taking a *testing.T so both the tests
// (via newOIDCTestServer) and the runnable examples can share one implementation.
func startOIDCServer() (*oidcTestServer, error) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, err
	}
	signer, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.RS256, Key: jose.JSONWebKey{Key: priv, KeyID: testKeyID}},
		(&jose.SignerOptions{}).WithType("JWT"),
	)
	if err != nil {
		return nil, err
	}
	jwks := jose.JSONWebKeySet{Keys: []jose.JSONWebKey{{
		Key:       priv.Public(),
		KeyID:     testKeyID,
		Algorithm: string(jose.RS256),
		Use:       "sig",
	}}}

	ts := &oidcTestServer{signer: signer}
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer":                                ts.issuer,
			"jwks_uri":                              ts.issuer + "/keys",
			"id_token_signing_alg_values_supported": []string{string(jose.RS256)},
		})
	})
	mux.HandleFunc("/keys", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(jwks)
	})
	ts.server = httptest.NewTLSServer(mux)
	ts.issuer = ts.server.URL
	return ts, nil
}

// newOIDCTestServer is the *testing.T wrapper around startOIDCServer, wiring
// cleanup so the server is closed at the end of the test.
func newOIDCTestServer(t *testing.T) *oidcTestServer {
	t.Helper()
	ts, err := startOIDCServer()
	if err != nil {
		t.Fatalf("starting OIDC test server: %v", err)
	}
	t.Cleanup(ts.server.Close)
	return ts
}

func (ts *oidcTestServer) close()               { ts.server.Close() }
func (ts *oidcTestServer) client() *http.Client { return ts.server.Client() }

// verifier builds a fixed-audience Verifier via full OIDC discovery over TLS.
func (ts *oidcTestServer) verifier(t *testing.T) *verify.Verifier {
	t.Helper()
	v, err := oidc.NewRemoteVerifier(context.Background(), ts.issuer, testAudience, oidc.WithHTTPClient(ts.client()))
	if err != nil {
		t.Fatalf("NewRemoteVerifier: %v", err)
	}
	return v
}

// signClaims mints a compact RS256 JWS over claims using the server's advertised
// key. It returns an error so the runnable examples can reuse it.
func (ts *oidcTestServer) signClaims(claims map[string]any) (string, error) {
	payload, err := json.Marshal(claims)
	if err != nil {
		return "", err
	}
	jws, err := ts.signer.Sign(payload)
	if err != nil {
		return "", err
	}
	return jws.CompactSerialize()
}

// sign is the *testing.T wrapper around signClaims.
func (ts *oidcTestServer) sign(t *testing.T, claims map[string]any) string {
	t.Helper()
	tok, err := ts.signClaims(claims)
	if err != nil {
		t.Fatalf("signing claims: %v", err)
	}
	return tok
}

// baseClaims returns a valid validating-webhook-bound token claim set for
// testGroup with a comfortably future expiry and this server's issuer. Tests
// mutate the returned map to build sad-path variants.
func (ts *oidcTestServer) baseClaims() map[string]any {
	now := time.Now()
	return map[string]any{
		"iss": ts.issuer,
		"sub": testSubject,
		"aud": []string{testAudience},
		"exp": now.Add(5 * time.Minute).Unix(),
		"nbf": now.Add(-1 * time.Minute).Unix(),
		"iat": now.Unix(),
		"kubernetes.io": map[string]any{
			"validatingWebhookConfiguration": map[string]any{"name": "vwc", "uid": "vwc-uid"},
			"attestationClaims": map[string]any{
				allowedAPIGroupClaimKey: []string{testGroup},
			},
		},
	}
}

// coreGroupClaims returns an otherwise-valid token claim set authorized for the
// core API group (""). Before the fail-closed behavior, an undecodable
// AdmissionReview body defaulted the review group to "", which such a token
// would match.
func (ts *oidcTestServer) coreGroupClaims() map[string]any {
	c := ts.baseClaims()
	k8s := c["kubernetes.io"].(map[string]any)
	k8s["attestationClaims"] = map[string]any{
		allowedAPIGroupClaimKey: []string{""},
	}
	return c
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
// verifier built over a real TLS OIDC discovery/JWKS endpoint, the enforce-only
// adapter, and a spy downstream ReviewHandler. Each case asserts the HTTP
// status, whether the downstream was reached, and that no response leaks claim
// identifiers. There is no permissive mode: every failure is a uniform 401 and
// the downstream is never reached. Signatures are real RS256 JWTs verified by
// go-oidc; the deny reason is intentionally NOT observable at the HTTP boundary
// (anti-enumeration), so it is asserted via the log-reason path elsewhere.
func TestWithTokenVerification_EndToEnd(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)

	tests := []struct {
		name string
		// token builds the presented token; nil omits the header.
		token func() string
		// reviewGroup is the API group of the AdmissionReview resource.
		reviewGroup string
		// wantStatus is the expected HTTP status code.
		wantStatus int
		// wantNextReached asserts whether the downstream handler ran.
		wantNextReached bool
	}{
		{
			name:            "valid token, group matches -> next reached, 200 allow",
			token:           func() string { return ts.sign(t, ts.baseClaims()) },
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
		},
		{
			name: "expired token -> 401, next not reached",
			token: func() string {
				c := ts.baseClaims()
				c["exp"] = time.Now().Add(-1 * time.Minute).Unix()
				return ts.sign(t, c)
			},
			reviewGroup:     testGroup,
			wantStatus:      http.StatusUnauthorized,
			wantNextReached: false,
		},
		{
			name: "wrong audience -> 401, next not reached",
			token: func() string {
				c := ts.baseClaims()
				c["aud"] = []string{"someone.else.example.com"}
				return ts.sign(t, c)
			},
			reviewGroup:     testGroup,
			wantStatus:      http.StatusUnauthorized,
			wantNextReached: false,
		},
		{
			name: "allowedAPIGroup mismatch vs review group -> 401, next not reached",
			token: func() string {
				// Token authorizes testGroup, but the review is for another group.
				return ts.sign(t, ts.baseClaims())
			},
			reviewGroup:     "batch",
			wantStatus:      http.StatusUnauthorized,
			wantNextReached: false,
		},
		{
			name: "wildcard allowedAPIGroup -> allowed for any review group",
			token: func() string {
				c := ts.baseClaims()
				k8s := c["kubernetes.io"].(map[string]any)
				k8s["attestationClaims"] = map[string]any{
					allowedAPIGroupClaimKey: []string{wildcardAPIGroup},
				}
				return ts.sign(t, c)
			},
			reviewGroup:     "any.group.example.com",
			wantStatus:      http.StatusOK,
			wantNextReached: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			spy := newSpyHandler()
			adapter := admissionhttp.WithTokenVerification(v, spy.serve)

			var token string
			if tc.token != nil {
				token = tc.token()
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
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	adapter := admissionhttp.WithTokenVerification(v, spy.serve)

	body := admissionReviewBody(t, testGroup)
	token := ts.sign(t, ts.baseClaims())
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
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	srv := httptest.NewServer(admissionhttp.WithTokenVerification(v, spy.serve))
	defer srv.Close()

	body := admissionReviewBody(t, testGroup)
	token := ts.sign(t, ts.baseClaims())

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

// TestWithTokenVerification_UndecodableBodyFailsClosed proves the adapter fails
// closed: an undecodable AdmissionReview body does not default the review group
// to the core group (""). The request is denied with a generic 401 and the
// downstream handler is never reached, even when the presented token is
// otherwise valid and authorized for the core group.
func TestWithTokenVerification_UndecodableBodyFailsClosed(t *testing.T) {
	ts := newOIDCTestServer(t)
	token := ts.sign(t, ts.coreGroupClaims())
	undecodable := []byte("{ this is not a valid AdmissionReview")

	v := ts.verifier(t)
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
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	// A tiny limit guarantees the AdmissionReview body exceeds it.
	adapter := admissionhttp.WithTokenVerification(v, spy.serve, admissionhttp.WithMaxBodyBytes(16))

	body := admissionReviewBody(t, testGroup)
	if int64(len(body)) <= 16 {
		t.Fatalf("test body must exceed the limit, got %d bytes", len(body))
	}
	token := ts.sign(t, ts.baseClaims())

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
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)

	t.Run("valid token and matching group -> nil", func(t *testing.T) {
		token := ts.sign(t, ts.baseClaims())
		if err := admissionhttp.VerifyAdmissionReview(context.Background(), v, admissionReview(testGroup), token); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("group not authorized -> generic failure", func(t *testing.T) {
		token := ts.sign(t, ts.baseClaims())
		err := admissionhttp.VerifyAdmissionReview(context.Background(), v, admissionReview("batch"), token)
		if err == nil || !errors.Is(err, verify.ErrVerificationFailed) {
			t.Fatalf("want generic verification failure, got %v", err)
		}
	})

	t.Run("nil request -> fails closed", func(t *testing.T) {
		token := ts.sign(t, ts.baseClaims())
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
