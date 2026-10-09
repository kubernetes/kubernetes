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

// These tests use signed tokens and a TLS issuer so HTTP failures also exercise
// token verification.
package admission_test

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

	jose "github.com/go-jose/go-jose/v4"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/component-helpers/auth/admission"
	"k8s.io/component-helpers/auth/admission/internal/oidc"
	"k8s.io/component-helpers/auth/admission/internal/verify"
)

const (
	testKeyID    = "test-signing-key"
	testAudience = "webhook.example.com"
	testGroup    = "apps"
	testSubject  = "system:serviceaccount:kube-system:webhook-auth"
	reviewUID    = "req-uid-1"

	// admissionReviewAPIGroupsClaimKey is the KEP-6060 attestation claim key (the
	// server-side authentication.AttestationAdmissionReviewAPIGroups value). The
	// core package's constant is unexported, so the tests hard-code the wire
	// string it must match.
	admissionReviewAPIGroupsClaimKey = "admissionReviewAPIGroups"
	wildcardAPIGroup                 = "*"
)

// oidcTestServer is a httptest TLS server serving an OIDC discovery document and
// a JWKS with a single locally generated RSA signing key, so the adapter tests
// run the real production path (oidc.NewRemoteVerifier → go-oidc discovery +
// signature/iss/aud/exp checks) with no skip-verify option.
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
	// The apiserver's local JWKS endpoint. The in-cluster verifier
	// (oidc.NewLocalKeySetVerifier) fetches keys here rather than via the
	// discovery jwks_uri, so the in-cluster end-to-end test needs it served
	// alongside /keys. Both serve the same signing key.
	mux.HandleFunc("/openid/v1/jwks", func(w http.ResponseWriter, _ *http.Request) {
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
	v, err := oidc.NewRemoteVerifier(context.Background(), ts.issuer, testAudience, ts.client())
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
			"validatingwebhookconfiguration": map[string]any{"name": "vwc", "uid": "vwc-uid"},
			"attestations": map[string]any{
				admissionReviewAPIGroupsClaimKey: []string{testGroup},
			},
		},
	}
}

// coreGroupClaims provides a valid core-group token to catch malformed reviews
// accidentally accepted as empty-group requests.
func (ts *oidcTestServer) coreGroupClaims() map[string]any {
	c := ts.baseClaims()
	k8s := c["kubernetes.io"].(map[string]any)
	k8s["attestations"] = map[string]any{
		admissionReviewAPIGroupsClaimKey: []string{""},
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

func configMapUpdateReviewBody(t *testing.T, value string) []byte {
	t.Helper()
	object, err := json.Marshal(&corev1.ConfigMap{
		TypeMeta:   metav1.TypeMeta{APIVersion: "v1", Kind: "ConfigMap"},
		ObjectMeta: metav1.ObjectMeta{Name: "large-config"},
		Data:       map[string]string{"payload": value},
	})
	if err != nil {
		t.Fatalf("marshal ConfigMap: %v", err)
	}
	review := admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
		Request: &admissionv1.AdmissionRequest{
			UID:       types.UID(reviewUID),
			Kind:      metav1.GroupVersionKind{Version: "v1", Kind: "ConfigMap"},
			Resource:  metav1.GroupVersionResource{Version: "v1", Resource: "configmaps"},
			Operation: admissionv1.Update,
			Object:    runtime.RawExtension{Raw: object},
			OldObject: runtime.RawExtension{Raw: object},
		},
	}
	body, err := json.Marshal(review)
	if err != nil {
		t.Fatalf("marshal AdmissionReview: %v", err)
	}
	return body
}

// spyHandler is the downstream admission webhook the adapter forwards to. It is
// an admission.AdmissionHandler: it receives the AdmissionRequest the
// adapter already decoded and verified, records what it observed, and returns a
// canned allow response (the adapter wraps and writes it).
type spyHandler struct {
	mu         sync.Mutex
	reached    bool
	gotRequest *admissionv1.AdmissionRequest
	allowVal   bool
}

func newSpyHandler() *spyHandler {
	return &spyHandler{allowVal: true}
}

// serve matches admission.AdmissionHandler.
func (s *spyHandler) serve(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
	s.mu.Lock()
	s.reached = true
	s.gotRequest = req
	s.mu.Unlock()
	return &admissionv1.AdmissionResponse{Allowed: s.allowVal}
}

func (s *spyHandler) wasReached() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.reached
}

func (s *spyHandler) request() *admissionv1.AdmissionRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.gotRequest
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

// TestWithTokenVerification_EndToEnd checks that failed requests cannot reach
// admission logic or disclose claim details in the response.
func TestWithTokenVerification_EndToEnd(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)

	tests := []struct {
		name string
		// token builds the presented token; nil omits the header.
		token func() string
		// reviewGroup is the API group of the AdmissionReview resource.
		reviewGroup string
		// wantStatus is the expected HTTP status code. A missing/invalid bearer
		// token is rejected pre-decode with a bare 401; a decoded-but-unauthorized
		// request is answered with HTTP 200 and an explicit Allowed:false deny.
		wantStatus int
		// wantNextReached asserts whether the downstream handler ran.
		wantNextReached bool
		// wantAllowed is the expected AdmissionResponse.Allowed (checked on a 200).
		wantAllowed bool
	}{
		{
			name:            "valid token, group matches -> next reached, 200 allow",
			token:           func() string { return ts.sign(t, ts.baseClaims()) },
			reviewGroup:     testGroup,
			wantStatus:      http.StatusOK,
			wantNextReached: true,
			wantAllowed:     true,
		},
		{
			name:            "missing Authorization header -> pre-decode 401, next not reached",
			token:           nil,
			reviewGroup:     testGroup,
			wantStatus:      http.StatusUnauthorized,
			wantNextReached: false,
		},
		{
			name: "expired token -> 200 deny, next not reached",
			token: func() string {
				c := ts.baseClaims()
				c["exp"] = time.Now().Add(-1 * time.Minute).Unix()
				return ts.sign(t, c)
			},
			reviewGroup:     testGroup,
			wantStatus:      http.StatusOK,
			wantNextReached: false,
			wantAllowed:     false,
		},
		{
			name: "wrong audience -> 200 deny, next not reached",
			token: func() string {
				c := ts.baseClaims()
				c["aud"] = []string{"someone.else.example.com"}
				return ts.sign(t, c)
			},
			reviewGroup:     testGroup,
			wantStatus:      http.StatusOK,
			wantNextReached: false,
			wantAllowed:     false,
		},
		{
			name: "admissionReviewAPIGroups mismatch vs review group -> 200 deny, next not reached",
			token: func() string {
				// Token authorizes testGroup, but the review is for another group.
				return ts.sign(t, ts.baseClaims())
			},
			reviewGroup:     "batch",
			wantStatus:      http.StatusOK,
			wantNextReached: false,
			wantAllowed:     false,
		},
		{
			name: "wildcard admissionReviewAPIGroups -> allowed for any review group",
			token: func() string {
				c := ts.baseClaims()
				k8s := c["kubernetes.io"].(map[string]any)
				k8s["attestations"] = map[string]any{
					admissionReviewAPIGroupsClaimKey: []string{wildcardAPIGroup},
				}
				return ts.sign(t, c)
			},
			reviewGroup:     "any.group.example.com",
			wantStatus:      http.StatusOK,
			wantNextReached: true,
			wantAllowed:     true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			spy := newSpyHandler()
			adapter := admission.NewHandlerForTest(v, spy.serve)

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
			// On a 200, assert the explicit allow/deny verdict and, for a deny, the
			// 401 Result.Code and echoed request UID.
			if tc.wantStatus == http.StatusOK {
				var review admissionv1.AdmissionReview
				if err := json.Unmarshal(rec.Body.Bytes(), &review); err != nil {
					t.Fatalf("decode response: %v", err)
				}
				if review.Response == nil {
					t.Fatal("response AdmissionReview had no Response")
				}
				if review.Response.Allowed != tc.wantAllowed {
					t.Errorf("Allowed = %v, want %v", review.Response.Allowed, tc.wantAllowed)
				}
				if string(review.Response.UID) != reviewUID {
					t.Errorf("response UID = %q, want %q (adapter must echo request UID)", review.Response.UID, reviewUID)
				}
				if !tc.wantAllowed {
					if review.Response.Result == nil || review.Response.Result.Code != http.StatusUnauthorized {
						t.Errorf("deny Result.Code = %+v, want 401", review.Response.Result)
					}
				}
			}

			// No response must leak webhook identifiers regardless of outcome.
			assertNoLeak(t, rec.Body.String())
		})
	}
}

// TestWithTokenVerification_ForwardsRequestAndEchoesUID proves the seam: on a
// valid token the adapter forwards the DECODED AdmissionRequest to the
// downstream AdmissionHandler, then wraps the handler's AdmissionResponse in an
// AdmissionReview whose response UID echoes the request UID.
func TestWithTokenVerification_ForwardsRequestAndEchoesUID(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	adapter := admission.NewHandlerForTest(v, spy.serve)

	body := admissionReviewBody(t, testGroup)
	token := ts.sign(t, ts.baseClaims())
	req := newRequest(t, body, token)
	rec := httptest.NewRecorder()

	adapter.ServeHTTP(rec, req)

	if !spy.wasReached() {
		t.Fatal("downstream handler was not reached on a valid token")
	}
	got := spy.request()
	if got == nil {
		t.Fatal("downstream did not receive a decoded AdmissionRequest")
	}
	if got.Resource.Group != testGroup {
		t.Errorf("decoded request group = %q, want %q", got.Resource.Group, testGroup)
	}
	if string(got.UID) != reviewUID {
		t.Errorf("decoded request UID = %q, want %q", got.UID, reviewUID)
	}

	var review admissionv1.AdmissionReview
	if err := json.Unmarshal(rec.Body.Bytes(), &review); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if review.Response == nil || !review.Response.Allowed {
		t.Errorf("expected allow response, got %+v", review.Response)
	}
	if string(review.Response.UID) != reviewUID {
		t.Errorf("response UID = %q, want %q (adapter must echo request UID)", review.Response.UID, reviewUID)
	}
}

func TestWithTokenVerification_DoesNotMutateSharedDownstreamResponse(t *testing.T) {
	ts := newOIDCTestServer(t)
	shared := &admissionv1.AdmissionResponse{Allowed: false}
	handler := admission.NewHandlerForTest(ts.verifier(t),
		func(context.Context, *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
			return shared
		})
	token := ts.sign(t, ts.baseClaims())

	for _, uid := range []types.UID{"first-request", "second-request"} {
		review := admissionReview(testGroup)
		review.Request.UID = uid
		body, err := json.Marshal(review)
		if err != nil {
			t.Fatalf("marshal admission review: %v", err)
		}
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, newRequest(t, body, token))
		if rec.Code != http.StatusOK {
			t.Fatalf("response status = %d, want 200", rec.Code)
		}
		var got admissionv1.AdmissionReview
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatalf("decode admission response: %v", err)
		}
		if got.Response == nil || got.Response.UID != uid || got.Response.Allowed {
			t.Errorf("response for %q = %+v, want a denial with the matching UID", uid, got.Response)
		}
		if shared.UID != "" {
			t.Fatalf("adapter mutated downstream's reusable response UID to %q", shared.UID)
		}
	}
}

// TestWithTokenVerification_OverHTTPServer runs the adapter behind a real
// httptest.Server to demonstrate the end-to-end wire path, including the
// Authorization header travelling over an actual HTTP round trip.
func TestWithTokenVerification_OverHTTPServer(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	srv := httptest.NewServer(admission.NewHandlerForTest(v, spy.serve))
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
// closed: an undecodable body is not defaulted to the core group (""). It is
// denied with a generic 401 and never reaches the downstream, even when the
// token is valid and authorized for the core group.
func TestWithTokenVerification_UndecodableBodyFailsClosed(t *testing.T) {
	ts := newOIDCTestServer(t)
	token := ts.sign(t, ts.coreGroupClaims())
	undecodable := []byte("{ this is not a valid AdmissionReview")

	v := ts.verifier(t)
	spy := newSpyHandler()
	adapter := admission.NewHandlerForTest(v, spy.serve)

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

func TestWithTokenVerification_LargeUpdateReview(t *testing.T) {
	ts := newOIDCTestServer(t)
	spy := newSpyHandler()
	handler := admission.NewHandlerForTest(ts.verifier(t), spy.serve)
	body := configMapUpdateReviewBody(t, strings.Repeat(`\`, 900<<10))
	if len(body) <= 3<<20 || len(body) >= 8<<20 {
		t.Fatalf("review body = %d bytes, want between 3 and 8 MiB", len(body))
	}
	claims := ts.baseClaims()
	claims["kubernetes.io"].(map[string]any)["attestations"] = map[string]any{
		admissionReviewAPIGroupsClaimKey: []string{"*"},
	}
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, newRequest(t, body, ts.sign(t, claims)))
	if rec.Code != http.StatusOK || !spy.wasReached() {
		t.Fatalf("large valid review: status=%d, downstream reached=%v", rec.Code, spy.wasReached())
	}
	got := spy.request()
	if got == nil || len(got.Object.Raw) == 0 || len(got.OldObject.Raw) == 0 {
		t.Fatalf("downstream did not receive both update objects: %+v", got)
	}
	var response admissionv1.AdmissionReview
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.Response == nil || !response.Response.Allowed || response.Response.UID != types.UID(reviewUID) {
		t.Fatalf("large valid review response: %+v", response.Response)
	}
}

// TestWithTokenVerification_OverLimitBodyRejected proves the over-limit body
// guard: a body larger than the configured limit is rejected with a generic 401,
// never decoded truncated or forwarded downstream.
func TestWithTokenVerification_OverLimitBodyRejected(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	spy := newSpyHandler()
	// A tiny limit guarantees the AdmissionReview body exceeds it.
	adapter := admission.NewHandlerForTest(v, spy.serve, admission.WithMaxBodyBytesForTest(16))

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

// TestVerifierVerify exercises the already-decoded consumer path without
// allowing an unauthorized group or nil request through the admission gate.
func TestVerifierVerify(t *testing.T) {
	ts := newOIDCTestServer(t)
	vv := admission.NewVerifierForTest(ts.verifier(t))

	t.Run("valid token and matching group -> nil", func(t *testing.T) {
		token := ts.sign(t, ts.baseClaims())
		if err := vv.Verify(context.Background(), admissionReview(testGroup).Request, token); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("group not authorized -> generic failure", func(t *testing.T) {
		token := ts.sign(t, ts.baseClaims())
		err := vv.Verify(context.Background(), admissionReview("batch").Request, token)
		if err == nil || err.Error() != "webhook token verification failed" {
			t.Fatalf("want generic verification failure, got %v", err)
		}
	})

	t.Run("nil request -> fails closed", func(t *testing.T) {
		token := ts.sign(t, ts.baseClaims())
		err := vv.Verify(context.Background(), nil, token)
		if err == nil || err.Error() != "webhook token verification failed" {
			t.Fatalf("want generic verification failure for a nil request, got %v", err)
		}
	})
}

// TestWithTokenVerification_InnerObjectGroupChecked rejects objects whose API
// group differs from the authorized request resource.
func TestWithTokenVerification_InnerObjectGroupChecked(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	token := ts.sign(t, ts.baseClaims())

	reviewWithObject := func(resourceGroup, objectAPIVersion string) []byte {
		obj, err := json.Marshal(map[string]any{"apiVersion": objectAPIVersion, "kind": "Deployment"})
		if err != nil {
			t.Fatalf("marshal object: %v", err)
		}
		review := &admissionv1.AdmissionReview{
			TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
			Request: &admissionv1.AdmissionRequest{
				UID:      types.UID(reviewUID),
				Resource: metav1.GroupVersionResource{Group: resourceGroup, Version: "v1", Resource: "deployments"},
				Object:   runtime.RawExtension{Raw: obj},
			},
		}
		body, err := json.Marshal(review)
		if err != nil {
			t.Fatalf("marshal review: %v", err)
		}
		return body
	}

	t.Run("object group matches resource group -> allowed", func(t *testing.T) {
		spy := newSpyHandler()
		adapter := admission.NewHandlerForTest(v, spy.serve)
		rec := httptest.NewRecorder()
		adapter.ServeHTTP(rec, newRequest(t, reviewWithObject(testGroup, "apps/v1"), token))
		if !spy.wasReached() {
			t.Error("downstream should be reached when the object group matches")
		}
	})

	t.Run("object group differs from resource group -> denied", func(t *testing.T) {
		spy := newSpyHandler()
		adapter := admission.NewHandlerForTest(v, spy.serve)
		rec := httptest.NewRecorder()
		adapter.ServeHTTP(rec, newRequest(t, reviewWithObject(testGroup, "batch/v1"), token))
		if spy.wasReached() {
			t.Error("downstream must not be reached when the object group mismatches")
		}
		var review admissionv1.AdmissionReview
		if err := json.Unmarshal(rec.Body.Bytes(), &review); err != nil {
			t.Fatalf("decode response: %v", err)
		}
		if review.Response == nil || review.Response.Allowed {
			t.Errorf("expected a deny, got %+v", review.Response)
		}
	})
}

// TestWithTokenVerification_InnerObjectGroupChecksBothObjects checks both sides
// of an update so either object can cause a group mismatch to be denied.
func TestWithTokenVerification_InnerObjectGroupChecksBothObjects(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	token := ts.sign(t, ts.baseClaims())

	rawObj := func(apiVersion string) []byte {
		b, err := json.Marshal(map[string]any{"apiVersion": apiVersion, "kind": "Deployment"})
		if err != nil {
			t.Fatalf("marshal object: %v", err)
		}
		return b
	}
	updateReview := func(objectAPIVersion, oldObjectAPIVersion string) []byte {
		review := &admissionv1.AdmissionReview{
			TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
			Request: &admissionv1.AdmissionRequest{
				UID:       types.UID(reviewUID),
				Resource:  metav1.GroupVersionResource{Group: testGroup, Version: "v1", Resource: "deployments"},
				Object:    runtime.RawExtension{Raw: rawObj(objectAPIVersion)},
				OldObject: runtime.RawExtension{Raw: rawObj(oldObjectAPIVersion)},
			},
		}
		body, err := json.Marshal(review)
		if err != nil {
			t.Fatalf("marshal review: %v", err)
		}
		return body
	}

	// Checking only Object would let an unauthorized OldObject through.
	t.Run("oldObject group mismatch is denied even when object matches", func(t *testing.T) {
		spy := newSpyHandler()
		adapter := admission.NewHandlerForTest(v, spy.serve)
		rec := httptest.NewRecorder()
		adapter.ServeHTTP(rec, newRequest(t, updateReview("apps/v1", "batch/v1"), token))
		if spy.wasReached() {
			t.Error("downstream must not be reached when the oldObject group mismatches")
		}
		var review admissionv1.AdmissionReview
		if err := json.Unmarshal(rec.Body.Bytes(), &review); err != nil {
			t.Fatalf("decode response: %v", err)
		}
		if review.Response == nil || review.Response.Allowed {
			t.Errorf("expected a deny, got %+v", review.Response)
		}
	})

	t.Run("object group mismatch is denied even when oldObject matches", func(t *testing.T) {
		spy := newSpyHandler()
		adapter := admission.NewHandlerForTest(v, spy.serve)
		rec := httptest.NewRecorder()
		adapter.ServeHTTP(rec, newRequest(t, updateReview("batch/v1", "apps/v1"), token))
		if spy.wasReached() {
			t.Error("downstream must not be reached when the object group mismatches")
		}
	})

	t.Run("both objects matching is allowed", func(t *testing.T) {
		spy := newSpyHandler()
		adapter := admission.NewHandlerForTest(v, spy.serve)
		rec := httptest.NewRecorder()
		adapter.ServeHTTP(rec, newRequest(t, updateReview("apps/v1", "apps/v1"), token))
		if !spy.wasReached() {
			t.Error("downstream should be reached when both object groups match")
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
