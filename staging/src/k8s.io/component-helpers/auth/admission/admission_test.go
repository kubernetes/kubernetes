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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/hmac"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	jose "github.com/go-jose/go-jose/v4"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	"k8s.io/component-helpers/auth/admission"
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

// oidcTestServer serves discovery and signing keys for real RSA verification.
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
	// Local verification uses the fixed JWKS path.
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
func (ts *oidcTestServer) verifier(t *testing.T) *admission.Verifier {
	t.Helper()
	v, err := admission.NewVerifier(context.Background(), admission.WithRemoteIssuer(ts.issuer, testAudience, ts.client()))
	if err != nil {
		t.Fatalf("NewVerifier: %v", err)
	}
	return v
}

func (ts *oidcTestServer) localVerifier(audience string) (*admission.Verifier, error) {
	claims := ts.baseClaims()
	claims["sub"] = "system:serviceaccount:webhooks:receiver"
	claims["aud"] = []string{"https://kubernetes.default.svc"}
	claims["kubernetes.io"] = map[string]any{
		"namespace":      "webhooks",
		"serviceaccount": map[string]any{"name": "receiver", "uid": "receiver-uid"},
	}
	projectedToken, err := ts.signClaims(claims)
	if err != nil {
		return nil, err
	}
	config := &rest.Config{
		Host:        ts.issuer,
		BearerToken: projectedToken,
		TLSClientConfig: rest.TLSClientConfig{
			CAData: pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: ts.server.Certificate().Raw}),
		},
	}
	return admission.NewLocalVerifierForTest(context.Background(), config, audience)
}

func (ts *oidcTestServer) localHandler(next admission.AdmissionHandler, audience string) (*admission.Handler, error) {
	v, err := ts.localVerifier(audience)
	if err != nil {
		return nil, err
	}
	return admission.NewHandlerForTest(v, next), nil
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

func admissionRequest(kind metav1.GroupVersionKind, resource metav1.GroupVersionResource, subresource string, operation admissionv1.Operation) *admissionv1.AdmissionRequest {
	apiVersion := kind.Version
	if kind.Group != "" {
		apiVersion = kind.Group + "/" + kind.Version
	}
	raw, _ := json.Marshal(metav1.TypeMeta{APIVersion: apiVersion, Kind: kind.Kind})
	req := &admissionv1.AdmissionRequest{
		UID:                types.UID(reviewUID),
		Kind:               kind,
		Resource:           resource,
		SubResource:        subresource,
		RequestKind:        &kind,
		RequestResource:    &resource,
		RequestSubResource: subresource,
		Operation:          operation,
		Name:               "my-deploy",
	}
	switch operation {
	case admissionv1.Create, admissionv1.Connect:
		req.Object.Raw = raw
	case admissionv1.Update:
		req.Object.Raw = raw
		req.OldObject.Raw = bytes.Clone(raw)
	case admissionv1.Delete:
		req.OldObject.Raw = raw
	}
	return req
}

func admissionReview(apiGroup string) *admissionv1.AdmissionReview {
	return &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
		Request: admissionRequest(
			metav1.GroupVersionKind{Group: apiGroup, Version: "v1", Kind: "Deployment"},
			metav1.GroupVersionResource{Group: apiGroup, Version: "v1", Resource: "deployments"},
			"", admissionv1.Create,
		),
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
			name: "expired token -> pre-decode 401",
			token: func() string {
				c := ts.baseClaims()
				c["exp"] = time.Now().Add(-1 * time.Minute).Unix()
				return ts.sign(t, c)
			},
			reviewGroup:     testGroup,
			wantStatus:      http.StatusUnauthorized,
			wantNextReached: false,
			wantAllowed:     false,
		},
		{
			name: "wrong audience -> pre-decode 401",
			token: func() string {
				c := ts.baseClaims()
				c["aud"] = []string{"someone.else.example.com"}
				return ts.sign(t, c)
			},
			reviewGroup:     testGroup,
			wantStatus:      http.StatusUnauthorized,
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
	if len(body) <= 3<<20 || len(body) >= 7<<20 {
		t.Fatalf("review body = %d bytes, want between 3 and 7 MiB", len(body))
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
	spy := newSpyHandler()
	adapter := admission.NewHandlerForTest(ts.verifier(t), spy.serve, admission.WithMaxBodyBytesForTest(16))
	body := admissionReviewBody(t, testGroup)
	if len(body) <= 16 {
		t.Fatalf("test body must exceed the limit, got %d bytes", len(body))
	}
	rec := httptest.NewRecorder()
	adapter.ServeHTTP(rec, newRequest(t, body, ts.sign(t, ts.baseClaims())))
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413", rec.Code)
	}
	if spy.wasReached() {
		t.Fatal("downstream handler reached with an over-limit body")
	}
	assertNoLeak(t, rec.Body.String())
}

// TestVerifierVerify exercises the already-decoded consumer path without
// allowing an unauthorized group or nil request through the admission gate.
func TestVerifierVerify(t *testing.T) {
	ts := newOIDCTestServer(t)
	vv := ts.verifier(t)

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

var (
	_ http.Handler                                                                                       = (*admission.Handler)(nil)
	_ func(context.Context, admission.AdmissionHandler, ...admission.Option) (*admission.Handler, error) = admission.WithTokenVerification
	_ func(context.Context, ...admission.Option) (*admission.Verifier, error)                            = admission.NewVerifier
	_ func(*admission.Verifier, context.Context, *admissionv1.AdmissionRequest, string) error            = (*admission.Verifier).Verify
	_ func(*admission.Verifier) error                                                                    = (*admission.Verifier).HealthCheck
	_ func(*admission.Handler) error                                                                     = (*admission.Handler).HealthCheck
	_ func(*http.Request) (string, bool)                                                                 = admission.BearerToken
	_ func(string) admission.Option                                                                      = admission.WithAudience
	_ func(string, string, *http.Client) admission.Option                                                = admission.WithRemoteIssuer
)

type observedBody struct {
	reader    *bytes.Reader
	reads     int
	bytesRead int
	closes    int
}

func (b *observedBody) Read(p []byte) (int, error) {
	b.reads++
	n, err := b.reader.Read(p)
	b.bytesRead += n
	return n, err
}

func (b *observedBody) Close() error {
	b.closes++
	return nil
}

func TestAuthenticationBeforeBodyRead(t *testing.T) {
	ts := newOIDCTestServer(t)
	other := newOIDCTestServer(t)
	v := ts.verifier(t)
	cases := []struct {
		name  string
		token func() string
	}{
		{name: "missing", token: func() string { return "" }},
		{name: "malformed", token: func() string { return "not-a-jwt" }},
		{name: "wrong key", token: func() string { return other.sign(t, ts.baseClaims()) }},
		{name: "wrong issuer", token: func() string {
			claims := ts.baseClaims()
			claims["iss"] = "https://other.example"
			return ts.sign(t, claims)
		}},
		{name: "wrong audience", token: func() string {
			claims := ts.baseClaims()
			claims["aud"] = []string{"https://other.example/validate"}
			return ts.sign(t, claims)
		}},
		{name: "expired", token: func() string {
			claims := ts.baseClaims()
			claims["exp"] = time.Now().Add(-time.Minute).Unix()
			return ts.sign(t, claims)
		}},
		{name: "missing binding", token: func() string {
			claims := ts.baseClaims()
			delete(claims["kubernetes.io"].(map[string]any), "validatingwebhookconfiguration")
			return ts.sign(t, claims)
		}},
		{name: "webhook and pod binding", token: func() string {
			claims := ts.baseClaims()
			claims["kubernetes.io"].(map[string]any)["pod"] = map[string]any{"name": "pod", "uid": "pod-uid"}
			return ts.sign(t, claims)
		}},
		{name: "scalar Kubernetes claims", token: func() string {
			claims := ts.baseClaims()
			claims["kubernetes.io"] = "not an object"
			return ts.sign(t, claims)
		}},
		{name: "mixed wildcard attestation", token: func() string {
			claims := ts.baseClaims()
			claims["kubernetes.io"].(map[string]any)["attestations"] = map[string]any{admissionReviewAPIGroupsClaimKey: []string{"apps", "*"}}
			return ts.sign(t, claims)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			spy := newSpyHandler()
			h := admission.NewHandlerForTest(v, spy.serve, admission.WithMaxBodyBytesForTest(16))
			body := &observedBody{reader: bytes.NewReader(bytes.Repeat([]byte("x"), 8192))}
			r := newRequest(t, nil, tc.token())
			r.Body = body
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, r)
			if rec.Code != http.StatusUnauthorized || rec.Body.String() != "webhook token verification failed\n" {
				t.Fatalf("authentication denial = (%d, %q)", rec.Code, rec.Body.String())
			}
			if body.reads != 0 || spy.wasReached() {
				t.Fatalf("body reads = %d, callback reached = %t", body.reads, spy.wasReached())
			}
		})
	}
}

func TestVerifierRequiresCompactJWT(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	compact := ts.sign(t, ts.baseClaims())
	parts := strings.Split(compact, ".")
	if len(parts) != 3 {
		t.Fatalf("fixture is not a compact JWT: %d parts", len(parts))
	}
	for _, tc := range []struct {
		name          string
		serialization map[string]any
	}{
		{name: "flattened JWS", serialization: map[string]any{"protected": parts[0], "payload": parts[1], "signature": parts[2]}},
		{name: "general JWS", serialization: map[string]any{"payload": parts[1], "signatures": []map[string]any{{"protected": parts[0], "signature": parts[2]}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := json.Marshal(tc.serialization)
			if err != nil {
				t.Fatal(err)
			}
			for _, token := range []string{string(raw), " \n" + string(raw)} {
				if err := v.Verify(context.Background(), admissionReview(testGroup).Request, token); err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("non-compact JWT: Verify = %v, want generic denial", err)
				}
			}
			spy := newSpyHandler()
			h := admission.NewHandlerForTest(v, spy.serve)
			body := &observedBody{reader: bytes.NewReader(admissionReviewBody(t, testGroup))}
			req := newRequest(t, nil, string(raw))
			req.Body = body
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusUnauthorized || rec.Body.String() != "webhook token verification failed\n" || body.reads != 0 || spy.wasReached() {
				t.Fatalf("non-compact HTTP token: status=%d body=%q reads=%d callback=%t", rec.Code, rec.Body.String(), body.reads, spy.wasReached())
			}
		})
	}
	if err := v.Verify(context.Background(), admissionReview(testGroup).Request, compact); err != nil {
		t.Fatalf("compact JWT denied: %v", err)
	}
}

func TestAuthenticatedBodyBoundaries(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	token := ts.sign(t, ts.baseClaims())
	raw := admissionReviewBody(t, testGroup)
	for _, tc := range []struct {
		name   string
		limit  int64
		status int
	}{
		{name: "exact limit", limit: int64(len(raw)), status: http.StatusOK},
		{name: "one byte over", limit: int64(len(raw) - 1), status: http.StatusRequestEntityTooLarge},
	} {
		t.Run(tc.name, func(t *testing.T) {
			spy := newSpyHandler()
			h := admission.NewHandlerForTest(v, spy.serve, admission.WithMaxBodyBytesForTest(tc.limit))
			body := &observedBody{reader: bytes.NewReader(raw)}
			r := newRequest(t, nil, token)
			r.Body = body
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, r)
			if rec.Code != tc.status {
				t.Fatalf("status = %d, want %d; body %q", rec.Code, tc.status, rec.Body.String())
			}
			if body.bytesRead > int(tc.limit+1) || body.closes != 1 {
				t.Fatalf("body bytes read %d, closes %d; limit %d", body.bytesRead, body.closes, tc.limit)
			}
			if spy.wasReached() != (tc.status == http.StatusOK) {
				t.Fatalf("callback reached = %t for status %d", spy.wasReached(), tc.status)
			}
		})
	}
}

func testJWKS(t *testing.T, server *httptest.Server, path string) []byte {
	t.Helper()
	resp, err := server.Client().Get(server.URL + path)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("JWKS fixture = (%d, %v)", resp.StatusCode, err)
	}
	return data
}

func TestLibraryKeyRotationAndAvailability(t *testing.T) {
	first := newOIDCTestServer(t)
	second := newECDSAOIDCIssuer(t)
	var keys atomic.Value
	keys.Store(testJWKS(t, first.server, "/keys"))
	var status atomic.Int32
	status.Store(http.StatusOK)
	var keyRequests atomic.Int32
	var server *httptest.Server
	server = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/.well-known/openid-configuration" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"issuer":                                server.URL,
				"jwks_uri":                              server.URL + "/keys",
				"id_token_signing_alg_values_supported": []string{"RS256", "ES256"},
			})
			return
		}
		if r.URL.Path != "/keys" {
			http.NotFound(w, r)
			return
		}
		keyRequests.Add(1)
		w.WriteHeader(int(status.Load()))
		_, _ = w.Write(keys.Load().([]byte))
	}))
	defer server.Close()
	newVerifier := func() (*admission.Verifier, error) {
		return admission.NewVerifier(context.Background(), admission.WithRemoteIssuer(server.URL, testAudience, server.Client()))
	}
	v, err := newVerifier()
	if err != nil {
		t.Fatal(err)
	}
	if got := keyRequests.Load(); got != 1 {
		t.Fatalf("initial availability requests = %d, want 1", got)
	}
	firstClaims := first.baseClaims()
	firstClaims["iss"] = server.URL
	if err := v.Verify(context.Background(), admissionReview(testGroup).Request, first.sign(t, firstClaims)); err != nil {
		t.Fatalf("first signing key: %v", err)
	}
	if got := keyRequests.Load(); got != 2 {
		t.Fatalf("first verification requests = %d, want 2 (lazy key fetch)", got)
	}
	keys.Store(testJWKS(t, second.server, "/openid/v1/jwks"))
	secondClaims := second.validWebhookClaims(testAudience, "validatingwebhookconfiguration", "vwc", "uid", []string{testGroup})
	secondClaims["iss"] = server.URL
	token := second.sign(t, secondClaims)
	if err := v.Verify(context.Background(), admissionReview(testGroup).Request, token); err != nil {
		t.Fatalf("rotated signing key: %v", err)
	}
	if got := keyRequests.Load(); got != 3 {
		t.Fatalf("rotation requests = %d, want 3", got)
	}
	status.Store(http.StatusServiceUnavailable)
	if err := v.HealthCheck(); err == nil {
		t.Fatal("unavailable endpoint reported available")
	}
	before := keyRequests.Load()
	if err := v.Verify(context.Background(), admissionReview(testGroup).Request, token); err != nil {
		t.Fatalf("availability failure invalidated cached signing keys: %v", err)
	}
	if keyRequests.Load() != before {
		t.Fatal("availability probe changed the library key cache")
	}
	status.Store(http.StatusOK)
	keys.Store([]byte("not JSON"))
	if err := v.HealthCheck(); err != nil {
		t.Fatalf("HTTP200 endpoint is available regardless of JWKS contents: %v", err)
	}
	fresh, err := newVerifier()
	if err != nil {
		t.Fatalf("availability-only construction: %v", err)
	}
	if err := fresh.Verify(context.Background(), admissionReview(testGroup).Request, token); err == nil {
		t.Fatal("real verification accepted malformed signing keys")
	}
	status.Store(http.StatusServiceUnavailable)
	if unavailable, err := newVerifier(); err == nil || unavailable != nil {
		t.Fatalf("construction with unavailable endpoint = (%v, %v)", unavailable, err)
	}
}

type requestTransportFunc func(*http.Request) (*http.Response, error)

func (f requestTransportFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestLocalVerificationUsesProjectedIssuerAndTransport(t *testing.T) {
	ts := newOIDCTestServer(t)
	const projectedIssuer = "https://projected-issuer.example"
	projectedClaims := ts.baseClaims()
	projectedClaims["iss"] = projectedIssuer
	projectedClaims["aud"] = []string{"https://kubernetes.default.svc"}
	delete(projectedClaims, "kubernetes.io")
	projected := ts.sign(t, projectedClaims)
	var requests atomic.Int32
	config := &rest.Config{
		Host:        ts.issuer + "/",
		BearerToken: projected,
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
	if err := v.HealthCheck(); err != nil {
		t.Fatal(err)
	}
	if requests.Load() < 3 {
		t.Fatalf("local availability/verification requests = %d, want at least 3", requests.Load())
	}
	if config.Timeout != 0 {
		t.Fatal("constructor mutated the supplied REST configuration")
	}
	for _, tc := range []struct {
		name  string
		token string
	}{
		{name: "malformed projected token", token: "opaque"},
		{name: "missing projected issuer", token: ts.sign(t, map[string]any{"aud": []string{"https://kubernetes.default.svc"}})},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := rest.CopyConfig(config)
			bad.BearerToken = tc.token
			before := requests.Load()
			if verifier, err := admission.NewLocalVerifierForTest(context.Background(), bad, testAudience); err == nil || verifier != nil {
				t.Fatalf("local bootstrap = (%v, %v)", verifier, err)
			}
			if requests.Load() != before {
				t.Fatal("invalid local bootstrap performed network I/O")
			}
		})
	}
	badCA := rest.CopyConfig(config)
	badCA.CAData = []byte("not a certificate")
	if verifier, err := admission.NewLocalVerifierForTest(context.Background(), badCA, testAudience); err == nil || verifier != nil {
		t.Fatalf("invalid client-go CA configuration = (%v, %v)", verifier, err)
	}
}

func TestLocalVerificationRejectsEveryRedirect(t *testing.T) {
	issuer := newOIDCTestServer(t)
	for _, status := range []int{http.StatusMovedPermanently, http.StatusFound, http.StatusSeeOther, http.StatusTemporaryRedirect, http.StatusPermanentRedirect} {
		for _, crossOrigin := range []bool{false, true} {
			t.Run(fmt.Sprintf("status-%d/cross-origin-%t", status, crossOrigin), func(t *testing.T) {
				var foreignCalls, redirectedCalls atomic.Int32
				foreign := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					foreignCalls.Add(1)
					_, _ = w.Write([]byte(`{"keys":[]}`))
				}))
				defer foreign.Close()
				var origin *httptest.Server
				origin = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path != "/openid/v1/jwks" {
						redirectedCalls.Add(1)
						_, _ = w.Write([]byte(`{"keys":[]}`))
						return
					}
					target := origin.URL + "/redirected"
					if crossOrigin {
						target = foreign.URL + "/keys"
					}
					http.Redirect(w, r, target, status)
				}))
				defer origin.Close()
				ca := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: origin.Certificate().Raw})
				ca = append(ca, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: foreign.Certificate().Raw})...)
				config := &rest.Config{
					Host:            origin.URL,
					BearerToken:     issuer.sign(t, issuer.baseClaims()),
					TLSClientConfig: rest.TLSClientConfig{CAData: ca},
				}
				if verifier, err := admission.NewLocalVerifierForTest(context.Background(), config, testAudience); err == nil || verifier != nil {
					t.Fatalf("redirecting local endpoint = (%v, %v)", verifier, err)
				}
				if foreignCalls.Load() != 0 || redirectedCalls.Load() != 0 {
					t.Fatal("local redirect forwarded a projected credential")
				}
			})
		}
	}
}

func TestRemoteIssuerConfiguration(t *testing.T) {
	for _, issuer := range []string{
		"http://issuer.example", "https://user:password@issuer.example", "https://issuer.example?query=value", "https://issuer.example#fragment",
	} {
		t.Run(issuer, func(t *testing.T) {
			if verifier, err := admission.NewVerifier(context.Background(), admission.WithRemoteIssuer(issuer, testAudience, nil)); err == nil || verifier != nil {
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
	if verifier, err := admission.NewVerifier(context.Background(), admission.WithRemoteIssuer(server.URL, testAudience, server.Client())); err == nil || verifier != nil {
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
	v, err := admission.NewVerifier(context.Background(), admission.WithRemoteIssuer(server.URL, testAudience, client))
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

func TestVerifierErrorsAreGeneric(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	good := ts.sign(t, ts.baseClaims())
	invalidClaims := ts.baseClaims()
	delete(invalidClaims, "kubernetes.io")
	generic := v.Verify(context.Background(), nil, good)
	for _, tc := range []struct {
		name  string
		token string
		req   *admissionv1.AdmissionRequest
	}{
		{name: "signature", token: "invalid", req: admissionReview(testGroup).Request},
		{name: "Kubernetes claims", token: ts.sign(t, invalidClaims), req: admissionReview(testGroup).Request},
		{name: "group", token: good, req: admissionReview("batch").Request},
		{name: "nil request", token: good},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := v.Verify(context.Background(), tc.req, tc.token)
			if err == nil || err != generic || err.Error() != "webhook token verification failed" {
				t.Fatalf("verification error = %v, want the same generic failure", err)
			}
		})
	}
	if err := v.Verify(context.Background(), admissionReview(testGroup).Request, good); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := v.Verify(ctx, admissionReview(testGroup).Request, good); err != generic {
		t.Fatalf("cancelled verification = %v, want generic failure", err)
	}
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
		review := admissionReview(resourceGroup)
		review.Request.Object.Raw = obj
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
		review := admissionReview(testGroup)
		review.Request.Operation = admissionv1.Update
		review.Request.Object.Raw = rawObj(objectAPIVersion)
		review.Request.OldObject.Raw = rawObj(oldObjectAPIVersion)
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

const (
	exampleAudience = "webhook.example.com"
	exampleAPIGroup = "apps" // the API group this webhook's token is authorized for
)

// exampleSignedToken signs a webhook-bound token with the test issuer.
func exampleSignedToken(issuer *oidcTestServer, group string) string {
	claims := map[string]any{
		"iss": issuer.issuer,
		"aud": []string{exampleAudience},
		"exp": time.Now().Add(5 * time.Minute).Unix(),
		"nbf": time.Now().Add(-1 * time.Minute).Unix(),
		"kubernetes.io": map[string]any{
			// A token binds to one webhook configuration.
			"validatingwebhookconfiguration": map[string]any{
				"name": "my-webhook",
				"uid":  "webhook-uid",
			},
			"attestations": map[string]any{
				admissionReviewAPIGroupsClaimKey: []string{group},
			},
		},
	}
	token, err := issuer.signClaims(claims)
	if err != nil {
		panic(err)
	}
	return token
}

// exampleReview builds a minimal AdmissionReview for a resource in group. A real
// webhook receives this from the API server on the wire.
func exampleReview(group string) *admissionv1.AdmissionReview {
	review := admissionReview(group)
	review.Request.UID = ""
	return review
}

// Example_rawHTTPWebhook puts token verification in front of admission logic.
func Example_rawHTTPWebhook() {
	// A test TLS issuer stands in for the configured OIDC issuer.
	issuer, err := startOIDCServer()
	if err != nil {
		panic(err)
	}
	defer issuer.close()

	// Admission logic runs only after token verification succeeds.
	admit := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
	}

	// Bind the expected audience to the issuer, not to the inbound request.
	h, err := admission.WithTokenVerification(context.Background(), admit,
		admission.WithRemoteIssuer(issuer.issuer, exampleAudience, issuer.client()))
	if err != nil {
		panic(err)
	}

	srv := httptest.NewServer(h)
	defer srv.Close()

	// Send a signed request to exercise the mounted handler.
	body, _ := json.Marshal(exampleReview(exampleAPIGroup))
	req, _ := http.NewRequest(http.MethodPost, srv.URL, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+exampleSignedToken(issuer, exampleAPIGroup))

	resp, err := srv.Client().Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	var out admissionv1.AdmissionReview
	_ = json.NewDecoder(resp.Body).Decode(&out)
	if resp.StatusCode == http.StatusOK && out.Response != nil && out.Response.Allowed {
		fmt.Println("admitted")
	}
	// Output: admitted
}

// Example_inClusterWebhook uses a projected issuer and a configured audience.
func Example_inClusterWebhook() {
	admit := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
	}
	// Use the webhook configuration's audience, never a value from the request.
	handler, err := admission.WithTokenVerification(context.Background(), admit,
		admission.WithAudience("https://hook.webhooks.svc:443/validate"))
	if err != nil {
		panic(err)
	}
	mux := http.NewServeMux()
	mux.Handle("/validate", handler)
	_ = mux // Serve using your webhook's existing HTTP server.
}

// Example_controllerRuntimeStyle verifies an already-decoded review. The HTTP
// layer must pass the bearer token separately from the AdmissionReview.
func Example_controllerRuntimeStyle() {
	issuer, err := startOIDCServer()
	if err != nil {
		panic(err)
	}
	defer issuer.close()

	v, err := admission.NewVerifier(context.Background(),
		admission.WithRemoteIssuer(issuer.issuer, exampleAudience, issuer.client()))
	if err != nil {
		panic(err)
	}

	// A framework supplies the decoded review; HTTP middleware supplies the token.
	ctx := context.Background()
	review := exampleReview(exampleAPIGroup)
	token := exampleSignedToken(issuer, exampleAPIGroup)

	// Treat any verification error as a deny; details remain internal.
	if err := v.Verify(ctx, review.Request, token); err != nil {
		fmt.Println("denied") // return a 401 / deny AdmissionResponse here
		return
	}
	fmt.Println("admitted") // proceed to your real admission decision
	// Output: admitted
}

// TestExampleEndToEnd_MinimalWebhook checks that a signed request reaches
// admission logic while an unauthenticated request does not.
func TestExampleEndToEnd_MinimalWebhook(t *testing.T) {
	ts := newOIDCTestServer(t)

	var reached bool
	admit := func(_ context.Context, _ *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		reached = true
		return &admissionv1.AdmissionResponse{Allowed: true}
	}
	h, err := admission.WithTokenVerification(context.Background(), admit,
		admission.WithRemoteIssuer(ts.issuer, exampleAudience, ts.client()))
	if err != nil {
		t.Fatalf("WithTokenVerification: %v", err)
	}
	srv := httptest.NewServer(h)
	defer srv.Close()

	validToken := exampleSignedToken(ts, exampleAPIGroup)

	tests := []struct {
		name            string
		bearer          string // empty means no Authorization header
		wantStatus      int
		wantNextReached bool
	}{
		{
			name:            "valid bearer token -> 200, downstream reached",
			bearer:          validToken,
			wantStatus:      http.StatusOK,
			wantNextReached: true,
		},
		{
			name:            "no bearer token -> 401, downstream NOT reached",
			bearer:          "",
			wantStatus:      http.StatusUnauthorized,
			wantNextReached: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			reached = false

			body, err := json.Marshal(exampleReview(exampleAPIGroup))
			if err != nil {
				t.Fatalf("marshal review: %v", err)
			}
			req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, srv.URL, bytes.NewReader(body))
			if err != nil {
				t.Fatalf("new request: %v", err)
			}
			req.Header.Set("Content-Type", "application/json")
			if tc.bearer != "" {
				req.Header.Set("Authorization", "Bearer "+tc.bearer)
			}

			resp, err := srv.Client().Do(req)
			if err != nil {
				t.Fatalf("do request: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != tc.wantStatus {
				t.Errorf("status = %d, want %d", resp.StatusCode, tc.wantStatus)
			}
			if reached != tc.wantNextReached {
				t.Errorf("downstream reached = %v, want %v", reached, tc.wantNextReached)
			}
		})
	}
}

// TestExampleEndToEnd_InClusterExplicitAudience keeps receiver identity independent of requests.
func TestExampleEndToEnd_InClusterExplicitAudience(t *testing.T) {
	ts := newOIDCTestServer(t)
	const svcName, svcNamespace = "webhook", "default"
	audience := verify.AudienceForService(svcName, svcNamespace, 443, "/validate")
	claims := ts.baseClaims()
	claims["aud"] = []string{audience}
	token := ts.sign(t, claims)

	var reached bool
	admit := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		reached = true
		return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
	}
	h, err := ts.localHandler(admit, audience)
	if err != nil {
		t.Fatalf("construct in-cluster handler: %v", err)
	}
	if err := h.HealthCheck(); err != nil {
		t.Fatalf("explicit audience should be ready at construction: %v", err)
	}
	server := httptest.NewServer(h)
	defer server.Close()

	request := func(bearer string) *http.Response {
		body, err := json.Marshal(exampleReview(exampleAPIGroup))
		if err != nil {
			t.Fatalf("marshal AdmissionReview: %v", err)
		}
		req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/validate", bytes.NewReader(body))
		if err != nil {
			t.Fatalf("build admission request: %v", err)
		}
		req.Host = svcName + "." + svcNamespace + ".svc"
		req.Header.Set("Content-Type", "application/json")
		if bearer != "" {
			req.Header.Set("Authorization", "Bearer "+bearer)
		}
		response, err := server.Client().Do(req)
		if err != nil {
			t.Fatalf("send admission request: %v", err)
		}
		return response
	}

	response := request(token)
	defer response.Body.Close()
	var review admissionv1.AdmissionReview
	if err := json.NewDecoder(response.Body).Decode(&review); err != nil {
		t.Fatalf("decode verified AdmissionReview: %v", err)
	}
	if response.StatusCode != http.StatusOK || review.Response == nil || !review.Response.Allowed || !reached {
		t.Fatalf("verified request: HTTP %d response %+v reached %t", response.StatusCode, review.Response, reached)
	}

	reached = false
	missing := request("")
	defer missing.Body.Close()
	if missing.StatusCode != http.StatusUnauthorized || reached {
		t.Fatalf("missing bearer: HTTP %d reached %t, want 401 without admission", missing.StatusCode, reached)
	}
}

// TestExampleEndToEnd_InClusterSpoofedHostCannotChooseAudience guards the
// receiver-owned expected audience against a first request for another Service.
func TestExampleEndToEnd_InClusterSpoofedHostCannotChooseAudience(t *testing.T) {
	ts := newOIDCTestServer(t)
	trustedAudience := verify.AudienceForService("webhook", "default", 443, "/validate")
	otherAudience := verify.AudienceForService("other", "default", 443, "/validate")
	var reached bool
	h, err := ts.localHandler(func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		reached = true
		return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
	}, trustedAudience)
	if err != nil {
		t.Fatalf("construct in-cluster handler: %v", err)
	}
	server := httptest.NewServer(h)
	defer server.Close()
	verifyRequest := func(audience, host string, wantStatus int) *admissionv1.AdmissionResponse {
		claims := ts.baseClaims()
		claims["aud"] = []string{audience}
		body, err := json.Marshal(exampleReview(exampleAPIGroup))
		if err != nil {
			t.Fatalf("marshal AdmissionReview: %v", err)
		}
		req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/validate", bytes.NewReader(body))
		if err != nil {
			t.Fatalf("build admission request: %v", err)
		}
		req.Host = host
		req.Header.Set("Authorization", "Bearer "+ts.sign(t, claims))
		response, err := server.Client().Do(req)
		if err != nil {
			t.Fatalf("send admission request: %v", err)
		}
		defer response.Body.Close()
		if response.StatusCode != wantStatus {
			t.Fatalf("admission status = %d, want %d", response.StatusCode, wantStatus)
		}
		if wantStatus != http.StatusOK {
			raw, err := io.ReadAll(response.Body)
			if err != nil || string(raw) != "webhook token verification failed\n" {
				t.Fatalf("authentication denial = (%q, %v)", raw, err)
			}
			return nil
		}
		var review admissionv1.AdmissionReview
		if err := json.NewDecoder(response.Body).Decode(&review); err != nil {
			t.Fatalf("decode AdmissionReview: %v", err)
		}
		return review.Response
	}
	if denied := verifyRequest(otherAudience, "other.default.svc", http.StatusUnauthorized); denied != nil || reached {
		t.Fatalf("spoofed first request: response %+v reached %t", denied, reached)
	}
	if err := h.HealthCheck(); err != nil {
		t.Fatalf("endpoint availability after rejected request: %v", err)
	}
	allowed := verifyRequest(trustedAudience, "webhook.default.svc", http.StatusOK)
	if allowed == nil || !allowed.Allowed || !reached {
		t.Fatalf("trusted follow-up: response %+v reached %t, want allow", allowed, reached)
	}
}

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
			token, ok := admission.BearerToken(r)
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
	adapter := admission.NewHandlerForTest(v, spy.serve)

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
	adapter := admission.NewHandlerForTest(v, spy.serve)

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

// TestWithTokenVerification_HugeBodyDefaultLimit rejects a decodable review
// above the default limit before it reaches admission logic.
func TestWithTokenVerification_HugeBodyDefaultLimit(t *testing.T) {
	ts := newOIDCTestServer(t)
	spy := newSpyHandler()
	adapter := admission.NewHandlerForTest(ts.verifier(t), spy.serve)
	body := configMapUpdateReviewBody(t, strings.Repeat("a", 5<<20))
	if len(body) <= 7<<20 {
		t.Fatalf("review body = %d bytes, want more than 7 MiB", len(body))
	}
	rec := httptest.NewRecorder()
	adapter.ServeHTTP(rec, newRequest(t, body, ts.sign(t, ts.baseClaims())))
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413", rec.Code)
	}
	if spy.wasReached() {
		t.Fatal("downstream reached on an over-default-limit body")
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
	adapter := admission.NewHandlerForTest(v, spy.serve)

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

// TestWithTokenVerification_RemoteConfigValidation rejects incomplete remote
// settings rather than falling back to in-cluster verification or skipping the audience.
func TestWithTokenVerification_RemoteConfigValidation(t *testing.T) {
	admit := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		return &admissionv1.AdmissionResponse{UID: req.UID, Allowed: true}
	}

	// Reject partial remote settings before attempting issuer discovery.
	partials := []struct {
		name     string
		issuer   string
		audience string
	}{
		{name: "issuer set, audience empty", issuer: "https://issuer.example.com"},
		{name: "audience set, issuer empty", audience: testAudience},
		{name: "both empty"},
	}
	for _, tc := range partials {
		t.Run(tc.name, func(t *testing.T) {
			h, err := admission.WithTokenVerification(context.Background(), admit, admission.WithRemoteIssuer(tc.issuer, tc.audience, nil))
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
		h, err := admission.WithTokenVerification(context.Background(), admit,
			admission.WithRemoteIssuer(ts.issuer, testAudience, ts.client()))
		if err != nil {
			t.Fatalf("complete remote config: unexpected error: %v", err)
		}
		if h == nil {
			t.Error("expected a non-nil handler for a complete remote config")
		}
	})

	// Local bootstrap must not select its audience from a request.
	t.Run("in-cluster path accepts an explicitly configured audience", func(t *testing.T) {
		ts := newOIDCTestServer(t)
		h, err := ts.localHandler(admit, testAudience)
		if err != nil {
			t.Fatalf("in-cluster path with trusted audience: %v", err)
		}
		if h == nil {
			t.Error("expected a handler with a bound audience")
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
	adapter := admission.NewHandlerForTest(v, spy.serve)

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

// closeOnceBody detects a second decode through repeated closes or reads after Close.
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

// TestWithTokenVerification_DecodeOnce_BodyConsumedExactlyOnce checks that the
// adapter closes the body once and does not decode it again afterward.
func TestWithTokenVerification_DecodeOnce_BodyConsumedExactlyOnce(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)

	// The downstream receives only the decoded request, so body reads here
	// can only come from the adapter.
	var gotRequest *admissionv1.AdmissionRequest
	reached := false
	next := func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		reached = true
		gotRequest = req
		return &admissionv1.AdmissionResponse{Allowed: true}
	}
	adapter := admission.NewHandlerForTest(v, next)

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

// TestVerifierVerify_ZeroDecode asserts the decoded entry point cannot read
// an HTTP body because Verifier.Verify takes only an AdmissionRequest and token.
func TestVerifierVerify_ZeroDecode(t *testing.T) {
	ts := newOIDCTestServer(t)
	vv := ts.verifier(t)
	token := ts.sign(t, ts.baseClaims())

	req := admissionReview(testGroup).Request
	if err := vv.Verify(context.Background(), req, token); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// A nil request fails closed with the generic error, still without any decode.
	err := vv.Verify(context.Background(), nil, token)
	if err == nil || err.Error() != "webhook token verification failed" {
		t.Fatalf("nil request: want generic verification failure, got %v", err)
	}
}

// TestNewVerifier_InClusterExplicitAudience verifies that an explicitly bound
// audience makes the decoded in-cluster path ready before its first request.
func TestNewVerifier_InClusterExplicitAudience(t *testing.T) {
	ts := newOIDCTestServer(t)

	v, err := ts.localVerifier(testAudience)
	if err != nil {
		t.Fatalf("NewVerifier: %v", err)
	}
	// Ready at construction: the audience was bound eagerly, so no first request
	// is needed to leave not-ready.
	if err := v.HealthCheck(); err != nil {
		t.Fatalf("expected a ready verifier, got not-ready: %v", err)
	}

	token := ts.sign(t, ts.baseClaims())
	if err := v.Verify(context.Background(), admissionReview(testGroup).Request, token); err != nil {
		t.Errorf("unexpected verification error: %v", err)
	}

	// A token carrying the wrong audience is denied fail-closed (the bound audience
	// is enforced natively as the go-oidc ClientID).
	wrong := ts.baseClaims()
	wrong["aud"] = []string{"someone.else.example.com"}
	err = v.Verify(context.Background(), admissionReview(testGroup).Request, ts.sign(t, wrong))
	if err == nil || err.Error() != "webhook token verification failed" {
		t.Errorf("wrong audience: want generic verification failure, got %v", err)
	}
}

// TestWithAudience_ConflictsWithRemoteIssuer proves WithAudience and
// WithRemoteIssuer cannot be combined: the audience would be set two ways, so the
// builder fails closed at construction rather than applying a silent precedence.
func TestWithAudience_ConflictsWithRemoteIssuer(t *testing.T) {
	ts := newOIDCTestServer(t)

	v, err := admission.NewVerifier(context.Background(),
		admission.WithRemoteIssuer(ts.issuer, testAudience, ts.client()),
		admission.WithAudience(testAudience))
	if err == nil {
		t.Fatalf("expected a construction error combining WithAudience and WithRemoteIssuer, got verifier %v", v)
	}
	if v != nil {
		t.Errorf("expected a nil verifier on a construction error, got %v", v)
	}
	if !strings.Contains(err.Error(), "WithAudience cannot be combined with WithRemoteIssuer") {
		t.Errorf("error = %q, want it to mention the WithAudience/WithRemoteIssuer conflict", err)
	}
}

// TestWithAudience_EmptyIsError proves an explicit but empty audience is a hard
// construction error, never a verifier built with an audience-skipping audience.
func TestWithAudience_EmptyIsError(t *testing.T) {
	v, err := admission.NewVerifier(context.Background(), admission.WithAudience(""))
	if err == nil || v != nil {
		t.Fatalf("NewVerifier with empty audience = (%v, %v), want (nil, error)", v, err)
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
		{name: "empty audience", opts: []admission.Option{admission.WithAudience("")}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, err := admission.WithTokenVerification(context.Background(), admit, tc.opts...)
			if h != nil || err == nil || !strings.Contains(err.Error(), "WithAudience") {
				t.Fatalf("WithTokenVerification = (%v, %v), want audience error", h, err)
			}
			v, err := admission.NewVerifier(context.Background(), tc.opts...)
			if v != nil || err == nil || !strings.Contains(err.Error(), "WithAudience") {
				t.Fatalf("NewVerifier = (%v, %v), want audience error", v, err)
			}
		})
	}
}

// ecdsaOIDCIssuer mimics the kube-apiserver's default ECDSA P-256 ServiceAccount
// token issuer, serving /.well-known/openid-configuration and /openid/v1/jwks
// while exposing its signing key so tests can mint both valid KEP-6060 tokens
// and adversarial tokens that kube-apiserver's TokenRequest API would refuse.
type ecdsaOIDCIssuer struct {
	server *httptest.Server
	issuer string
	signer jose.Signer
}

func newECDSAOIDCIssuer(t *testing.T) *ecdsaOIDCIssuer {
	t.Helper()
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate ECDSA P-256 signing key: %v", err)
	}
	const keyID = "apiserver-ecdsa-key-1"
	signer, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.ES256, Key: jose.JSONWebKey{Key: priv, KeyID: keyID}},
		(&jose.SignerOptions{}).WithType("JWT"),
	)
	if err != nil {
		t.Fatalf("create ES256 signer: %v", err)
	}
	jwks := jose.JSONWebKeySet{Keys: []jose.JSONWebKey{{
		Key:       priv.Public(),
		KeyID:     keyID,
		Algorithm: string(jose.ES256),
		Use:       "sig",
	}}}

	ts := &ecdsaOIDCIssuer{signer: signer}
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer":                                ts.issuer,
			"jwks_uri":                              ts.issuer + "/openid/v1/jwks",
			"response_types_supported":              []string{"id_token"},
			"subject_types_supported":               []string{"public"},
			"id_token_signing_alg_values_supported": []string{string(jose.ES256), string(jose.RS256)},
		})
	})
	mux.HandleFunc("/openid/v1/jwks", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(jwks)
	})
	ts.server = httptest.NewTLSServer(mux)
	ts.issuer = ts.server.URL
	t.Cleanup(ts.server.Close)
	return ts
}

func (ts *ecdsaOIDCIssuer) sign(t *testing.T, claims map[string]any) string {
	t.Helper()
	payload, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshal claims: %v", err)
	}
	jws, err := ts.signer.Sign(payload)
	if err != nil {
		t.Fatalf("sign claims: %v", err)
	}
	compact, err := jws.CompactSerialize()
	if err != nil {
		t.Fatalf("serialize JWS: %v", err)
	}
	return compact
}

// validWebhookClaims builds the exact JWT claim shape minted by kube-apiserver's
// pkg/serviceaccount.Claims when APIServerWebhookAuthenticationToken is enabled.
func (ts *ecdsaOIDCIssuer) validWebhookClaims(audience, bindingKind, configName, configUID string, groups []string) map[string]any {
	now := time.Now()
	k8sClaims := map[string]any{
		"namespace": "kube-system",
		"serviceaccount": map[string]any{
			"name": "webhook-authenticator",
			"uid":  "sa-uid-1111",
		},
		"attestations": map[string]any{
			admissionReviewAPIGroupsClaimKey: groups,
		},
	}
	if bindingKind != "" {
		k8sClaims[bindingKind] = map[string]any{
			"name": configName,
			"uid":  configUID,
		}
	}
	return map[string]any{
		"iss":           ts.issuer,
		"sub":           "system:serviceaccount:kube-system:webhook-authenticator",
		"aud":           []string{audience},
		"exp":           now.Add(5 * time.Minute).Unix(),
		"nbf":           now.Add(-1 * time.Minute).Unix(),
		"iat":           now.Unix(),
		"jti":           "jti-webhook-token-1",
		"kubernetes.io": k8sClaims,
	}
}

// forgeAlgNoneJWT builds an unsigned compact JWT ("alg":"none") for probing.
func forgeAlgNoneJWT(t *testing.T, claims map[string]any) string {
	t.Helper()
	hdr := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none","typ":"JWT"}`))
	payloadBytes, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshal alg:none claims: %v", err)
	}
	payload := base64.RawURLEncoding.EncodeToString(payloadBytes)
	return hdr + "." + payload + "."
}

// forgeHS256JWT builds a symmetric HS256 JWT for algorithm-confusion probing.
func forgeHS256JWT(t *testing.T, claims map[string]any) string {
	t.Helper()
	hdr := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"HS256","kid":"apiserver-ecdsa-key-1","typ":"JWT"}`))
	payloadBytes, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshal HS256 claims: %v", err)
	}
	payload := base64.RawURLEncoding.EncodeToString(payloadBytes)
	mac := hmac.New(sha256.New, []byte("probing-symmetric-secret"))
	mac.Write([]byte(hdr + "." + payload))
	sig := base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
	return hdr + "." + payload + "." + sig
}

// TestDummyWebhookEndToEnd exercises real signing and TLS in both verification modes.
func TestDummyWebhookEndToEnd(t *testing.T) {
	issuer := newECDSAOIDCIssuer(t)
	otherIssuer := newECDSAOIDCIssuer(t)

	validateAudience := verify.AudienceForService("dummy-webhook", "webhook-system", 443, "/validate")
	mutateAudience := verify.AudienceForService("dummy-webhook", "webhook-system", 443, "/mutate")

	modes := []struct {
		name        string
		newVerifier func(string) (*admission.Verifier, error)
	}{
		{
			name: "in-cluster",
			newVerifier: func(audience string) (*admission.Verifier, error) {
				projected := issuer.sign(t, map[string]any{
					"iss": issuer.issuer,
					"sub": "system:serviceaccount:webhook-system:receiver",
					"aud": []string{"https://kubernetes.default.svc"},
					"exp": time.Now().Add(time.Hour).Unix(),
				})
				config := &rest.Config{
					Host:        issuer.issuer,
					BearerToken: projected,
					TLSClientConfig: rest.TLSClientConfig{
						CAData: pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: issuer.server.Certificate().Raw}),
					},
				}
				return admission.NewLocalVerifierForTest(context.Background(), config, audience)
			},
		},
		{
			name: "remote-issuer",
			newVerifier: func(audience string) (*admission.Verifier, error) {
				return admission.NewVerifier(context.Background(), admission.WithRemoteIssuer(issuer.issuer, audience, issuer.server.Client()))
			},
		},
	}

	for _, mode := range modes {
		t.Run(mode.name, func(t *testing.T) {
			var validateCalls, mutateCalls atomic.Int32

			validateVerifier, err := mode.newVerifier(validateAudience)
			validateHandler := admission.NewHandlerForTest(
				validateVerifier,
				func(_ context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
					validateCalls.Add(1)
					return &admissionv1.AdmissionResponse{
						Allowed: true,
						Result:  &metav1.Status{Status: metav1.StatusSuccess, Message: "validated:" + req.Resource.Resource},
					}
				},
				admission.WithMaxBodyBytesForTest(4096),
			)
			if err != nil {
				t.Fatalf("construct /validate handler: %v", err)
			}

			patchType := admissionv1.PatchTypeJSONPatch
			mutateVerifier, err := mode.newVerifier(mutateAudience)
			mutateHandler := admission.NewHandlerForTest(
				mutateVerifier,
				func(_ context.Context, _ *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
					mutateCalls.Add(1)
					return &admissionv1.AdmissionResponse{
						Allowed:   true,
						PatchType: &patchType,
						Patch:     []byte(`[{"op":"add","path":"/metadata/annotations","value":{"kep6060":"verified"}}]`),
					}
				},
				admission.WithMaxBodyBytesForTest(4096),
			)
			if err != nil {
				t.Fatalf("construct /mutate handler: %v", err)
			}

			mux := http.NewServeMux()
			mux.Handle("/validate", validateHandler)
			mux.Handle("/mutate", mutateHandler)
			webhookServer := httptest.NewTLSServer(mux)
			t.Cleanup(webhookServer.Close)

			appsReview := func(uid types.UID) []byte {
				b, err := json.Marshal(&admissionv1.AdmissionReview{
					TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
					Request: &admissionv1.AdmissionRequest{
						UID:       uid,
						Kind:      metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
						Resource:  metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
						Operation: admissionv1.Create,
						Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":"demo"}}`)},
					},
				})
				if err != nil {
					t.Fatalf("marshal apps review: %v", err)
				}
				return b
			}

			validToken := issuer.sign(t, issuer.validWebhookClaims(
				validateAudience,
				"validatingwebhookconfiguration",
				"dummy-validating-cfg",
				"vwc-uid-1234",
				[]string{"apps"},
			))

			// 1. Pre-decode probing matrix: every probe must return HTTP 401 with
			// the exact same response body and never reach the handler.
			preDecodeProbes := []struct {
				name       string
				method     string
				authHeader string
				body       []byte
				wantStatus int
			}{
				{
					name:   "missing Authorization header",
					method: http.MethodPost,
					body:   appsReview("probe-1"),
				},
				{
					name:       "Basic auth scheme instead of Bearer",
					method:     http.MethodPost,
					authHeader: "Basic dXNlcjpwYXNz",
					body:       appsReview("probe-2"),
				},
				{
					name:       "Bearer with empty token",
					method:     http.MethodPost,
					authHeader: "Bearer ",
					body:       appsReview("probe-3"),
				},
				{
					name:       "Bearer with extra leading space",
					method:     http.MethodPost,
					authHeader: "Bearer  " + validToken,
					body:       appsReview("probe-4"),
				},
				{
					name:       "GET probe with valid token and no body",
					method:     http.MethodGet,
					authHeader: "Bearer " + validToken,
					body:       nil,
				},
				{
					name:       "DELETE probe with valid token and no body",
					method:     http.MethodDelete,
					authHeader: "Bearer " + validToken,
					body:       nil,
				},
				{
					name:       "malformed JSON body with valid token",
					method:     http.MethodPost,
					authHeader: "Bearer " + validToken,
					body:       []byte(`{"not":"an admission review"`),
				},
				{
					name:       "AdmissionReview with nil Request and valid token",
					method:     http.MethodPost,
					authHeader: "Bearer " + validToken,
					body:       []byte(`{"apiVersion":"admission.k8s.io/v1","kind":"AdmissionReview"}`),
				},
				{
					name:       "oversized body with invalid token",
					method:     http.MethodPost,
					authHeader: "Bearer not-a-jwt",
					body:       bytes.Repeat([]byte("A"), 5000),
				},
				{
					name:       "oversized body exceeding maxBodyBytes with valid token",
					wantStatus: http.StatusRequestEntityTooLarge,
					method:     http.MethodPost,
					authHeader: "Bearer " + validToken,
					body:       []byte(`{"apiVersion":"admission.k8s.io/v1","kind":"AdmissionReview","request":{"uid":"big","object":{"apiVersion":"apps/v1","kind":"Deployment","pad":"` + strings.Repeat("A", 5000) + `"}}}`),
				},
			}

			var canonical401Body string
			for i, probe := range preDecodeProbes {
				t.Run("pre-decode/"+probe.name, func(t *testing.T) {
					before := validateCalls.Load()
					var bodyReader io.Reader
					if probe.body != nil {
						bodyReader = bytes.NewReader(probe.body)
					}
					req, err := http.NewRequestWithContext(context.Background(), probe.method, webhookServer.URL+"/validate", bodyReader)
					if err != nil {
						t.Fatalf("new request: %v", err)
					}
					req.Header.Set("Content-Type", "application/json")
					if probe.authHeader != "" {
						req.Header.Set("Authorization", probe.authHeader)
					}
					resp, err := webhookServer.Client().Do(req)
					if err != nil {
						t.Fatalf("execute request: %v", err)
					}
					defer resp.Body.Close()
					rawBytes, err := io.ReadAll(resp.Body)
					if err != nil {
						t.Fatalf("read response body: %v", err)
					}
					wantStatus := probe.wantStatus
					if wantStatus == 0 {
						wantStatus = http.StatusUnauthorized
					}
					if resp.StatusCode != wantStatus {
						t.Fatalf("status = %d, want %d; body = %q", resp.StatusCode, wantStatus, rawBytes)
					}
					if validateCalls.Load() != before {
						t.Fatal("downstream handler was invoked on pre-decode rejection")
					}
					if wantStatus == http.StatusRequestEntityTooLarge {
						if string(rawBytes) != "request body exceeds the configured limit\n" {
							t.Fatalf("oversized response = %q", rawBytes)
						}
						return
					}
					if string(rawBytes) != "webhook token verification failed\n" {
						t.Fatalf("body = %q, want generic 401 message", string(rawBytes))
					}
					if i == 0 {
						canonical401Body = string(rawBytes)
					} else if string(rawBytes) != canonical401Body {
						t.Fatalf("pre-decode response body %q differs from canonical %q (oracle leak)", string(rawBytes), canonical401Body)
					}
				})
			}

			// Authentication failures are opaque; authenticated policy denials echo UID.
			const fixedProbeUID = types.UID("anti-oracle-probe-uid")
			defaultAppsRequest := func() *admissionv1.AdmissionRequest {
				return &admissionv1.AdmissionRequest{
					UID:       fixedProbeUID,
					Kind:      metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
					Resource:  metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
					Operation: admissionv1.Create,
					Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":"demo"}}`)},
				}
			}

			postDecodeProbes := []struct {
				name         string
				host         string
				token        func() string
				req          func() *admissionv1.AdmissionRequest
				policyDenial bool
			}{
				{
					name:  "non-JWT opaque token",
					token: func() string { return "not-a-jwt-token" },
					req:   defaultAppsRequest,
				},
				{
					name: "unsigned alg:none JWT",
					token: func() string {
						return forgeAlgNoneJWT(t, issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "symmetric HS256 algorithm confusion JWT",
					token: func() string {
						return forgeHS256JWT(t, issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "signed by untrusted key with same issuer claim",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"})
						return otherIssuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "untrusted issuer claim",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"})
						c["iss"] = "https://evil-issuer.example.com"
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "wrong audience (token minted for /mutate sent to /validate)",
					token: func() string {
						return issuer.sign(t, issuer.validWebhookClaims(mutateAudience, "mutatingwebhookconfiguration", "mwc", "uid-1", []string{"apps"}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "spoofed HTTP Host header attempting to match wrong token audience",
					host: "other-webhook.webhook-system.svc:443",
					token: func() string {
						otherAud := verify.AudienceForService("other-webhook", "webhook-system", 443, "/validate")
						return issuer.sign(t, issuer.validWebhookClaims(otherAud, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "expired token",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"})
						c["exp"] = time.Now().Add(-2 * time.Minute).Unix()
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "not-before in the future",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"})
						c["nbf"] = time.Now().Add(10 * time.Minute).Unix()
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "ordinary Pod-bound ServiceAccount token",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "pod", "my-pod", "pod-uid-1", []string{"apps"})
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "ordinary Secret-bound ServiceAccount token",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "secret", "my-secret", "sec-uid-1", []string{"apps"})
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "ordinary Node-bound ServiceAccount token",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "node", "node-1", "node-uid-1", []string{"apps"})
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "unbound ServiceAccount token (no webhook binding)",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "", "", "", []string{"apps"})
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "token bound to both Validating and Mutating webhook configurations",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"})
						c["kubernetes.io"].(map[string]any)["mutatingwebhookconfiguration"] = map[string]any{"name": "mwc", "uid": "uid-2"}
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "webhook binding missing UID",
					token: func() string {
						return issuer.sign(t, issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "", []string{"apps"}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "webhook binding missing Name",
					token: func() string {
						return issuer.sign(t, issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "", "uid-1", []string{"apps"}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "missing admissionReviewAPIGroups attestation",
					token: func() string {
						c := issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"})
						delete(c["kubernetes.io"].(map[string]any), "attestations")
						return issuer.sign(t, c)
					},
					req: defaultAppsRequest,
				},
				{
					name: "empty admissionReviewAPIGroups slice",
					token: func() string {
						return issuer.sign(t, issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "empty-string element in admissionReviewAPIGroups slice",
					token: func() string {
						return issuer.sign(t, issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{""}))
					},
					req: defaultAppsRequest,
				},
				{
					name: "multi-valued admissionReviewAPIGroups slice (disallowed by KEP-6060)",
					token: func() string {
						return issuer.sign(t, issuer.validWebhookClaims(validateAudience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps", "batch"}))
					},
					req: defaultAppsRequest,
				},
				{
					name:         "unauthorized API group (token authorizes apps, request is batch)",
					policyDenial: true,
					token:        func() string { return validToken },
					req: func() *admissionv1.AdmissionRequest {
						return &admissionv1.AdmissionRequest{
							UID:       fixedProbeUID,
							Kind:      metav1.GroupVersionKind{Group: "batch", Version: "v1", Kind: "Job"},
							Resource:  metav1.GroupVersionResource{Group: "batch", Version: "v1", Resource: "jobs"},
							Operation: admissionv1.Create,
							Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"batch/v1","kind":"Job"}`)},
						}
					},
				},
				{
					name:         "unauthorized core API group (token authorizes apps, request is core Pod)",
					policyDenial: true,
					token:        func() string { return validToken },
					req: func() *admissionv1.AdmissionRequest {
						return &admissionv1.AdmissionRequest{
							UID:       fixedProbeUID,
							Kind:      metav1.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
							Resource:  metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
							Operation: admissionv1.Create,
							Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"v1","kind":"Pod"}`)},
						}
					},
				},
				{
					name:         "subresource group spoofing (deployments/scale carrying rbac Role)",
					policyDenial: true,
					token:        func() string { return validToken },
					req: func() *admissionv1.AdmissionRequest {
						return &admissionv1.AdmissionRequest{
							UID:         fixedProbeUID,
							Kind:        metav1.GroupVersionKind{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "Role"},
							Resource:    metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
							SubResource: "scale",
							Operation:   admissionv1.Update,
							Object:      runtime.RawExtension{Raw: []byte(`{"apiVersion":"rbac.authorization.k8s.io/v1","kind":"Role"}`)},
							OldObject:   runtime.RawExtension{Raw: []byte(`{"apiVersion":"rbac.authorization.k8s.io/v1","kind":"Role"}`)},
						}
					},
				},
				{
					name:         "raw Object group mismatches AdmissionRequest.Resource.Group",
					policyDenial: true,
					token:        func() string { return validToken },
					req: func() *admissionv1.AdmissionRequest {
						r := defaultAppsRequest()
						r.Object = runtime.RawExtension{Raw: []byte(`{"apiVersion":"v1","kind":"Secret","metadata":{"name":"stolen"}}`)}
						return r
					},
				},
				{
					name:         "raw OldObject group mismatches AdmissionRequest.Resource.Group",
					policyDenial: true,
					token:        func() string { return validToken },
					req: func() *admissionv1.AdmissionRequest {
						r := defaultAppsRequest()
						r.Operation = admissionv1.Update
						r.OldObject = runtime.RawExtension{Raw: []byte(`{"apiVersion":"v1","kind":"Secret","metadata":{"name":"stolen"}}`)}
						return r
					},
				},
			}

			var canonicalDenialBytes []byte
			for _, probe := range postDecodeProbes {
				t.Run("post-decode/"+probe.name, func(t *testing.T) {
					before := validateCalls.Load()
					reviewBytes, err := json.Marshal(&admissionv1.AdmissionReview{
						TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
						Request:  probe.req(),
					})
					if err != nil {
						t.Fatalf("marshal probe review: %v", err)
					}
					req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, webhookServer.URL+"/validate", bytes.NewReader(reviewBytes))
					if err != nil {
						t.Fatalf("new request: %v", err)
					}
					if probe.host != "" {
						req.Host = probe.host
					}
					req.Header.Set("Content-Type", "application/json")
					req.Header.Set("Authorization", "Bearer "+probe.token())

					resp, err := webhookServer.Client().Do(req)
					if err != nil {
						t.Fatalf("execute request: %v", err)
					}
					defer resp.Body.Close()
					rawBytes, err := io.ReadAll(resp.Body)
					if err != nil {
						t.Fatalf("read response body: %v", err)
					}
					if validateCalls.Load() != before {
						t.Fatal("downstream handler was invoked on rejection")
					}
					if !probe.policyDenial {
						if resp.StatusCode != http.StatusUnauthorized || string(rawBytes) != canonical401Body {
							t.Fatalf("authentication denial = (%d, %q), want generic 401", resp.StatusCode, rawBytes)
						}
						return
					}
					if resp.StatusCode != http.StatusOK {
						t.Fatalf("status = %d, want 200; body = %q", resp.StatusCode, rawBytes)
					}
					var decoded admissionv1.AdmissionReview
					if err := json.Unmarshal(rawBytes, &decoded); err != nil {
						t.Fatalf("decode AdmissionReview response: %v", err)
					}
					if decoded.Response == nil || decoded.Response.Allowed {
						t.Fatalf("expected explicit Allowed:false denial, got %+v", decoded.Response)
					}
					if decoded.Response.UID != fixedProbeUID {
						t.Fatalf("UID = %q, want %q", decoded.Response.UID, fixedProbeUID)
					}
					if decoded.Response.Result == nil || decoded.Response.Result.Code != http.StatusUnauthorized || decoded.Response.Result.Message != "webhook token verification failed" {
						t.Fatalf("unexpected denial status: %+v", decoded.Response.Result)
					}
					if canonicalDenialBytes == nil {
						canonicalDenialBytes = rawBytes
					} else if !bytes.Equal(rawBytes, canonicalDenialBytes) {
						t.Fatalf("post-decode denial bytes %q differ from canonical %q (oracle leak)", string(rawBytes), string(canonicalDenialBytes))
					}
				})
			}

			// 3. Positive exact-scope matrix: verify that valid tokens for both
			// validating and mutating webhooks succeed across named groups, core "",
			// wildcard "*", and the three legitimate cross-group subresources.
			positiveCases := []struct {
				name        string
				path        string
				audience    string
				bindingKind string
				attGroup    string
				req         *admissionv1.AdmissionRequest
				wantMutated bool
			}{
				{
					name:        "validating apps/v1 Deployment",
					path:        "/validate",
					audience:    validateAudience,
					bindingKind: "validatingwebhookconfiguration",
					attGroup:    "apps",
					req:         defaultAppsRequest(),
				},
				{
					name:        "mutating apps/v1 Deployment",
					path:        "/mutate",
					audience:    mutateAudience,
					bindingKind: "mutatingwebhookconfiguration",
					attGroup:    "apps",
					req:         defaultAppsRequest(),
					wantMutated: true,
				},
				{
					name:        "validating core v1 Pod with wildcard attestation",
					path:        "/validate",
					audience:    validateAudience,
					bindingKind: "validatingwebhookconfiguration",
					attGroup:    "*",
					req: &admissionv1.AdmissionRequest{
						UID:       "pos-core-pod",
						Kind:      metav1.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
						Resource:  metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
						Operation: admissionv1.Create,
						Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"v1","kind":"Pod","metadata":{"name":"p1"}}`)},
					},
				},
				{
					name:        "validating deployments/scale cross-group subresource",
					path:        "/validate",
					audience:    validateAudience,
					bindingKind: "validatingwebhookconfiguration",
					attGroup:    "apps",
					req: &admissionv1.AdmissionRequest{
						UID:         "pos-scale",
						Kind:        metav1.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"},
						Resource:    metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
						SubResource: "scale",
						Operation:   admissionv1.Update,
						Object:      runtime.RawExtension{Raw: []byte(`{"apiVersion":"autoscaling/v1","kind":"Scale"}`)},
						OldObject:   runtime.RawExtension{Raw: []byte(`{"apiVersion":"autoscaling/v1","kind":"Scale"}`)},
					},
				},
				{
					name:        "validating pods/eviction cross-group subresource",
					path:        "/validate",
					audience:    validateAudience,
					bindingKind: "validatingwebhookconfiguration",
					attGroup:    "*",
					req: &admissionv1.AdmissionRequest{
						UID:         "pos-eviction",
						Kind:        metav1.GroupVersionKind{Group: "policy", Version: "v1", Kind: "Eviction"},
						Resource:    metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
						SubResource: "eviction",
						Operation:   admissionv1.Create,
						Object:      runtime.RawExtension{Raw: []byte(`{"apiVersion":"policy/v1","kind":"Eviction"}`)},
					},
				},
				{
					name:        "validating serviceaccounts/token cross-group subresource",
					path:        "/validate",
					audience:    validateAudience,
					bindingKind: "validatingwebhookconfiguration",
					attGroup:    "*",
					req: &admissionv1.AdmissionRequest{
						UID:         "pos-satoken",
						Kind:        metav1.GroupVersionKind{Group: "authentication.k8s.io", Version: "v1", Kind: "TokenRequest"},
						Resource:    metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "serviceaccounts"},
						SubResource: "token",
						Operation:   admissionv1.Create,
						Object:      runtime.RawExtension{Raw: []byte(`{"apiVersion":"authentication.k8s.io/v1","kind":"TokenRequest"}`)},
					},
				},
			}

			for _, tc := range positiveCases {
				if tc.req == nil {
					continue
				}
				t.Run("positive/"+tc.name, func(t *testing.T) {
					token := issuer.sign(t, issuer.validWebhookClaims(tc.audience, tc.bindingKind, "cfg-name", "cfg-uid", []string{tc.attGroup}))
					reviewBytes, err := json.Marshal(&admissionv1.AdmissionReview{
						TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
						Request:  tc.req,
					})
					if err != nil {
						t.Fatalf("marshal positive review: %v", err)
					}
					req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, webhookServer.URL+tc.path, bytes.NewReader(reviewBytes))
					if err != nil {
						t.Fatalf("new request: %v", err)
					}
					req.Header.Set("Content-Type", "application/json")
					req.Header.Set("Authorization", "Bearer "+token)

					resp, err := webhookServer.Client().Do(req)
					if err != nil {
						t.Fatalf("execute request: %v", err)
					}
					defer resp.Body.Close()
					if resp.StatusCode != http.StatusOK {
						t.Fatalf("status = %d, want 200", resp.StatusCode)
					}
					var decoded admissionv1.AdmissionReview
					if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
						t.Fatalf("decode response: %v", err)
					}
					if decoded.Response == nil || !decoded.Response.Allowed || decoded.Response.UID != tc.req.UID {
						t.Fatalf("unexpected positive response: %+v", decoded.Response)
					}
					if tc.wantMutated && len(decoded.Response.Patch) == 0 {
						t.Fatalf("expected JSONPatch on mutating response, got none")
					}
				})
			}
		})
	}
}

// TestTokenReuseDoesNotSkipRequestChecks covers policy after library key-cache reuse.
func TestTokenReuseDoesNotSkipRequestChecks(t *testing.T) {
	issuer := newECDSAOIDCIssuer(t)
	audience := verify.AudienceForService("dummy-webhook", "webhook-system", 443, "/validate")

	var admitted atomic.Int32
	handler, err := admission.WithTokenVerification(
		context.Background(),
		func(_ context.Context, _ *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
			admitted.Add(1)
			return &admissionv1.AdmissionResponse{Allowed: true}
		},
		admission.WithRemoteIssuer(issuer.issuer, audience, issuer.server.Client()),
	)
	if err != nil {
		t.Fatalf("construct handler: %v", err)
	}
	srv := httptest.NewTLSServer(handler)
	t.Cleanup(srv.Close)

	token := issuer.sign(t, issuer.validWebhookClaims(audience, "validatingwebhookconfiguration", "vwc", "uid-1", []string{"apps"}))

	send := func(req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		t.Helper()
		b, err := json.Marshal(&admissionv1.AdmissionReview{
			TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
			Request:  req,
		})
		if err != nil {
			t.Fatalf("marshal review: %v", err)
		}
		httpReq, err := http.NewRequestWithContext(context.Background(), http.MethodPost, srv.URL, bytes.NewReader(b))
		if err != nil {
			t.Fatalf("new request: %v", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+token)
		resp, err := srv.Client().Do(httpReq)
		if err != nil {
			t.Fatalf("do request: %v", err)
		}
		defer resp.Body.Close()
		var out admissionv1.AdmissionReview
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatalf("decode response: %v", err)
		}
		return out.Response
	}

	validApps := &admissionv1.AdmissionRequest{
		UID:       "req-1",
		Kind:      metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
		Resource:  metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
		Operation: admissionv1.Create,
		Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"apps/v1","kind":"Deployment"}`)},
	}

	// Prime the library's key cache.
	if resp := send(validApps); resp == nil || !resp.Allowed || admitted.Load() != 1 {
		t.Fatalf("first valid request failed: %+v", resp)
	}

	// Present the exact same reused token on a batch/v1 request: must be denied.
	unauthorizedBatch := &admissionv1.AdmissionRequest{
		UID:       "req-2",
		Kind:      metav1.GroupVersionKind{Group: "batch", Version: "v1", Kind: "Job"},
		Resource:  metav1.GroupVersionResource{Group: "batch", Version: "v1", Resource: "jobs"},
		Operation: admissionv1.Create,
		Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"batch/v1","kind":"Job"}`)},
	}
	if resp := send(unauthorizedBatch); resp == nil || resp.Allowed || admitted.Load() != 1 {
		t.Fatalf("reused token must not authorize batch/v1: %+v", resp)
	}

	// Present the exact same reused token on an apps/v1 request whose embedded
	// Object is a core v1 Secret: must be denied.
	spoofedObject := &admissionv1.AdmissionRequest{
		UID:       "req-3",
		Kind:      metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
		Resource:  metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
		Operation: admissionv1.Create,
		Object:    runtime.RawExtension{Raw: []byte(`{"apiVersion":"v1","kind":"Secret"}`)},
	}
	if resp := send(spoofedObject); resp == nil || resp.Allowed || admitted.Load() != 1 {
		t.Fatalf("reused token must not allow spoofed Object GVK: %+v", resp)
	}

	// Follow-up valid apps/v1 request with the reused token still succeeds.
	validApps.UID = "req-4"
	if resp := send(validApps); resp == nil || !resp.Allowed || admitted.Load() != 2 {
		t.Fatalf("second valid request failed: %+v", resp)
	}
}

func TestVerifier_SubresourceKindAndObjectGroups(t *testing.T) {
	ts := newOIDCTestServer(t)
	verifier, err := admission.NewVerifier(context.Background(), admission.WithRemoteIssuer(ts.issuer, testAudience, ts.client()))
	if err != nil {
		t.Fatalf("construct token verifier: %v", err)
	}
	token := ts.sign(t, ts.baseClaims())
	appsResource := metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}
	appsKind := metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}
	scaleKind := metav1.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"}
	for _, tc := range []struct {
		name     string
		kind     metav1.GroupVersionKind
		sub      string
		object   string
		old      string
		wantDeny bool
	}{
		{
			name: "legitimate deployments scale subresource",
			kind: scaleKind, sub: "scale", object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`,
		},
		{
			name: "ordinary resource cannot switch kind groups",
			kind: scaleKind, object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`, wantDeny: true,
		},
		{
			name: "forged scale subresource cannot carry a Role",
			kind: metav1.GroupVersionKind{Group: "rbac.authorization.k8s.io", Version: "v1", Kind: "Role"},
			sub:  "scale", object: `{"apiVersion":"rbac.authorization.k8s.io/v1","kind":"Role"}`, wantDeny: true,
		},
		{
			name: "unknown subresource cannot switch to autoscaling group",
			kind: scaleKind, sub: "other", object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`, wantDeny: true,
		},
		{
			name: "subresource object differs from admitted Scale kind",
			kind: scaleKind, sub: "scale", object: `{"apiVersion":"apps/v1","kind":"Deployment"}`, wantDeny: true,
		},
		{
			name: "subresource without a kind cannot claim another object group",
			sub:  "scale", object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`, wantDeny: true,
		},
		{
			name: "old object mismatches scale kind",
			kind: scaleKind, sub: "scale", object: `{"apiVersion":"autoscaling/v1","kind":"Scale"}`,
			old: `{"apiVersion":"apps/v1","kind":"Deployment"}`, wantDeny: true,
		},
		{
			name: "ordinary object Kind disagrees with AdmissionRequest Kind",
			kind: appsKind, object: `{"apiVersion":"apps/v1","kind":"Pod"}`, wantDeny: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			operation := admissionv1.Create
			if tc.sub == "scale" || tc.old != "" {
				operation = admissionv1.Update
			}
			request := admissionRequest(tc.kind, appsResource, tc.sub, operation)
			request.Object.Raw = []byte(tc.object)
			if operation == admissionv1.Update {
				request.OldObject.Raw = bytes.Clone(request.Object.Raw)
			}
			if tc.old != "" {
				request.OldObject.Raw = []byte(tc.old)
			}
			err := verifier.Verify(context.Background(), request, token)
			if tc.wantDeny {
				if err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("verification = %v, want generic denial", err)
				}
			} else if err != nil {
				t.Fatalf("valid subresource request denied: %v", err)
			}
		})
	}
}

func TestVerifier_DecodedObjectsMatchAdmittedKind(t *testing.T) {
	ts := newOIDCTestServer(t)
	verifier, err := admission.NewVerifier(context.Background(), admission.WithRemoteIssuer(ts.issuer, testAudience, ts.client()))
	if err != nil {
		t.Fatalf("construct token verifier: %v", err)
	}
	token := ts.sign(t, ts.baseClaims())
	object := func(apiVersion, kind string) runtime.Object {
		return &unstructured.Unstructured{Object: map[string]any{"apiVersion": apiVersion, "kind": kind}}
	}
	for _, tc := range []struct {
		name     string
		current  runtime.Object
		old      runtime.Object
		wantDeny bool
	}{
		{name: "matching current and old objects", current: object("apps/v1", "Deployment"), old: object("apps/v1", "Deployment")},
		{name: "wrong current group", current: object("batch/v1", "Job"), wantDeny: true},
		{name: "wrong old group", current: object("apps/v1", "Deployment"), old: object("batch/v1", "Job"), wantDeny: true},
		{name: "missing object type metadata", current: &unstructured.Unstructured{Object: map[string]any{"metadata": map[string]any{"name": "missing"}}}, wantDeny: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			operation := admissionv1.Create
			if tc.old != nil {
				operation = admissionv1.Update
			}
			req := admissionRequest(
				metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
				metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
				"", operation,
			)
			req.Object = runtime.RawExtension{Object: tc.current}
			if tc.old != nil {
				req.OldObject = runtime.RawExtension{Object: tc.old}
			}
			err := verifier.Verify(context.Background(), req, token)
			if tc.wantDeny {
				if err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("verification = %v, want generic denial", err)
				}
			} else if err != nil {
				t.Fatalf("matching decoded objects denied: %v", err)
			}
		})
	}
}

func TestVerifier_CoreCrossGroupSubresources(t *testing.T) {
	ts := newOIDCTestServer(t)
	verifier, err := admission.NewVerifier(context.Background(), admission.WithRemoteIssuer(ts.issuer, testAudience, ts.client()))
	if err != nil {
		t.Fatalf("construct token verifier: %v", err)
	}
	claims := ts.baseClaims()
	claims["kubernetes.io"].(map[string]any)["attestations"] = map[string]any{
		admissionReviewAPIGroupsClaimKey: []string{"*"},
	}
	token := ts.sign(t, claims)
	for _, tc := range []struct {
		name, resource, subresource, group, kind, object string
		wantDeny                                         bool
	}{
		{
			name: "Pod eviction", resource: "pods", subresource: "eviction", group: "policy", kind: "Eviction",
			object: `{"apiVersion":"policy/v1","kind":"Eviction"}`,
		},
		{
			name: "ServiceAccount token", resource: "serviceaccounts", subresource: "token", group: "authentication.k8s.io", kind: "TokenRequest",
			object: `{"apiVersion":"authentication.k8s.io/v1","kind":"TokenRequest"}`,
		},
		{
			name: "Pod eviction cannot claim a TokenRequest Kind", resource: "pods", subresource: "eviction", group: "authentication.k8s.io", kind: "TokenRequest",
			object: `{"apiVersion":"authentication.k8s.io/v1","kind":"TokenRequest"}`, wantDeny: true,
		},
		{
			name: "Secret cannot spoof a Pod eviction", resource: "secrets", subresource: "eviction", group: "policy", kind: "Eviction",
			object: `{"apiVersion":"policy/v1","kind":"Eviction"}`, wantDeny: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := admissionRequest(
				metav1.GroupVersionKind{Group: tc.group, Version: "v1", Kind: tc.kind},
				metav1.GroupVersionResource{Version: "v1", Resource: tc.resource},
				tc.subresource, admissionv1.Create,
			)
			request.Object.Raw = []byte(tc.object)
			err := verifier.Verify(context.Background(), request, token)
			if tc.wantDeny {
				if err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("cross-group spoof: verification = %v, want generic denial", err)
				}
			} else if err != nil {
				t.Fatalf("legitimate cross-group subresource denied: %v", err)
			}
		})
	}
}

func TestVerifier_RejectsMissingObjectTypeMetaForCoreGroup(t *testing.T) {
	ts := newOIDCTestServer(t)
	verifier, err := admission.NewVerifier(context.Background(), admission.WithRemoteIssuer(ts.issuer, testAudience, ts.client()))
	if err != nil {
		t.Fatalf("construct token verifier: %v", err)
	}
	claims := ts.baseClaims()
	claims["kubernetes.io"].(map[string]any)["attestations"] = map[string]any{
		admissionReviewAPIGroupsClaimKey: []string{"*"},
	}
	token := ts.sign(t, claims)
	for _, tc := range []struct{ name, object string }{
		{name: "JSON null", object: `null`},
		{name: "empty object", object: `{}`},
		{name: "versionless apiVersion", object: `{"apiVersion":"/","kind":"Pod"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := admissionRequest(
				metav1.GroupVersionKind{Version: "v1", Kind: "Pod"},
				metav1.GroupVersionResource{Version: "v1", Resource: "pods"},
				"", admissionv1.Create,
			)
			request.Object.Raw = []byte(tc.object)
			if err := verifier.Verify(context.Background(), request, token); err == nil || err.Error() != "webhook token verification failed" {
				t.Fatalf("missing object type metadata: got %v, want generic denial", err)
			}
		})
	}
}

func TestVerifierOperationShapes(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	claims := ts.baseClaims()
	claims["kubernetes.io"].(map[string]any)["attestations"] = map[string]any{admissionReviewAPIGroupsClaimKey: []string{"*"}}
	token := ts.sign(t, claims)
	appsKind := metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}
	appsResource := metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}
	for _, tc := range []struct {
		name      string
		operation admissionv1.Operation
		mutate    func(*admissionv1.AdmissionRequest)
		deny      bool
	}{
		{name: "create", operation: admissionv1.Create},
		{name: "create missing object", operation: admissionv1.Create, deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Object = runtime.RawExtension{} }},
		{name: "create with old object", operation: admissionv1.Create, deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.OldObject = r.Object }},
		{name: "update", operation: admissionv1.Update},
		{name: "update missing current", operation: admissionv1.Update, deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Object = runtime.RawExtension{} }},
		{name: "update missing old", operation: admissionv1.Update, deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.OldObject = runtime.RawExtension{} }},
		{name: "delete old only", operation: admissionv1.Delete},
		{name: "delete missing old", operation: admissionv1.Delete, deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.OldObject = runtime.RawExtension{} }},
		{name: "delete with current", operation: admissionv1.Delete, deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Object = r.OldObject }},
		{name: "connect with options", operation: admissionv1.Connect, mutate: func(r *admissionv1.AdmissionRequest) {
			*r = *admissionRequest(metav1.GroupVersionKind{Version: "v1", Kind: "PodExecOptions"}, metav1.GroupVersionResource{Version: "v1", Resource: "pods"}, "exec", admissionv1.Connect)
		}},
		{name: "connect without options", operation: admissionv1.Connect, mutate: func(r *admissionv1.AdmissionRequest) { r.Object = runtime.RawExtension{} }},
		{name: "connect with old object", operation: admissionv1.Connect, deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.OldObject = r.Object }},
		{name: "core group", operation: admissionv1.Create, mutate: func(r *admissionv1.AdmissionRequest) {
			*r = *admissionRequest(metav1.GroupVersionKind{Version: "v1", Kind: "Pod"}, metav1.GroupVersionResource{Version: "v1", Resource: "pods"}, "", admissionv1.Create)
		}},
		{name: "CRD scale version differs from parent", operation: admissionv1.Update, mutate: func(r *admissionv1.AdmissionRequest) {
			*r = *admissionRequest(metav1.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"}, metav1.GroupVersionResource{Group: "example.com", Version: "v2", Resource: "widgets"}, "scale", admissionv1.Update)
		}},
		{name: "missing operation", operation: admissionv1.Create, deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Operation = "" }},
		{name: "unknown operation", operation: admissionv1.Create, deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Operation = "UNKNOWN" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := admissionRequest(appsKind, appsResource, "", tc.operation)
			if tc.mutate != nil {
				tc.mutate(req)
			}
			err := v.Verify(context.Background(), req, token)
			if tc.deny {
				if err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("Verify = %v, want generic denial", err)
				}
			} else if err != nil {
				t.Fatalf("legitimate operation denied: %v", err)
			}
			raw, err := json.Marshal(&admissionv1.AdmissionReview{TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"}, Request: req})
			if err != nil {
				t.Fatal(err)
			}
			spy := newSpyHandler()
			h := admission.NewHandlerForTest(v, spy.serve)
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, newRequest(t, raw, token))
			var result admissionv1.AdmissionReview
			if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if rec.Code != http.StatusOK || result.Response == nil || result.Response.Allowed == tc.deny || result.Response.UID != req.UID || spy.wasReached() == tc.deny {
				t.Fatalf("HTTP verification: status=%d response=%+v callback=%t", rec.Code, result.Response, spy.wasReached())
			}
		})
	}
}

type metadataOnlyObject struct {
	metav1.TypeMeta
	NonJSON chan struct{} `json:"nonJSON"`
}

func (o *metadataOnlyObject) DeepCopyObject() runtime.Object {
	if o == nil {
		return nil
	}
	copy := *o
	return &copy
}

func TestVerifierMetadataRepresentations(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	token := ts.sign(t, ts.baseClaims())
	decoded := func(apiVersion, kind string) runtime.Object {
		return &unstructured.Unstructured{Object: map[string]any{"apiVersion": apiVersion, "kind": kind}}
	}
	for _, tc := range []struct {
		name   string
		mutate func(*admissionv1.AdmissionRequest)
		deny   bool
	}{
		{name: "complete metadata"},
		{name: "missing kind", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Kind = metav1.GroupVersionKind{} }},
		{name: "missing kind version", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Kind.Version = "" }},
		{name: "missing kind name", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Kind.Kind = "" }},
		{name: "missing resource version", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Resource.Version = "" }},
		{name: "missing resource name", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Resource.Resource = "" }},
		{name: "null object", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Object.Raw = []byte(" \nnull\t ") }},
		{name: "malformed raw object", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Object.Raw = []byte("{") }},
		{name: "missing object version", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Object.Raw = []byte(`{"kind":"Deployment"}`) }},
		{name: "missing object kind", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Object.Raw = []byte(`{"apiVersion":"apps/v1"}`) }},
		{name: "wrong object version", deny: true, mutate: func(r *admissionv1.AdmissionRequest) {
			r.Object.Raw = []byte(`{"apiVersion":"apps/v2","kind":"Deployment"}`)
		}},
		{name: "wrong object kind", deny: true, mutate: func(r *admissionv1.AdmissionRequest) {
			r.Object.Raw = []byte(`{"apiVersion":"apps/v1","kind":"StatefulSet"}`)
		}},
		{name: "matching raw and decoded", mutate: func(r *admissionv1.AdmissionRequest) { r.Object.Object = decoded("apps/v1", "Deployment") }},
		{name: "conflicting raw and decoded", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { r.Object.Object = decoded("batch/v1", "Job") }},
		{name: "null raw and present decoded", deny: true, mutate: func(r *admissionv1.AdmissionRequest) {
			r.Object.Raw = []byte("null")
			r.Object.Object = decoded("apps/v1", "Deployment")
		}},
		{name: "raw with typed nil decoded", deny: true, mutate: func(r *admissionv1.AdmissionRequest) { var object *unstructured.Unstructured; r.Object.Object = object }},
		{name: "typed nil object", deny: true, mutate: func(r *admissionv1.AdmissionRequest) {
			var object *unstructured.Unstructured
			r.Object = runtime.RawExtension{Object: object}
		}},
		{name: "decoded metadata only", mutate: func(r *admissionv1.AdmissionRequest) {
			r.Object = runtime.RawExtension{Object: &metadataOnlyObject{TypeMeta: metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"}, NonJSON: make(chan struct{})}}
		}},
		{name: "unknown raw object", mutate: func(r *admissionv1.AdmissionRequest) {
			r.Object = runtime.RawExtension{Object: &runtime.Unknown{Raw: bytes.Clone(r.Object.Raw)}}
		}},
		{name: "unknown raw group mismatch", deny: true, mutate: func(r *admissionv1.AdmissionRequest) {
			u := &runtime.Unknown{Raw: []byte(`{"apiVersion":"batch/v1","kind":"Job"}`)}
			u.APIVersion, u.Kind = "apps/v1", "Deployment"
			r.Object = runtime.RawExtension{Object: u}
		}},
		{name: "unknown TypeMeta mismatch", deny: true, mutate: func(r *admissionv1.AdmissionRequest) {
			u := &runtime.Unknown{Raw: bytes.Clone(r.Object.Raw)}
			u.APIVersion, u.Kind = "v1", "Secret"
			r.Object = runtime.RawExtension{Object: u}
		}},
		{name: "unknown content type", deny: true, mutate: func(r *admissionv1.AdmissionRequest) {
			r.Object = runtime.RawExtension{Object: &runtime.Unknown{Raw: bytes.Clone(r.Object.Raw), ContentType: "application/yaml"}}
		}},
		{name: "original equivalent version differs", mutate: func(r *admissionv1.AdmissionRequest) {
			r.RequestKind = &metav1.GroupVersionKind{Group: "extensions", Version: "v1beta1", Kind: "Deployment"}
			r.RequestResource = &metav1.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "deployments"}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := admissionReview(testGroup).Request
			if tc.mutate != nil {
				tc.mutate(req)
			}
			err := v.Verify(context.Background(), req, token)
			if tc.deny {
				if err == nil || err.Error() != "webhook token verification failed" {
					t.Fatalf("Verify = %v, want generic denial", err)
				}
			} else if err != nil {
				t.Fatalf("matching metadata denied: %v", err)
			}
		})
	}
}

func TestVerifierRejectsMalformedUnknownMetadata(t *testing.T) {
	ts := newOIDCTestServer(t)
	v := ts.verifier(t)
	token := ts.sign(t, ts.baseClaims())
	for _, tc := range []struct {
		apiVersion string
		operation  admissionv1.Operation
		withRaw    bool
	}{
		{apiVersion: "bad/api/version", operation: admissionv1.Create, withRaw: true},
		{apiVersion: "/", operation: admissionv1.Create, withRaw: true},
		{apiVersion: "bad/api/version", operation: admissionv1.Connect},
	} {
		t.Run(tc.apiVersion+"/"+string(tc.operation), func(t *testing.T) {
			req := admissionReview(testGroup).Request
			object := &runtime.Unknown{}
			object.APIVersion = tc.apiVersion
			if tc.withRaw {
				object.Raw = bytes.Clone(req.Object.Raw)
			}
			req.Operation = tc.operation
			req.Object = runtime.RawExtension{Object: object}
			if err := v.Verify(context.Background(), req, token); err == nil || err.Error() != "webhook token verification failed" {
				t.Fatalf("malformed Unknown TypeMeta: verification = %v, want generic denial", err)
			}
		})
	}
}
