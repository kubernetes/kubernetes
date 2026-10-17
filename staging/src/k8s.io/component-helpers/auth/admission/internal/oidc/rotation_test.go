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

package oidc_test

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	jose "github.com/go-jose/go-jose/v4"

	"k8s.io/component-helpers/auth/admission/internal/oidc"
)

// rotatingOIDCServer serves replaceable signing keys so tests can check
// verification against a bounded JWKS snapshot.
type rotatingOIDCServer struct {
	server *httptest.Server
	issuer string

	mu     sync.Mutex
	priv   *rsa.PrivateKey
	signer jose.Signer
	kid    string
}

func newRotatingOIDCServer(t *testing.T) *rotatingOIDCServer {
	t.Helper()
	ts := &rotatingOIDCServer{}
	ts.rotate(t) // seed the first key

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
		ts.mu.Lock()
		jwks := jose.JSONWebKeySet{Keys: []jose.JSONWebKey{{
			Key:       ts.priv.Public(),
			KeyID:     ts.kid,
			Algorithm: string(jose.RS256),
			Use:       "sig",
		}}}
		ts.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(jwks)
	})

	ts.server = httptest.NewTLSServer(mux)
	ts.issuer = ts.server.URL
	t.Cleanup(ts.server.Close)
	return ts
}

// rotate changes the signing key served at /keys. The verifier cannot use it
// until its JWKS snapshot refreshes.
func (ts *rotatingOIDCServer) rotate(t *testing.T) {
	t.Helper()
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generating RSA key: %v", err)
	}
	// A distinct key ID keeps an old JWKS snapshot from matching the new key.
	kid := "rotating-key-" + time.Now().Format("150405.000000000")
	signer, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.RS256, Key: jose.JSONWebKey{Key: priv, KeyID: kid}},
		(&jose.SignerOptions{}).WithType("JWT"),
	)
	if err != nil {
		t.Fatalf("creating signer: %v", err)
	}
	ts.mu.Lock()
	ts.priv, ts.signer, ts.kid = priv, signer, kid
	ts.mu.Unlock()
}

func (ts *rotatingOIDCServer) client() *http.Client { return ts.server.Client() }

// sign uses the server's current signing key.
func (ts *rotatingOIDCServer) sign(t *testing.T, claims map[string]any) string {
	t.Helper()
	ts.mu.Lock()
	signer := ts.signer
	ts.mu.Unlock()
	return signWith(t, signer, claims)
}

// baseClaims returns a valid KEP-6060 token claim set for this server.
func (ts *rotatingOIDCServer) baseClaims() map[string]any {
	now := time.Now()
	return map[string]any{
		"iss": ts.issuer,
		"sub": testSubject,
		"aud": []string{testAudience},
		"exp": now.Add(time.Hour).Unix(),
		"nbf": now.Add(-time.Minute).Unix(),
		"iat": now.Unix(),
		"kubernetes.io": map[string]any{
			"validatingwebhookconfiguration": map[string]any{"name": testWebhookNm, "uid": testWebhookID},
			"attestations":                   map[string]any{admissionReviewAPIGroupsClaimKey: []string{testAPIGroup}},
		},
	}
}

// TestRemoteVerifier_KeyRotation verifies that an unknown signing key does not
// trigger a per-token JWKS fetch. The new key becomes usable at the next bounded
// refresh; a separate test advances the refresh clock without sleeping.
func TestRemoteVerifier_KeyRotation(t *testing.T) {
	ts := newRotatingOIDCServer(t)

	v, err := oidc.NewRemoteVerifier(context.Background(), ts.issuer, testAudience, ts.client())
	if err != nil {
		t.Fatalf("NewRemoteVerifier: %v", err)
	}

	if err := v.Verify(context.Background(), ts.sign(t, ts.baseClaims()), testAPIGroup); err != nil {
		t.Fatalf("pre-rotation token should verify: %v", err)
	}

	ts.rotate(t)
	if err := v.Verify(context.Background(), ts.sign(t, ts.baseClaims()), testAPIGroup); err == nil {
		t.Fatal("new key unexpectedly verified before the bounded JWKS refresh")
	}
}
