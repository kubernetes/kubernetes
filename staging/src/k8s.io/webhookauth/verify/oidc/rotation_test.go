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

	jose "gopkg.in/go-jose/go-jose.v2"

	"k8s.io/webhookauth/verify/oidc"
)

// rotatingOIDCServer is a TLS OIDC discovery + JWKS endpoint whose signing key
// can be rotated at runtime. It lets the test prove the verifier picks up a
// rotated key from the remote JWKS without reconstruction — go-oidc's remote key
// set refetches the JWKS when it encounters a key id it has not cached.
//
// It is deliberately distinct from the fixed-key oidcTestServer in
// oidc_test.go so neither helper has to grow mutable state it does not
// need.
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

// rotate generates a fresh RSA key with a NEW key id and makes it the server's
// current signing key. The new public key is what /keys will serve from now on.
func (ts *rotatingOIDCServer) rotate(t *testing.T) {
	t.Helper()
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generating RSA key: %v", err)
	}
	// A unique kid per rotation forces go-oidc to treat the new key as unknown
	// and refetch the JWKS, rather than relying on a cached key.
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

// sign mints a token with the server's CURRENT signing key.
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
			"validatingWebhookConfiguration": map[string]any{"name": testWebhookNm, "uid": testWebhookID},
			"attestationClaims":              map[string]any{allowedAPIGroupClaimKey: []string{testAPIGroup}},
		},
	}
}

// TestRemoteVerifier_KeyRotation proves the verifier tracks JWKS key rotation:
// after the issuer rotates its signing key (new key id, published in the same
// JWKS), a token signed by the NEW key still verifies through the SAME verifier
// instance. go-oidc refetches the rotated JWKS on the unseen key id — no verifier
// reconstruction is required, matching how an in-cluster webhook survives the
// apiserver rotating its service-account signing keys.
func TestRemoteVerifier_KeyRotation(t *testing.T) {
	ts := newRotatingOIDCServer(t)

	v, err := oidc.NewRemoteVerifier(context.Background(), ts.issuer, testAudience, oidc.WithHTTPClient(ts.client()))
	if err != nil {
		t.Fatalf("NewRemoteVerifier: %v", err)
	}

	// 1. A token from the original key verifies and primes go-oidc's key cache.
	if _, err := v.Verify(context.Background(), ts.sign(t, ts.baseClaims()), testAPIGroup); err != nil {
		t.Fatalf("pre-rotation token should verify: %v", err)
	}

	// 2. Rotate the issuer's signing key.
	ts.rotate(t)

	// 3. A token signed by the ROTATED key must verify through the same verifier:
	// go-oidc sees an unknown key id and refetches the now-updated JWKS.
	if _, err := v.Verify(context.Background(), ts.sign(t, ts.baseClaims()), testAPIGroup); err != nil {
		t.Fatalf("post-rotation token should verify after JWKS refresh: %v", err)
	}
}
