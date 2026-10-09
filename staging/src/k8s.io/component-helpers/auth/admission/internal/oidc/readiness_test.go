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
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	jose "github.com/go-jose/go-jose/v4"

	"k8s.io/component-helpers/auth/admission/internal/oidc"
)

func TestRemoteVerifier_SigningKeyReadiness(t *testing.T) {
	privateKey, signer := genSigner(t)
	keys := jwksBytes(t, &privateKey.PublicKey)
	var available atomic.Bool
	var empty atomic.Bool
	var issuer string
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer": issuer, "jwks_uri": issuer + "/keys",
			"id_token_signing_alg_values_supported": []string{"RS256"},
		})
	})
	mux.HandleFunc("/keys", func(w http.ResponseWriter, _ *http.Request) {
		if !available.Load() {
			http.Error(w, "signing keys unavailable", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if empty.Load() {
			_, _ = w.Write([]byte(`{"keys":[]}`))
			return
		}
		_, _ = w.Write(keys)
	})
	server := httptest.NewTLSServer(mux)
	defer server.Close()
	issuer = server.URL

	if _, err := oidc.NewRemoteVerifier(context.Background(), issuer, testAudience, server.Client()); err == nil {
		t.Fatal("discovery without an available JWKS must not produce a ready verifier")
	}
	available.Store(true)
	empty.Store(true)
	if _, err := oidc.NewRemoteVerifier(context.Background(), issuer, testAudience, server.Client()); err == nil {
		t.Fatal("an empty JWKS must not report ready")
	}
	empty.Store(false)
	verifier, err := oidc.NewRemoteVerifier(context.Background(), issuer, testAudience, server.Client())
	if err != nil {
		t.Fatalf("build verifier after JWKS recovery: %v", err)
	}
	if err := verifier.HealthCheck(); err != nil {
		t.Fatalf("healthy signing keys reported not ready: %v", err)
	}
	if err := verifier.Verify(context.Background(), signWith(t, signer, localClaims(issuer)), testAPIGroup); err != nil {
		t.Fatalf("verify token after signing-key recovery: %v", err)
	}

	available.Store(false)
	if err := verifier.HealthCheck(); err == nil {
		t.Fatal("HealthCheck reported ready when the JWKS endpoint became unavailable")
	}
	available.Store(true)
	if err := verifier.HealthCheck(); err != nil {
		t.Fatalf("HealthCheck after JWKS recovery: %v", err)
	}
}

func TestRemoteVerifier_RejectsMalformedSigningKeys(t *testing.T) {
	var issuer string
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer": issuer, "jwks_uri": issuer + "/keys",
			"id_token_signing_alg_values_supported": []string{"RS256"},
		})
	})
	mux.HandleFunc("/keys", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"keys":[{"kty":"RSA","n":"!","e":"!"}]}`))
	})
	server := httptest.NewTLSServer(mux)
	defer server.Close()
	issuer = server.URL

	if _, err := oidc.NewRemoteVerifier(context.Background(), issuer, testAudience, server.Client()); err == nil {
		t.Fatal("malformed public signing keys must not report the verifier ready")
	}
}

func TestRemoteVerifier_MixedSupportedAndUnsupportedKeys(t *testing.T) {
	privateKey, signer := genSigner(t)
	body, err := json.Marshal(map[string]any{"keys": []any{
		jose.JSONWebKey{Key: &privateKey.PublicKey, KeyID: testKeyID, Algorithm: string(jose.RS256), Use: "sig"},
		map[string]any{"kty": "EC", "crv": "secp256k1", "x": "!", "y": "!", "alg": "ES256K"},
		map[string]any{"kty": "OKP", "crv": "X448", "x": "AQ", "use": "enc"},
	}})
	if err != nil {
		t.Fatalf("marshal mixed signing keys: %v", err)
	}
	var issuer string
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer": issuer, "jwks_uri": issuer + "/keys",
			"id_token_signing_alg_values_supported": []string{"RS256"},
		})
	})
	mux.HandleFunc("/keys", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	})
	server := httptest.NewTLSServer(mux)
	defer server.Close()
	issuer = server.URL

	verifier, err := oidc.NewRemoteVerifier(context.Background(), issuer, testAudience, server.Client())
	if err != nil {
		t.Fatalf("usable RSA key should survive unrelated unsupported JWKs: %v", err)
	}
	if err := verifier.Verify(context.Background(), signWith(t, signer, localClaims(issuer)), testAPIGroup); err != nil {
		t.Fatalf("verify with usable RSA signing key: %v", err)
	}
}

func TestLocalVerifierRejectsUnsupportedOnlyJWKS(t *testing.T) {
	publicKey, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate EdDSA key: %v", err)
	}
	keys := jose.JSONWebKeySet{Keys: []jose.JSONWebKey{{
		Key: publicKey, KeyID: "ed25519-test", Algorithm: string(jose.EdDSA), Use: "sig",
	}}}

	var issuer string
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer": issuer, "jwks_uri": issuer + "/openid/v1/jwks",
			"id_token_signing_alg_values_supported": []string{string(jose.EdDSA)},
		})
	})
	mux.HandleFunc("/openid/v1/jwks", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(keys)
	})
	server := httptest.NewTLSServer(mux)
	defer server.Close()
	issuer = server.URL

	if _, err := oidc.NewLocalKeySetVerifier(context.Background(), issuer, testAudience, server.Client()); err == nil {
		t.Fatal("local readiness accepted only EdDSA keys despite its ES/RS signing-algorithm policy")
	}
	if _, err := oidc.NewRemoteVerifier(context.Background(), issuer, testAudience, server.Client()); err != nil {
		t.Fatalf("remote OIDC issuer may legitimately advertise EdDSA: %v", err)
	}
}

func TestLocalVerifierRejectsCrossOriginJWKSRedirect(t *testing.T) {
	privateKey, _ := genSigner(t)
	keys := jwksBytes(t, &privateKey.PublicKey)
	var redirected atomic.Bool
	other := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		redirected.Store(true)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(keys)
	}))
	defer other.Close()

	var issuer string
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"issuer": issuer, "jwks_uri": issuer + "/openid/v1/jwks"})
	})
	mux.HandleFunc("/openid/v1/jwks", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, other.URL+"/keys", http.StatusTemporaryRedirect)
	})
	server := httptest.NewTLSServer(mux)
	defer server.Close()
	issuer = server.URL

	_, err := oidc.NewLocalKeySetVerifier(context.Background(), issuer, testAudience, server.Client())
	if err == nil || !strings.Contains(err.Error(), "redirect leaves the trusted origin") {
		t.Fatalf("cross-origin local JWKS redirect: got %v, want a trust-boundary error", err)
	}
	if redirected.Load() {
		t.Fatal("local JWKS request reached the untrusted redirected host")
	}
}
