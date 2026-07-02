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

package josekeyset

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
)

// remoteHarness is a signing key with a chosen kid plus its public JWK.
type remoteHarness struct {
	signer jose.Signer
	jwk    jose.JSONWebKey
}

func newRemoteHarness(t *testing.T, kid string) *remoteHarness {
	t.Helper()
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	signer, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.RS256, Key: jose.JSONWebKey{Key: priv, KeyID: kid}},
		(&jose.SignerOptions{}).WithType("JWT"),
	)
	if err != nil {
		t.Fatalf("new signer: %v", err)
	}
	return &remoteHarness{
		signer: signer,
		jwk:    jose.JSONWebKey{Key: priv.Public(), KeyID: kid, Algorithm: string(jose.RS256), Use: "sig"},
	}
}

func (h *remoteHarness) mint(t *testing.T, claims map[string]interface{}) string {
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

// oidcServer is a configurable OIDC discovery + JWKS endpoint over TLS.
type oidcServer struct {
	srv *httptest.Server

	mu                  sync.Mutex
	issuer              string // value advertised in the discovery doc; defaults to srv.URL
	jwksURIOverride     string // when set, advertised as jwks_uri instead of srv.URL + "/jwks"
	redirectDiscoveryTo string // when set, the discovery endpoint 302-redirects here
	redirectJWKSTo      string // when set, the JWKS endpoint 302-redirects here
	jwks                jose.JSONWebKeySet
	badDiscovery        bool
	badJWKS             bool
	discoveryHits       int
	jwksHits            int
}

func newOIDCServer(t *testing.T, keys ...jose.JSONWebKey) *oidcServer {
	t.Helper()
	o := &oidcServer{jwks: jose.JSONWebKeySet{Keys: keys}}
	mux := http.NewServeMux()
	mux.HandleFunc(discoveryPath, func(w http.ResponseWriter, req *http.Request) {
		o.mu.Lock()
		defer o.mu.Unlock()
		o.discoveryHits++
		if o.redirectDiscoveryTo != "" {
			http.Redirect(w, req, o.redirectDiscoveryTo, http.StatusFound)
			return
		}
		if o.badDiscovery {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte("{not json"))
			return
		}
		issuer := o.issuer
		if issuer == "" {
			issuer = o.srv.URL
		}
		jwksURI := o.jwksURIOverride
		if jwksURI == "" {
			jwksURI = o.srv.URL + "/jwks"
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{
			"issuer":   issuer,
			"jwks_uri": jwksURI,
		})
	})
	mux.HandleFunc("/jwks", func(w http.ResponseWriter, req *http.Request) {
		o.mu.Lock()
		defer o.mu.Unlock()
		o.jwksHits++
		if o.redirectJWKSTo != "" {
			http.Redirect(w, req, o.redirectJWKSTo, http.StatusFound)
			return
		}
		if o.badJWKS {
			_, _ = w.Write([]byte("{not json"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(o.jwks)
	})
	o.srv = httptest.NewTLSServer(mux)
	t.Cleanup(o.srv.Close)
	return o
}

func (o *oidcServer) setKeys(keys ...jose.JSONWebKey) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.jwks = jose.JSONWebKeySet{Keys: keys}
}

func (o *oidcServer) hits() (discovery, jwks int) {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.discoveryHits, o.jwksHits
}

func tokenClaims() map[string]interface{} {
	return map[string]interface{}{
		"iss": "https://issuer.example.com",
		"sub": "system:serviceaccount:kube-system:webhook-auth",
		"aud": []string{"webhook.example.com"},
		"exp": time.Now().Add(5 * time.Minute).Unix(),
	}
}

func newRemote(t *testing.T, o *oidcServer, opts ...RemoteOption) *RemoteKeySet {
	t.Helper()
	opts = append([]RemoteOption{WithHTTPClient(o.srv.Client())}, opts...)
	rk, err := NewRemoteKeySet(o.srv.URL, opts...)
	if err != nil {
		t.Fatalf("NewRemoteKeySet: %v", err)
	}
	return rk
}

func TestRemoteKeySet_HappyPath(t *testing.T) {
	h := newRemoteHarness(t, "kid-a")
	o := newOIDCServer(t, h.jwk)
	rk := newRemote(t, o)

	tok := h.mint(t, tokenClaims())
	payload, err := rk.VerifySignature(context.Background(), tok)
	if err != nil {
		t.Fatalf("VerifySignature: %v", err)
	}
	var got map[string]interface{}
	if err := json.Unmarshal(payload, &got); err != nil {
		t.Fatalf("payload not JSON: %v", err)
	}
	if got["sub"] != "system:serviceaccount:kube-system:webhook-auth" {
		t.Errorf("unexpected payload sub: %v", got["sub"])
	}
}

func TestRemoteKeySet_Caching(t *testing.T) {
	h := newRemoteHarness(t, "kid-a")
	o := newOIDCServer(t, h.jwk)
	rk := newRemote(t, o)

	tok := h.mint(t, tokenClaims())
	for i := 0; i < 3; i++ {
		if _, err := rk.VerifySignature(context.Background(), tok); err != nil {
			t.Fatalf("verify %d: %v", i, err)
		}
	}
	if d, j := o.hits(); d != 1 || j != 1 {
		t.Errorf("expected 1 discovery + 1 jwks hit, got discovery=%d jwks=%d", d, j)
	}
}

func TestRemoteKeySet_KeyRotation(t *testing.T) {
	oldKey := newRemoteHarness(t, "kid-old")
	o := newOIDCServer(t, oldKey.jwk)
	// Tiny min-refresh so the rotation refetch is not rate-limited.
	rk := newRemote(t, o, WithMinRefreshInterval(time.Nanosecond))

	// Prime the cache with the old key.
	if _, err := rk.VerifySignature(context.Background(), oldKey.mint(t, tokenClaims())); err != nil {
		t.Fatalf("prime verify: %v", err)
	}

	// Issuer rotates to a new signing key.
	newKey := newRemoteHarness(t, "kid-new")
	o.setKeys(newKey.jwk)

	tok := newKey.mint(t, tokenClaims())
	if _, err := rk.VerifySignature(context.Background(), tok); err != nil {
		t.Fatalf("verify after rotation: %v", err)
	}
	if _, j := o.hits(); j < 2 {
		t.Errorf("expected a JWKS refetch on rotation, jwks hits=%d", j)
	}
}

func TestRemoteKeySet_PlainHTTP(t *testing.T) {
	h := newRemoteHarness(t, "kid-a")

	// Plain-http server (no TLS).
	o := &oidcServer{jwks: jose.JSONWebKeySet{Keys: []jose.JSONWebKey{h.jwk}}}
	mux := http.NewServeMux()
	mux.HandleFunc(discoveryPath, func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]string{
			"issuer":   o.srv.URL,
			"jwks_uri": o.srv.URL + "/jwks",
		})
	})
	mux.HandleFunc("/jwks", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(o.jwks)
	})
	o.srv = httptest.NewServer(mux)
	t.Cleanup(o.srv.Close)

	// HTTPS required by default: constructor must reject an http issuer.
	if _, err := NewRemoteKeySet(o.srv.URL, WithHTTPClient(o.srv.Client())); err == nil {
		t.Fatal("expected constructor to reject plain-http issuer, got nil")
	}

	// With the test-only opt-in, plain http is accepted and verification works.
	rk, err := NewRemoteKeySet(o.srv.URL, WithHTTPClient(o.srv.Client()), WithInsecureAllowHTTP())
	if err != nil {
		t.Fatalf("NewRemoteKeySet with WithInsecureAllowHTTP: %v", err)
	}
	if _, err := rk.VerifySignature(context.Background(), h.mint(t, tokenClaims())); err != nil {
		t.Fatalf("verify over http with opt-in: %v", err)
	}
}

// TestRemoteKeySet_Rejected groups the failure modes that share one shape:
// stand up a discovery+JWKS server in a rejecting configuration, then assert a
// valid token still fails to verify. Stateful cases (caching, rotation,
// rate-limiting) are kept as focused tests below because they assert hit counts.
func TestRemoteKeySet_Rejected(t *testing.T) {
	tests := []struct {
		name string
		// configure mutates the server into its rejecting state before verifying.
		configure func(o *oidcServer)
	}{
		{
			name:      "discovery advertises a foreign issuer -> rejected",
			configure: func(o *oidcServer) { o.issuer = "https://attacker.example.com" },
		},
		{
			name:      "malformed discovery document -> rejected",
			configure: func(o *oidcServer) { o.badDiscovery = true },
		},
		{
			name:      "malformed JWKS document -> rejected",
			configure: func(o *oidcServer) { o.badJWKS = true },
		},
		{
			name:      "jwks_uri points to a foreign host -> rejected (SSRF guard)",
			configure: func(o *oidcServer) { o.jwksURIOverride = "https://attacker.example.com/jwks" },
		},
		{
			name:      "discovery redirects to a foreign host -> rejected",
			configure: func(o *oidcServer) { o.redirectDiscoveryTo = "https://attacker.example.com" + discoveryPath },
		},
		{
			name:      "JWKS fetch redirects to a foreign host -> rejected",
			configure: func(o *oidcServer) { o.redirectJWKSTo = "https://attacker.example.com/jwks" },
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h := newRemoteHarness(t, "kid-a")
			o := newOIDCServer(t, h.jwk)
			tc.configure(o)

			rk := newRemote(t, o)
			if _, err := rk.VerifySignature(context.Background(), h.mint(t, tokenClaims())); err == nil {
				t.Fatal("expected verification failure, got nil")
			}
		})
	}
}

func TestRemoteKeySet_MinRefreshInterval(t *testing.T) {
	h := newRemoteHarness(t, "kid-a")
	o := newOIDCServer(t, h.jwk)
	// Large interval: a burst of bad tokens must not stampede the JWKS endpoint.
	rk := newRemote(t, o, WithMinRefreshInterval(time.Hour))

	// A token signed by a foreign key never in the published JWKS.
	foreign := newRemoteHarness(t, "kid-foreign")
	badTok := foreign.mint(t, tokenClaims())

	for i := 0; i < 5; i++ {
		if _, err := rk.VerifySignature(context.Background(), badTok); err == nil {
			t.Fatalf("expected verification failure on iteration %d", i)
		}
	}
	// Exactly one JWKS fetch (the initial load); the rate-limit blocks refetches.
	if _, j := o.hits(); j != 1 {
		t.Errorf("expected exactly 1 JWKS fetch under rate-limit, got %d", j)
	}
}
