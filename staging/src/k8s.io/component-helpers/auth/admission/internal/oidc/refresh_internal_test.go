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

package oidc

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	jose "github.com/go-jose/go-jose/v4"
)

type refreshTestClock struct{ nanoseconds atomic.Int64 }

func newRefreshTestClock() *refreshTestClock {
	clock := &refreshTestClock{}
	clock.nanoseconds.Store(time.Now().UnixNano())
	return clock
}

func (c *refreshTestClock) now() time.Time { return time.Unix(0, c.nanoseconds.Load()) }
func (c *refreshTestClock) advance(d time.Duration) {
	c.nanoseconds.Add(int64(d))
}

type refreshTestIssuer struct {
	server *httptest.Server
	mu     sync.Mutex
	issuer string

	discoveredIssuer string
	discoveryBody    string
	algorithms       []string
	keys             []jose.JSONWebKey
	jwksStatus       int
	jwksBody         string
	jwksRedirect     string
	blockJWKS        <-chan struct{}
	enteredJWKS      chan<- struct{}

	discoveryRequests int
	keyRequests       int
}

func newRefreshTestIssuer(t *testing.T, keys []jose.JSONWebKey, algorithms []string) *refreshTestIssuer {
	t.Helper()
	s := &refreshTestIssuer{keys: keys, algorithms: algorithms}
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, _ *http.Request) {
		s.mu.Lock()
		s.discoveryRequests++
		issuer := s.issuer
		if s.discoveredIssuer != "" {
			issuer = s.discoveredIssuer
		}
		algorithms := append([]string(nil), s.algorithms...)
		body := s.discoveryBody
		s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if body != "" {
			_, _ = w.Write([]byte(body))
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer": issuer, "jwks_uri": s.issuer + "/keys",
			"id_token_signing_alg_values_supported": algorithms,
		})
	})
	serveKeys := func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		s.keyRequests++
		keys := append([]jose.JSONWebKey(nil), s.keys...)
		status, body, redirect := s.jwksStatus, s.jwksBody, s.jwksRedirect
		block, entered := s.blockJWKS, s.enteredJWKS
		s.mu.Unlock()
		if entered != nil {
			select {
			case entered <- struct{}{}:
			default:
			}
		}
		if block != nil {
			select {
			case <-block:
			case <-r.Context().Done():
				return
			}
		}
		if redirect != "" {
			http.Redirect(w, r, redirect, http.StatusTemporaryRedirect)
			return
		}
		if status != 0 {
			http.Error(w, "signing keys unavailable", status)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if body != "" {
			_, _ = w.Write([]byte(body))
			return
		}
		_ = json.NewEncoder(w).Encode(jose.JSONWebKeySet{Keys: keys})
	}
	mux.HandleFunc("/keys", serveKeys)
	mux.HandleFunc("/openid/v1/jwks", serveKeys)
	s.server = httptest.NewTLSServer(mux)
	s.issuer = s.server.URL
	t.Cleanup(s.server.Close)
	return s
}

func (s *refreshTestIssuer) requests() (discovery, keys int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.discoveryRequests, s.keyRequests
}

func refreshSigningKey(t *testing.T, alg jose.SignatureAlgorithm, kid string) (jose.Signer, jose.JSONWebKey, any) {
	t.Helper()
	var privateKey, publicKey any
	switch alg {
	case jose.RS256:
		private, err := rsa.GenerateKey(rand.Reader, 2048)
		if err != nil {
			t.Fatalf("generate RSA signing key: %v", err)
		}
		privateKey, publicKey = private, &private.PublicKey
	case jose.ES256:
		private, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		if err != nil {
			t.Fatalf("generate ECDSA signing key: %v", err)
		}
		privateKey, publicKey = private, &private.PublicKey
	default:
		t.Fatalf("unsupported test signing algorithm %q", alg)
	}
	signer, err := jose.NewSigner(jose.SigningKey{
		Algorithm: alg, Key: jose.JSONWebKey{Key: privateKey, KeyID: kid},
	}, (&jose.SignerOptions{}).WithType("JWT"))
	if err != nil {
		t.Fatalf("create signer: %v", err)
	}
	return signer, jose.JSONWebKey{Key: publicKey, KeyID: kid, Algorithm: string(alg), Use: "sig"}, privateKey
}

func refreshToken(t *testing.T, signer jose.Signer, issuer, nonce string) string {
	t.Helper()
	now := time.Now()
	claims := map[string]any{
		"iss": issuer, "sub": "system:serviceaccount:kube-system:issuer",
		"aud": []string{"https://webhook.example.svc/validate"},
		"exp": now.Add(time.Hour).Unix(), "nbf": now.Add(-time.Minute).Unix(), "iat": now.Unix(),
		"jti": nonce,
		"kubernetes.io": map[string]any{
			"validatingwebhookconfiguration": map[string]any{"name": "test-webhook", "uid": "11111111-2222-3333-4444-555555555555"},
			"attestations":                   map[string]any{admissionReviewAPIGroupsClaimKey: []string{"apps"}},
		},
	}
	payload, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshal token claims: %v", err)
	}
	signed, err := signer.Sign(payload)
	if err != nil {
		t.Fatalf("sign token: %v", err)
	}
	compact, err := signed.CompactSerialize()
	if err != nil {
		t.Fatalf("serialize token: %v", err)
	}
	return compact
}

func wantRefreshTokenAccepted(t *testing.T, auth *refreshingAuthenticator, token string) {
	t.Helper()
	groups, err := auth.AuthenticateToken(context.Background(), token)
	if err != nil || len(groups) != 1 || groups[0] != "apps" {
		t.Fatalf("valid token: groups=%v, err=%v", groups, err)
	}
}

func wantRefreshTokenRejected(t *testing.T, auth *refreshingAuthenticator, token string) {
	t.Helper()
	if groups, err := auth.AuthenticateToken(context.Background(), token); err == nil {
		t.Fatalf("invalid token accepted with groups %v", groups)
	}
}

func requirePositiveCacheEntry(t *testing.T, snapshot *verifierSnapshot, token string) {
	t.Helper()
	if snapshot == nil {
		t.Fatal("signing-key snapshot is unavailable")
	}
	if _, found := snapshot.tokenCache.Get(sha256.Sum256([]byte(token))); !found {
		t.Fatal("recently verified token is not in its positive cache")
	}
}

func TestRemoteSigningKeysExpireAndRevoke(t *testing.T) {
	oldSigner, oldKey, _ := refreshSigningKey(t, jose.RS256, "old-key")
	remainingSigner, remainingKey, _ := refreshSigningKey(t, jose.RS256, "remaining-key")
	server := newRefreshTestIssuer(t, []jose.JSONWebKey{oldKey, remainingKey}, []string{"RS256"})
	clock := newRefreshTestClock()
	auth, err := newRemoteAuthenticator(context.Background(), server.issuer, "https://webhook.example.svc/validate", server.server.Client(), clock.now)
	if err != nil {
		t.Fatalf("create verifier: %v", err)
	}
	cachedOld := refreshToken(t, oldSigner, server.issuer, "cached-old")
	wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, cachedOld)
	oldSnapshot := auth.current.Load()
	requirePositiveCacheEntry(t, oldSnapshot, cachedOld)

	server.mu.Lock()
	server.keys = []jose.JSONWebKey{remainingKey}
	server.mu.Unlock()
	// A JWKS change cannot revoke a key before the bounded snapshot expires.
	wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, cachedOld)
	clock.advance(signingKeyMaxAge)
	// The fake clock advances the key lifetime without expiring the real 30s
	// positive cache. Reject the revoked key despite that live cache entry.
	requirePositiveCacheEntry(t, oldSnapshot, cachedOld)
	wantRefreshTokenRejected(t, auth.refreshingAuthenticator, refreshToken(t, oldSigner, server.issuer, "new-old"))
	if refreshed := auth.current.Load(); refreshed == nil || refreshed.tokenCache == oldSnapshot.tokenCache {
		t.Fatal("successful JWKS rotation reused the old positive token cache")
	}
	wantRefreshTokenRejected(t, auth.refreshingAuthenticator, cachedOld)
	wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, refreshToken(t, remainingSigner, server.issuer, "remaining"))
	if discoveries, keys := server.requests(); discoveries != 2 || keys != 2 {
		t.Fatalf("discovery/JWKS requests = %d/%d, want 2/2", discoveries, keys)
	}
}

func TestRemoteSigningAlgorithmsRefreshWithKeys(t *testing.T) {
	rsaSigner, rsaKey, _ := refreshSigningKey(t, jose.RS256, "rsa-key")
	ecSigner, ecKey, _ := refreshSigningKey(t, jose.ES256, "ec-key")
	server := newRefreshTestIssuer(t, []jose.JSONWebKey{rsaKey}, []string{"RS256"})
	clock := newRefreshTestClock()
	auth, err := newRemoteAuthenticator(context.Background(), server.issuer, "https://webhook.example.svc/validate", server.server.Client(), clock.now)
	if err != nil {
		t.Fatalf("create verifier: %v", err)
	}
	oldToken := refreshToken(t, rsaSigner, server.issuer, "old-rsa")
	wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, oldToken)

	server.mu.Lock()
	server.algorithms = []string{"ES256"}
	server.keys = []jose.JSONWebKey{ecKey}
	server.mu.Unlock()
	newToken := refreshToken(t, ecSigner, server.issuer, "new-ec")
	wantRefreshTokenRejected(t, auth.refreshingAuthenticator, newToken)
	clock.advance(signingKeyMaxAge)
	wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, newToken)
	wantRefreshTokenRejected(t, auth.refreshingAuthenticator, oldToken)
	if discoveries, keys := server.requests(); discoveries != 2 || keys != 2 {
		t.Fatalf("discovery/JWKS requests = %d/%d, want 2/2", discoveries, keys)
	}
}

func TestRemoteSigningKeysFailClosedAndRecover(t *testing.T) {
	tests := []struct {
		name string
		fail func(*refreshTestIssuer)
	}{
		{
			name: "unavailable JWKS",
			fail: func(s *refreshTestIssuer) { s.jwksStatus = http.StatusServiceUnavailable },
		},
		{
			name: "malformed-only JWKS",
			fail: func(s *refreshTestIssuer) { s.jwksBody = `{"keys":[{"kty":"RSA","n":"!","e":"!"}]}` },
		},
		{
			name: "non-positive RSA modulus",
			fail: func(s *refreshTestIssuer) { s.jwksBody = `{"keys":[{"kty":"RSA","n":"AA","e":"AQAB","alg":"RS256"}]}` },
		},
		{
			name: "issuer changed in discovery",
			fail: func(s *refreshTestIssuer) { s.discoveredIssuer = "https://different-issuer.example" },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			signer, key, _ := refreshSigningKey(t, jose.RS256, "key")
			server := newRefreshTestIssuer(t, []jose.JSONWebKey{key}, []string{"RS256"})
			clock := newRefreshTestClock()
			auth, err := newRemoteAuthenticator(context.Background(), server.issuer, "https://webhook.example.svc/validate", server.server.Client(), clock.now)
			if err != nil {
				t.Fatalf("create verifier: %v", err)
			}
			token := refreshToken(t, signer, server.issuer, "cached")
			wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, token)
			oldSnapshot := auth.current.Load()
			requirePositiveCacheEntry(t, oldSnapshot, token)
			server.mu.Lock()
			tt.fail(server)
			server.mu.Unlock()
			clock.advance(signingKeyMaxAge)
			requirePositiveCacheEntry(t, oldSnapshot, token)
			wantRefreshTokenRejected(t, auth.refreshingAuthenticator, token)
			if auth.current.Load() != nil {
				t.Fatal("failed signing-key refresh left the old cached token available")
			}
			discoveries, keys := server.requests()
			for range 10 {
				wantRefreshTokenRejected(t, auth.refreshingAuthenticator, token)
			}
			if againDiscovery, againKeys := server.requests(); againDiscovery != discoveries || againKeys != keys {
				t.Fatalf("retry storm after failed refresh: discovery/JWKS requests grew from %d/%d to %d/%d", discoveries, keys, againDiscovery, againKeys)
			}
			server.mu.Lock()
			server.jwksStatus, server.jwksBody, server.discoveredIssuer = 0, "", ""
			server.mu.Unlock()
			clock.advance(signingKeyRetryDelay)
			wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, token)
		})
	}
}

func TestHealthCheckInvalidatesTheVerifyingSnapshot(t *testing.T) {
	signer, key, _ := refreshSigningKey(t, jose.RS256, "key")
	server := newRefreshTestIssuer(t, []jose.JSONWebKey{key}, []string{"RS256"})
	clock := newRefreshTestClock()
	auth, err := newRemoteAuthenticator(context.Background(), server.issuer, "https://webhook.example.svc/validate", server.server.Client(), clock.now)
	if err != nil {
		t.Fatalf("create verifier: %v", err)
	}
	token := refreshToken(t, signer, server.issuer, "cached")
	wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, token)
	server.mu.Lock()
	server.jwksStatus = http.StatusServiceUnavailable
	server.mu.Unlock()
	if err := auth.HealthCheck(); err == nil {
		t.Fatal("unavailable JWKS reported healthy")
	}
	wantRefreshTokenRejected(t, auth.refreshingAuthenticator, token)
	server.mu.Lock()
	server.jwksStatus = 0
	server.mu.Unlock()
	if err := auth.HealthCheck(); err != nil {
		t.Fatalf("JWKS recovery reported unhealthy: %v", err)
	}
	wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, token)
}

func TestConcurrentSigningKeyRefreshAndCanceledWaiter(t *testing.T) {
	signer, key, _ := refreshSigningKey(t, jose.RS256, "key")
	server := newRefreshTestIssuer(t, []jose.JSONWebKey{key}, []string{"RS256"})
	clock := newRefreshTestClock()
	auth, err := newRemoteAuthenticator(context.Background(), server.issuer, "https://webhook.example.svc/validate", server.server.Client(), clock.now)
	if err != nil {
		t.Fatalf("create verifier: %v", err)
	}
	token := refreshToken(t, signer, server.issuer, "concurrent")
	clock.advance(signingKeyMaxAge)
	release := make(chan struct{})
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()
	entered := make(chan struct{}, 1)
	server.mu.Lock()
	server.blockJWKS, server.enteredJWKS = release, entered
	server.mu.Unlock()

	const requests = 20
	outcomes := make(chan error, requests)
	go func() {
		_, err := auth.AuthenticateToken(context.Background(), token)
		outcomes <- err
	}()
	select {
	case <-entered:
	case <-time.After(10 * time.Second):
		t.Fatal("refresh did not reach JWKS endpoint")
	}
	refreshStartedAt := clock.now()
	ctx, cancel := context.WithCancel(context.Background())
	canceledWaiter := make(chan error, 1)
	go func() {
		_, err := auth.AuthenticateToken(ctx, token)
		canceledWaiter <- err
	}()
	for range requests - 1 {
		go func() {
			_, err := auth.AuthenticateToken(context.Background(), token)
			outcomes <- err
		}()
	}
	cancel()
	select {
	case err := <-canceledWaiter:
		if err == nil {
			t.Fatal("canceled refresh waiter authenticated a token")
		}
	case <-time.After(time.Second):
		t.Fatal("canceled refresh waiter did not return promptly")
	}
	// Even when the HTTP response is delayed, it must not extend the
	// revocation window by the time spent waiting for the response.
	clock.advance(5 * time.Second)
	close(release)
	for range requests {
		select {
		case err := <-outcomes:
			if err != nil {
				t.Fatalf("concurrent verification after refresh: %v", err)
			}
		case <-time.After(10 * time.Second):
			t.Fatal("concurrent verification hung after refresh")
		}
	}
	if discoveries, keys := server.requests(); discoveries != 2 || keys != 2 {
		t.Fatalf("concurrent requests produced discovery/JWKS requests = %d/%d, want 2/2", discoveries, keys)
	}
	if got, want := auth.current.Load().validUntil, refreshStartedAt.Add(signingKeyMaxAge); !got.Equal(want) {
		t.Fatalf("snapshot deadline = %v, want refresh-start deadline %v", got, want)
	}
}

func TestUnknownKeyIDNeverFetchesPerToken(t *testing.T) {
	_, key, _ := refreshSigningKey(t, jose.RS256, "published-key")
	unknownSigner, _, _ := refreshSigningKey(t, jose.RS256, "unknown-key")
	server := newRefreshTestIssuer(t, []jose.JSONWebKey{key}, []string{"RS256"})
	clock := newRefreshTestClock()
	auth, err := newRemoteAuthenticator(context.Background(), server.issuer, "https://webhook.example.svc/validate", server.server.Client(), clock.now)
	if err != nil {
		t.Fatalf("create verifier: %v", err)
	}
	badToken := refreshToken(t, unknownSigner, server.issuer, "unknown-key")
	for range 20 {
		wantRefreshTokenRejected(t, auth.refreshingAuthenticator, badToken)
	}
	if discoveries, keys := server.requests(); discoveries != 1 || keys != 1 {
		t.Fatalf("unknown kid caused early discovery/JWKS requests = %d/%d, want 1/1", discoveries, keys)
	}
	clock.advance(signingKeyMaxAge)
	for range 20 {
		wantRefreshTokenRejected(t, auth.refreshingAuthenticator, badToken)
	}
	if discoveries, keys := server.requests(); discoveries != 2 || keys != 2 {
		t.Fatalf("unknown kid caused per-token discovery/JWKS requests = %d/%d, want 2/2", discoveries, keys)
	}
}

func TestRemoteSigningKeysRejectTinyRSAModulus(t *testing.T) {
	signer, validKey, _ := refreshSigningKey(t, jose.RS256, "valid")
	server := newRefreshTestIssuer(t, nil, []string{"RS256"})
	tinyKey := map[string]string{
		"kty": "RSA", "kid": "tiny", "alg": "RS256", "use": "sig", "n": "AQ", "e": "Aw",
	}
	tinyOnly, err := json.Marshal(map[string]any{"keys": []any{tinyKey}})
	if err != nil {
		t.Fatalf("marshal tiny RSA JWKS: %v", err)
	}
	server.mu.Lock()
	server.jwksBody = string(tinyOnly)
	server.mu.Unlock()
	if _, err := NewRemoteVerifier(context.Background(), server.issuer, "https://webhook.example.svc/validate", server.server.Client()); err == nil {
		t.Fatal("tiny RSA modulus made an unusable JWKS ready")
	}

	mixed, err := json.Marshal(map[string]any{"keys": []any{tinyKey, validKey}})
	if err != nil {
		t.Fatalf("marshal mixed RSA JWKS: %v", err)
	}
	server.mu.Lock()
	server.jwksBody = string(mixed)
	server.mu.Unlock()
	verifier, err := NewRemoteVerifier(context.Background(), server.issuer, "https://webhook.example.svc/validate", server.server.Client())
	if err != nil {
		t.Fatalf("usable RSA key should survive unrelated tiny key: %v", err)
	}
	token := refreshToken(t, signer, server.issuer, "cached-valid")
	if err := verifier.Verify(context.Background(), token, "apps"); err != nil {
		t.Fatalf("verify using valid key from mixed JWKS: %v", err)
	}
	server.mu.Lock()
	server.jwksBody = string(tinyOnly)
	server.mu.Unlock()
	if err := verifier.HealthCheck(); err == nil {
		t.Fatal("HealthCheck reported ready after JWKS contained only a tiny RSA key")
	}
	if err := verifier.Verify(context.Background(), token, "apps"); err == nil {
		t.Fatal("cached token remained accepted after unusable JWKS invalidated verifier")
	}
}

func TestRemoteSigningKeysReject1024BitRSA(t *testing.T) {
	privateKey, err := rsa.GenerateKey(rand.Reader, 1024)
	if err != nil {
		t.Fatalf("generate 1024-bit RSA key: %v", err)
	}
	server := newRefreshTestIssuer(t, []jose.JSONWebKey{{
		Key: &privateKey.PublicKey, KeyID: "rsa-1024", Algorithm: "RS256", Use: "sig",
	}}, []string{"RS256"})
	if _, err := NewRemoteVerifier(context.Background(), server.issuer, "https://webhook.example.svc/validate", server.server.Client()); err == nil {
		t.Fatal("JWKS with RSA key below RFC 7518's 2048-bit minimum reported ready")
	}
}

func TestSigningKeyIDMustMatchTheVerifyingKey(t *testing.T) {
	_, firstKey, _ := refreshSigningKey(t, jose.RS256, "first")
	_, secondKey, secondPrivate := refreshSigningKey(t, jose.RS256, "second")
	spoofedSigner, err := jose.NewSigner(jose.SigningKey{
		Algorithm: jose.RS256, Key: jose.JSONWebKey{Key: secondPrivate, KeyID: "first"},
	}, (&jose.SignerOptions{}).WithType("JWT"))
	if err != nil {
		t.Fatalf("create signer with mismatched kid: %v", err)
	}
	server := newRefreshTestIssuer(t, []jose.JSONWebKey{firstKey, secondKey}, []string{"RS256"})
	auth, err := newRemoteAuthenticator(context.Background(), server.issuer, "https://webhook.example.svc/validate", server.server.Client(), time.Now)
	if err != nil {
		t.Fatalf("create verifier: %v", err)
	}
	wantRefreshTokenRejected(t, auth.refreshingAuthenticator, refreshToken(t, spoofedSigner, server.issuer, "spoofed-kid"))
}

func TestLocalSigningKeysPinIssuerAndOriginOnRefresh(t *testing.T) {
	signer, key, _ := refreshSigningKey(t, jose.RS256, "local-key")
	server := newRefreshTestIssuer(t, []jose.JSONWebKey{key}, []string{"RS256"})
	clock := newRefreshTestClock()
	auth, err := newLocalAuthenticator(context.Background(), server.issuer, "https://webhook.example.svc/validate", server.server.Client(), clock.now)
	if err != nil {
		t.Fatalf("create local verifier: %v", err)
	}
	token := refreshToken(t, signer, server.issuer, "cached-local")
	wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, token)

	server.mu.Lock()
	server.discoveredIssuer = "https://different-issuer.example"
	server.mu.Unlock()
	clock.advance(signingKeyMaxAge)
	wantRefreshTokenRejected(t, auth.refreshingAuthenticator, token)
	server.mu.Lock()
	server.discoveredIssuer = ""
	server.mu.Unlock()
	if err := auth.HealthCheck(); err != nil {
		t.Fatalf("recover local issuer: %v", err)
	}
	wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, token)

	var redirected atomic.Bool
	other := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		redirected.Store(true)
		_, _ = w.Write([]byte(`{"keys":[]}`))
	}))
	defer other.Close()
	server.mu.Lock()
	server.jwksRedirect = other.URL + "/keys"
	server.mu.Unlock()
	clock.advance(signingKeyMaxAge)
	wantRefreshTokenRejected(t, auth.refreshingAuthenticator, token)
	if redirected.Load() {
		t.Fatal("local JWKS refresh followed a cross-origin redirect")
	}
}

func TestRemoteDiscoveryBodyIsBoundedOnConstructionAndRefresh(t *testing.T) {
	signer, key, _ := refreshSigningKey(t, jose.RS256, "key")
	server := newRefreshTestIssuer(t, []jose.JSONWebKey{key}, []string{"RS256"})
	doc, err := json.Marshal(map[string]any{
		"issuer": server.issuer, "jwks_uri": server.issuer + "/keys",
		"id_token_signing_alg_values_supported": []string{"RS256"},
	})
	if err != nil {
		t.Fatalf("marshal discovery document: %v", err)
	}
	tooLarge := string(doc) + strings.Repeat(" ", maxDiscoveryBytes)
	server.mu.Lock()
	server.discoveryBody = tooLarge
	server.mu.Unlock()
	if _, err := newRemoteAuthenticator(context.Background(), server.issuer, "https://webhook.example.svc/validate", server.server.Client(), time.Now); err == nil {
		t.Fatal("oversized discovery document was accepted at construction")
	}

	server.mu.Lock()
	server.discoveryBody = ""
	server.mu.Unlock()
	clock := newRefreshTestClock()
	auth, err := newRemoteAuthenticator(context.Background(), server.issuer, "https://webhook.example.svc/validate", server.server.Client(), clock.now)
	if err != nil {
		t.Fatalf("create verifier after recovery: %v", err)
	}
	token := refreshToken(t, signer, server.issuer, "cached")
	wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, token)
	server.mu.Lock()
	server.discoveryBody = tooLarge
	server.mu.Unlock()
	clock.advance(signingKeyMaxAge)
	wantRefreshTokenRejected(t, auth.refreshingAuthenticator, token)
}
