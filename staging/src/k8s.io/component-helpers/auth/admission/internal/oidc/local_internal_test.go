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
	"encoding/json"
	"encoding/pem"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	jose "github.com/go-jose/go-jose/v4"
	"k8s.io/client-go/rest"
)

func projectedTokenFile(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(path, []byte("local-service-account-token"), 0600); err != nil {
		t.Fatalf("write projected ServiceAccount token: %v", err)
	}
	return path
}

func testServerCA(server *httptest.Server) []byte {
	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: server.Certificate().Raw})
}

func TestInClusterHTTPClientKeepsCAAndProjectedBearer(t *testing.T) {
	authorizations := make(chan string, 1)
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorizations <- r.Header.Get("Authorization")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	ca := testServerCA(server)
	cfg := &rest.Config{
		Host:            server.URL,
		BearerTokenFile: projectedTokenFile(t),
		TLSClientConfig: rest.TLSClientConfig{CAData: ca},
	}
	client, err := serviceAccountTLSClientForConfig(cfg)
	if err != nil {
		t.Fatalf("construct trusted in-cluster discovery client: %v", err)
	}
	resp, err := client.Get(server.URL)
	if err != nil {
		t.Fatalf("trusted local TLS request: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("trusted local TLS status = %d, want 204", resp.StatusCode)
	}
	if got := <-authorizations; got != "Bearer local-service-account-token" {
		t.Fatalf("same-origin discovery bearer = %q, want projected ServiceAccount token", got)
	}

	cfg.TLSClientConfig.CAData = nil
	if _, err := serviceAccountTLSClientForConfig(cfg); err == nil {
		t.Fatal("missing projected CA must fail closed")
	}
	cfg.TLSClientConfig.CAData = ca
	cfg.TLSClientConfig.Insecure = true
	if _, err := serviceAccountTLSClientForConfig(cfg); err == nil {
		t.Fatal("insecure serving-certificate verification must fail closed")
	}
}

func TestLocalVerifierAuthenticatesDiscoveryAndEveryJWKSRefresh(t *testing.T) {
	signer, key, _ := refreshSigningKey(t, jose.RS256, "local-key")
	var discoveryCalls, keyCalls atomic.Int32
	var issuer string
	mux := http.NewServeMux()
	mux.HandleFunc(openIDConfigPath, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer local-service-account-token" {
			http.Error(w, "ServiceAccount bearer required", http.StatusForbidden)
			return
		}
		discoveryCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"issuer": issuer, "jwks_uri": issuer + "/ignored-external-jwks"})
	})
	mux.HandleFunc(localJWKSPath, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer local-service-account-token" {
			http.Error(w, "ServiceAccount bearer required", http.StatusForbidden)
			return
		}
		keyCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(jose.JSONWebKeySet{Keys: []jose.JSONWebKey{key}})
	})
	server := httptest.NewTLSServer(mux)
	defer server.Close()
	issuer = server.URL

	cfg := &rest.Config{
		Host:            server.URL,
		BearerTokenFile: projectedTokenFile(t),
		TLSClientConfig: rest.TLSClientConfig{CAData: testServerCA(server)},
	}
	client, err := serviceAccountTLSClientForConfig(cfg)
	if err != nil {
		t.Fatalf("create projected-bearer client: %v", err)
	}
	clock := newRefreshTestClock()
	auth, err := newLocalAuthenticator(context.Background(), server.URL, "https://webhook.example.svc/validate", client, clock.now)
	if err != nil {
		t.Fatalf("construct local verifier with authenticated discovery: %v", err)
	}
	token := refreshToken(t, signer, server.URL, "cached-local")
	wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, token)
	clock.advance(signingKeyMaxAge)
	wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, token)
	if got := discoveryCalls.Load(); got != 2 {
		t.Fatalf("authenticated discovery calls = %d, want initial and refresh", got)
	}
	if got := keyCalls.Load(); got != 2 {
		t.Fatalf("authenticated local JWKS calls = %d, want initial and refresh", got)
	}
}

func TestLocalVerifierNeverSendsProjectedBearerAcrossRedirect(t *testing.T) {
	signer, key, _ := refreshSigningKey(t, jose.RS256, "local-key")
	var redirected atomic.Int32
	other := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		redirected.Add(1)
		_, _ = w.Write([]byte(`{"keys":[]}`))
	}))
	defer other.Close()

	localRequests := make(chan string, 4)
	var redirectJWKS atomic.Bool
	var issuer string
	mux := http.NewServeMux()
	mux.HandleFunc(openIDConfigPath, func(w http.ResponseWriter, r *http.Request) {
		localRequests <- r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"issuer": issuer, "jwks_uri": other.URL + "/keys"})
	})
	mux.HandleFunc(localJWKSPath, func(w http.ResponseWriter, r *http.Request) {
		localRequests <- r.Header.Get("Authorization")
		if redirectJWKS.Load() {
			http.Redirect(w, r, other.URL+"/keys", http.StatusTemporaryRedirect)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(jose.JSONWebKeySet{Keys: []jose.JSONWebKey{key}})
	})
	server := httptest.NewTLSServer(mux)
	defer server.Close()
	issuer = server.URL

	// Trust both certificates: without the origin pin, the redirected request
	// would reach the other server rather than fail TLS before leaking a bearer.
	ca := append(testServerCA(server), testServerCA(other)...)
	cfg := &rest.Config{
		Host:            server.URL,
		BearerTokenFile: projectedTokenFile(t),
		TLSClientConfig: rest.TLSClientConfig{CAData: ca},
	}
	client, err := serviceAccountTLSClientForConfig(cfg)
	if err != nil {
		t.Fatalf("create projected-bearer client: %v", err)
	}
	clock := newRefreshTestClock()
	auth, err := newLocalAuthenticator(context.Background(), server.URL, "https://webhook.example.svc/validate", client, clock.now)
	if err != nil {
		t.Fatalf("create local verifier: %v", err)
	}
	token := refreshToken(t, signer, server.URL, "cached-local")
	wantRefreshTokenAccepted(t, auth.refreshingAuthenticator, token)
	redirectJWKS.Store(true)
	clock.advance(signingKeyMaxAge)
	_, err = auth.AuthenticateToken(context.Background(), token)
	if err == nil || !strings.Contains(err.Error(), "redirect leaves the trusted origin") {
		t.Fatalf("cross-origin JWKS refresh should fail before sending bearer: %v", err)
	}
	if got := redirected.Load(); got != 0 {
		t.Fatalf("projected bearer reached off-origin server %d times", got)
	}
	for range 4 {
		select {
		case got := <-localRequests:
			if got != "Bearer local-service-account-token" {
				t.Fatalf("same-origin discovery/JWKS bearer = %q, want projected token", got)
			}
		default:
			t.Fatal("did not request both fixed local discovery and JWKS URLs on initial load and refresh")
		}
	}
}
