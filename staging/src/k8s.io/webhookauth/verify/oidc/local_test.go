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
	"testing"
	"time"

	jose "gopkg.in/go-jose/go-jose.v2"

	"k8s.io/webhookauth/verify/oidc"
)

// genSigner returns a fresh RSA key and a matching RS256 JWT signer.
func genSigner(t *testing.T) (*rsa.PrivateKey, jose.Signer) {
	t.Helper()
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generating RSA key: %v", err)
	}
	signer, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.RS256, Key: jose.JSONWebKey{Key: priv, KeyID: testKeyID}},
		(&jose.SignerOptions{}).WithType("JWT"),
	)
	if err != nil {
		t.Fatalf("creating signer: %v", err)
	}
	return priv, signer
}

// jwksBytes marshals pub as a single-key JWKS, as the apiserver's JWKS endpoint
// would serve it.
func jwksBytes(t *testing.T, pub *rsa.PublicKey) []byte {
	t.Helper()
	set := jose.JSONWebKeySet{Keys: []jose.JSONWebKey{{
		Key:       pub,
		KeyID:     testKeyID,
		Algorithm: string(jose.RS256),
		Use:       "sig",
	}}}
	b, err := json.Marshal(set)
	if err != nil {
		t.Fatalf("marshaling JWKS: %v", err)
	}
	return b
}

// localClaims is a valid KEP-6060 token claim set for the given issuer, with the
// audience the tests bind and the wildcard attestation group.
func localClaims(issuer string) map[string]any {
	now := time.Now()
	return map[string]any{
		"iss": issuer,
		"sub": testSubject,
		"aud": []string{testAudience},
		"exp": now.Add(time.Hour).Unix(),
		"nbf": now.Add(-time.Minute).Unix(),
		"iat": now.Unix(),
		"kubernetes.io": map[string]any{
			"attestations": map[string]any{
				admissionReviewAPIGroupsClaimKey: []string{testAPIGroup},
			},
		},
	}
}

// discoveryDoc is the subset of the OIDC discovery document the tests serve.
type discoveryDoc struct {
	Issuer  string `json:"issuer"`
	JWKSURI string `json:"jwks_uri"`
}

// localAPIServer is a TLS test double for the in-cluster apiserver: it serves an
// OIDC discovery document at /.well-known/openid-configuration and a JWKS at the
// local /openid/v1/jwks endpoint. Individual handlers are swappable so each test
// can exercise a corner case.
type localAPIServer struct {
	*httptest.Server
	signer jose.Signer

	// mutated by tests before requests arrive (handlers read them lazily).
	discovery    discoveryDoc
	discoveryRaw string // if non-empty, served verbatim (for malformed-body tests)
	wellKnown404 bool
}

// newLocalAPIServer stands up the double. By default the discovery issuer is the
// server URL and its jwks_uri is a DECOY endpoint that serves a WRONG key, so any
// test that verifies a real token proves keys came from the local /openid/v1/jwks
// path and not from the advertised jwks_uri.
func newLocalAPIServer(t *testing.T) *localAPIServer {
	t.Helper()
	priv, signer := genSigner(t)
	decoyPriv, _ := genSigner(t)

	s := &localAPIServer{signer: signer}

	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, r *http.Request) {
		if s.wellKnown404 {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if s.discoveryRaw != "" {
			_, _ = w.Write([]byte(s.discoveryRaw))
			return
		}
		_ = json.NewEncoder(w).Encode(s.discovery)
	})
	// The authoritative local key set — the real signing key.
	mux.HandleFunc("/openid/v1/jwks", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(jwksBytes(t, &priv.PublicKey))
	})
	// The decoy the discovery document points at — a DIFFERENT key. If the code
	// ever followed jwks_uri, real-token verification would fail.
	mux.HandleFunc("/keys-decoy", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(jwksBytes(t, &decoyPriv.PublicKey))
	})

	s.Server = httptest.NewTLSServer(mux)
	t.Cleanup(s.Server.Close)
	s.discovery = discoveryDoc{Issuer: s.Server.URL, JWKSURI: s.Server.URL + "/keys-decoy"}
	return s
}

// TestNewLocalKeySetVerifier_HappyPath verifies the whole in-cluster construction
// path: read the issuer from the local discovery document, fetch keys from the
// local JWKS endpoint, and — once the audience is bound — verify a real token.
func TestNewLocalKeySetVerifier_HappyPath(t *testing.T) {
	s := newLocalAPIServer(t)

	v, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, oidc.WithHTTPClient(s.Client()))
	if err != nil {
		t.Fatalf("NewLocalKeySetVerifier: %v", err)
	}
	if err := v.BindAudience(testAudience); err != nil {
		t.Fatalf("BindAudience: %v", err)
	}

	token := signWith(t, s.signer, localClaims(s.URL))
	if err := v.Verify(context.Background(), token, testAPIGroup); err != nil {
		t.Fatalf("expected verification to succeed, got %v", err)
	}
}

// TestNewLocalKeySetVerifier_UsesLocalJWKSNotDiscoveryURI is the load-bearing
// corner case: the discovery document's jwks_uri points at a decoy serving the
// WRONG key, while the local /openid/v1/jwks serves the real key. Verification
// must succeed, proving the verifier ignores jwks_uri and fetches keys from the
// hardcoded local endpoint so the request never leaves the cluster network.
func TestNewLocalKeySetVerifier_UsesLocalJWKSNotDiscoveryURI(t *testing.T) {
	s := newLocalAPIServer(t)
	// Make the decoy explicit and unreachable-looking; the local endpoint is what
	// must be used.
	s.discovery.JWKSURI = "https://external.issuer.invalid/keys"

	v, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, oidc.WithHTTPClient(s.Client()))
	if err != nil {
		t.Fatalf("NewLocalKeySetVerifier: %v", err)
	}
	if err := v.BindAudience(testAudience); err != nil {
		t.Fatalf("BindAudience: %v", err)
	}

	token := signWith(t, s.signer, localClaims(s.URL))
	if err := v.Verify(context.Background(), token, testAPIGroup); err != nil {
		t.Fatalf("verification must use the local JWKS, not jwks_uri; got %v", err)
	}
}

// TestNewLocalKeySetVerifier_DeferredUntilBind confirms the returned verifier is
// deferred: before an audience is bound it reports unhealthy and denies every
// token, matching the in-cluster contract (audience comes from the first request).
func TestNewLocalKeySetVerifier_DeferredUntilBind(t *testing.T) {
	s := newLocalAPIServer(t)

	v, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, oidc.WithHTTPClient(s.Client()))
	if err != nil {
		t.Fatalf("NewLocalKeySetVerifier: %v", err)
	}

	if err := v.HealthCheck(); err == nil {
		t.Fatal("expected not-ready before an audience is bound")
	}
	token := signWith(t, s.signer, localClaims(s.URL))
	if err := v.Verify(context.Background(), token, testAPIGroup); err == nil {
		t.Fatal("expected deny before an audience is bound")
	}

	if err := v.BindAudience(testAudience); err != nil {
		t.Fatalf("BindAudience: %v", err)
	}
	if err := v.HealthCheck(); err != nil {
		t.Fatalf("expected ready after bind, got %v", err)
	}
}

// TestNewLocalKeySetVerifier_RejectsWrongIssuerToken confirms a token minted by a
// different issuer is denied even after the audience is bound: the issuer read
// from local discovery is enforced.
func TestNewLocalKeySetVerifier_RejectsWrongIssuerToken(t *testing.T) {
	s := newLocalAPIServer(t)

	v, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, oidc.WithHTTPClient(s.Client()))
	if err != nil {
		t.Fatalf("NewLocalKeySetVerifier: %v", err)
	}
	if err := v.BindAudience(testAudience); err != nil {
		t.Fatalf("BindAudience: %v", err)
	}

	token := signWith(t, s.signer, localClaims("https://someone.else.example"))
	if err := v.Verify(context.Background(), token, testAPIGroup); err == nil {
		t.Fatal("expected a token with a foreign issuer to be denied")
	}
}

// TestNewLocalKeySetVerifier_TrimsTrailingSlash confirms a trailing slash on the
// apiserver URL does not produce a double slash in the discovery/JWKS paths.
func TestNewLocalKeySetVerifier_TrimsTrailingSlash(t *testing.T) {
	s := newLocalAPIServer(t)

	v, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL+"/", oidc.WithHTTPClient(s.Client()))
	if err != nil {
		t.Fatalf("NewLocalKeySetVerifier with trailing slash: %v", err)
	}
	if err := v.BindAudience(testAudience); err != nil {
		t.Fatalf("BindAudience: %v", err)
	}
	token := signWith(t, s.signer, localClaims(s.URL))
	if err := v.Verify(context.Background(), token, testAPIGroup); err != nil {
		t.Fatalf("expected verification to succeed, got %v", err)
	}
}

// TestNewLocalKeySetVerifier_ConstructionErrors covers every way construction can
// fail before a verifier exists: no URL, and discovery documents that are absent,
// malformed, or missing the issuer. Each must error rather than build a verifier
// against an unknown issuer.
func TestNewLocalKeySetVerifier_ConstructionErrors(t *testing.T) {
	t.Run("empty apiServerURL", func(t *testing.T) {
		if _, err := oidc.NewLocalKeySetVerifier(context.Background(), ""); err == nil {
			t.Fatal("expected an error for an empty apiServerURL")
		}
	})

	t.Run("discovery document not found", func(t *testing.T) {
		s := newLocalAPIServer(t)
		s.wellKnown404 = true
		if _, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, oidc.WithHTTPClient(s.Client())); err == nil {
			t.Fatal("expected an error when the discovery document is missing")
		}
	})

	t.Run("discovery document malformed", func(t *testing.T) {
		s := newLocalAPIServer(t)
		s.discoveryRaw = "{not json"
		if _, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, oidc.WithHTTPClient(s.Client())); err == nil {
			t.Fatal("expected an error for a malformed discovery document")
		}
	})

	t.Run("discovery document without issuer", func(t *testing.T) {
		s := newLocalAPIServer(t)
		s.discovery.Issuer = ""
		if _, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, oidc.WithHTTPClient(s.Client())); err == nil {
			t.Fatal("expected an error when the discovery document has no issuer")
		}
	})
}
