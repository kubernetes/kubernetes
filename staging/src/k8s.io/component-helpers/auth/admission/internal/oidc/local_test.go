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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	jose "github.com/go-jose/go-jose/v4"

	"k8s.io/component-helpers/auth/admission/internal/oidc"
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
			"validatingwebhookconfiguration": map[string]any{"name": "vwc", "uid": "vwc-uid"},
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

// newLocalAPIServer advertises a decoy JWKS so tests detect use of discovery's
// jwks_uri instead of the in-cluster key endpoint.
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
	mux.HandleFunc("/openid/v1/jwks", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(jwksBytes(t, &priv.PublicKey))
	})
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

	v, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, testAudience, s.Client())
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

func TestNewLocalKeySetVerifier_SigningAlgorithms(t *testing.T) {
	privateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generating ECDSA key: %v", err)
	}
	signer, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.ES256, Key: jose.JSONWebKey{Key: privateKey, KeyID: "ecdsa-test-key"}},
		(&jose.SignerOptions{}).WithType("JWT"),
	)
	if err != nil {
		t.Fatalf("creating ECDSA signer: %v", err)
	}
	keys := jose.JSONWebKeySet{Keys: []jose.JSONWebKey{{
		Key: &privateKey.PublicKey, KeyID: "ecdsa-test-key", Algorithm: string(jose.ES256), Use: "sig",
	}}}

	var issuer string
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer": issuer, "jwks_uri": issuer + "/openid/v1/jwks",
			"id_token_signing_alg_values_supported": []string{string(jose.ES256)},
		})
	})
	mux.HandleFunc("/openid/v1/jwks", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(keys)
	})
	server := httptest.NewTLSServer(mux)
	defer server.Close()
	issuer = server.URL

	verifier, err := oidc.NewLocalKeySetVerifier(context.Background(), issuer, testAudience, server.Client())
	if err != nil {
		t.Fatalf("building local verifier for ECDSA issuer: %v", err)
	}
	if err := verifier.BindAudience(testAudience); err != nil {
		t.Fatalf("binding ECDSA verifier audience: %v", err)
	}
	// The in-cluster path must accept a key algorithm the kube-apiserver can issue.
	if err := verifier.Verify(context.Background(), signWith(t, signer, localClaims(issuer)), testAPIGroup); err != nil {
		t.Fatalf("verify ECDSA-signed webhook token with local JWKS: %v", err)
	}

	// An added symmetric key must not make a bearer token's HMAC algorithm valid.
	secret := []byte("hmac-signing-key-for-unsupported-algorithm-test")
	macSigner, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.HS256, Key: jose.JSONWebKey{Key: secret, KeyID: "hmac-test-key"}},
		(&jose.SignerOptions{}).WithType("JWT"),
	)
	if err != nil {
		t.Fatalf("creating HMAC signer: %v", err)
	}
	keys.Keys = append(keys.Keys, jose.JSONWebKey{
		Key: secret, KeyID: "hmac-test-key", Algorithm: string(jose.HS256), Use: "sig",
	})
	if err := verifier.Verify(context.Background(), signWith(t, macSigner, localClaims(issuer)), testAPIGroup); err == nil {
		t.Fatal("local verifier accepted an HMAC-signed token")
	}
}

// TestNewLocalKeySetVerifier_UsesLocalJWKSNotDiscoveryURI checks that an
// externally advertised JWKS URI does not replace the local key endpoint.
func TestNewLocalKeySetVerifier_UsesLocalJWKSNotDiscoveryURI(t *testing.T) {
	s := newLocalAPIServer(t)
	s.discovery.JWKSURI = "https://external.issuer.invalid/keys"

	v, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, testAudience, s.Client())
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

// TestNewLocalKeySetVerifier_RequiresAudienceBeforeDiscovery ensures an
// unconfigured receiver fails before it can trust any incoming token.
func TestNewLocalKeySetVerifier_RequiresAudienceBeforeDiscovery(t *testing.T) {
	if _, err := oidc.NewLocalKeySetVerifier(context.Background(), "https://unreachable.invalid", "", &http.Client{}); err == nil {
		t.Fatal("expected a missing audience to fail before discovery")
	}

	s := newLocalAPIServer(t)
	v, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, testAudience, s.Client())
	if err != nil {
		t.Fatalf("NewLocalKeySetVerifier with a trusted audience: %v", err)
	}
	if err := v.HealthCheck(); err != nil {
		t.Fatalf("expected ready after construction: %v", err)
	}
	if err := v.Verify(context.Background(), signWith(t, s.signer, localClaims(s.URL)), testAPIGroup); err != nil {
		t.Fatalf("verify token for the configured audience: %v", err)
	}
}

// TestNewLocalKeySetVerifier_RejectsWrongIssuerToken confirms a token minted by a
// different issuer is denied even after the audience is bound: the issuer read
// from local discovery is enforced.
func TestNewLocalKeySetVerifier_RejectsWrongIssuerToken(t *testing.T) {
	s := newLocalAPIServer(t)

	v, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, testAudience, s.Client())
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

	v, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL+"/", testAudience, s.Client())
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

// TestNewLocalKeySetVerifier_ConstructionErrors covers each way construction can
// fail before a verifier exists — no URL, and discovery documents that are
// absent, malformed, or missing the issuer — each erroring rather than building
// against an unknown issuer.
func TestNewLocalKeySetVerifier_ConstructionErrors(t *testing.T) {
	t.Run("empty apiServerURL", func(t *testing.T) {
		if _, err := oidc.NewLocalKeySetVerifier(context.Background(), "", testAudience, &http.Client{}); err == nil {
			t.Fatal("expected an error for an empty apiServerURL")
		}
	})

	t.Run("discovery document not found", func(t *testing.T) {
		s := newLocalAPIServer(t)
		s.wellKnown404 = true
		if _, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, testAudience, s.Client()); err == nil {
			t.Fatal("expected an error when the discovery document is missing")
		}
	})

	t.Run("discovery document malformed", func(t *testing.T) {
		s := newLocalAPIServer(t)
		s.discoveryRaw = "{not json"
		if _, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, testAudience, s.Client()); err == nil {
			t.Fatal("expected an error for a malformed discovery document")
		}
	})

	t.Run("discovery document without issuer", func(t *testing.T) {
		s := newLocalAPIServer(t)
		s.discovery.Issuer = ""
		if _, err := oidc.NewLocalKeySetVerifier(context.Background(), s.URL, testAudience, s.Client()); err == nil {
			t.Fatal("expected an error when the discovery document has no issuer")
		}
	})
}
