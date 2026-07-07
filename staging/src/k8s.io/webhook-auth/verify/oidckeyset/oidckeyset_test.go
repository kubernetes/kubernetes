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

package oidckeyset_test

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	jose "gopkg.in/go-jose/go-jose.v2"

	"k8s.io/webhook-auth/verify"
	"k8s.io/webhook-auth/verify/oidckeyset"
)

const (
	testKeyID     = "test-signing-key"
	testAudience  = "https://webhook.example.svc/validate"
	testAPIGroup  = "*"
	testSubject   = "system:serviceaccount:kube-system:issuer"
	testWebhookNm = "example-validating-webhook"
	testWebhookID = "11111111-2222-3333-4444-555555555555"
)

// oidcTestServer is an httptest TLS server that serves an OIDC discovery
// document and a JWKS containing a single locally generated RSA signing key.
type oidcTestServer struct {
	server *httptest.Server
	issuer string
	signer jose.Signer
	priv   *rsa.PrivateKey
}

// newOIDCTestServer stands up a TLS server serving discovery + JWKS for a fresh
// RSA key. The returned server's issuer equals the server URL, so go-oidc's
// discovery issuer-match check passes without any skip.
func newOIDCTestServer(t *testing.T) *oidcTestServer {
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

	jwks := jose.JSONWebKeySet{Keys: []jose.JSONWebKey{{
		Key:       priv.Public(),
		KeyID:     testKeyID,
		Algorithm: string(jose.RS256),
		Use:       "sig",
	}}}

	ts := &oidcTestServer{signer: signer, priv: priv}

	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer":                                ts.issuer,
			"jwks_uri":                              ts.issuer + "/keys",
			"id_token_signing_alg_values_supported": []string{string(jose.RS256)},
		})
	})
	mux.HandleFunc("/keys", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(jwks)
	})

	ts.server = httptest.NewTLSServer(mux)
	ts.issuer = ts.server.URL
	t.Cleanup(ts.server.Close)
	return ts
}

// client returns an *http.Client whose transport trusts the test server's TLS
// cert. Injected via oidckeyset.WithHTTPClient so discovery and JWKS fetches
// succeed against the httptest TLS endpoint.
func (ts *oidcTestServer) client() *http.Client { return ts.server.Client() }

// signWith mints a compact JWS over claims using the given signer.
func signWith(t *testing.T, signer jose.Signer, claims map[string]any) string {
	t.Helper()
	payload, err := json.Marshal(claims)
	if err != nil {
		t.Fatalf("marshaling claims: %v", err)
	}
	jws, err := signer.Sign(payload)
	if err != nil {
		t.Fatalf("signing claims: %v", err)
	}
	compact, err := jws.CompactSerialize()
	if err != nil {
		t.Fatalf("serializing JWS: %v", err)
	}
	return compact
}

// sign mints a token signed by the server's advertised key.
func (ts *oidcTestServer) sign(t *testing.T, claims map[string]any) string {
	return signWith(t, ts.signer, claims)
}

// baseClaims returns a valid KEP-6060 token claim set for this server's issuer.
func (ts *oidcTestServer) baseClaims() map[string]any {
	now := time.Now()
	return map[string]any{
		"iss": ts.issuer,
		"sub": testSubject,
		"aud": []string{testAudience},
		"exp": now.Add(time.Hour).Unix(),
		"nbf": now.Add(-time.Minute).Unix(),
		"iat": now.Unix(),
		"kubernetes.io": map[string]any{
			"validatingWebhookConfiguration": map[string]any{
				"name": testWebhookNm,
				"uid":  testWebhookID,
			},
			"attestationClaims": map[string]any{
				verify.AllowedAPIGroupClaimKey: []string{testAPIGroup},
			},
		},
	}
}

// newVerifier builds a Verifier whose signatures are checked by an
// oidckeyset.NewRemoteKeySet pointed at the test server (full OIDC discovery).
func (ts *oidcTestServer) newVerifier(t *testing.T) *verify.Verifier {
	t.Helper()
	ks, err := oidckeyset.NewRemoteKeySet(context.Background(), ts.issuer, oidckeyset.WithHTTPClient(ts.client()))
	if err != nil {
		t.Fatalf("NewRemoteKeySet: %v", err)
	}
	v, err := verify.NewVerifier(ks, ts.issuer, []string{testAudience})
	if err != nil {
		t.Fatalf("NewVerifier: %v", err)
	}
	return v
}

// TestRemoteKeySet_EndToEnd exercises the full round trip: OIDC discovery over
// TLS, JWKS fetch, real RS256 signature verification, and the KEP-6060 claim
// policy layered on top by the core Verifier. This restores the end-to-end
// signature coverage the deleted hand-rolled josekeyset had.
func TestRemoteKeySet_EndToEnd(t *testing.T) {
	ts := newOIDCTestServer(t)

	// A second key NOT published in the JWKS, used to forge a signature.
	wrongPriv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generating wrong key: %v", err)
	}
	wrongSigner, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.RS256, Key: jose.JSONWebKey{Key: wrongPriv, KeyID: testKeyID}},
		(&jose.SignerOptions{}).WithType("JWT"),
	)
	if err != nil {
		t.Fatalf("creating wrong signer: %v", err)
	}

	v := ts.newVerifier(t)

	tests := []struct {
		name    string
		token   func() string
		wantErr bool
	}{
		{
			name:    "valid token signed by the advertised key is accepted",
			token:   func() string { return ts.sign(t, ts.baseClaims()) },
			wantErr: false,
		},
		{
			name: "token signed by a key absent from the JWKS is rejected",
			token: func() string {
				return signWith(t, wrongSigner, ts.baseClaims())
			},
			wantErr: true,
		},
		{
			name: "expired token is rejected",
			token: func() string {
				c := ts.baseClaims()
				c["exp"] = time.Now().Add(-time.Hour).Unix()
				return ts.sign(t, c)
			},
			wantErr: true,
		},
		{
			name: "token asserting a different issuer is rejected",
			token: func() string {
				c := ts.baseClaims()
				c["iss"] = "https://attacker.example.com"
				return ts.sign(t, c)
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			res, err := v.Verify(context.Background(), tc.token(), testAPIGroup)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected verification failure, got result %+v", res)
				}
				if !errors.Is(err, verify.ErrVerificationFailed) {
					t.Fatalf("expected ErrVerificationFailed, got %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if res.Issuer != ts.issuer {
				t.Errorf("issuer = %q, want %q", res.Issuer, ts.issuer)
			}
			if res.BoundObjectName != testWebhookNm {
				t.Errorf("bound object name = %q, want %q", res.BoundObjectName, testWebhookNm)
			}
			if res.AllowedAPIGroup != testAPIGroup {
				t.Errorf("allowedAPIGroup = %q, want %q", res.AllowedAPIGroup, testAPIGroup)
			}
		})
	}
}

// TestRemoteKeySet_VerifySignature tests the KeySet seam directly: a validly
// signed token yields its payload bytes; a token signed by an unknown key errors.
func TestRemoteKeySet_VerifySignature(t *testing.T) {
	ts := newOIDCTestServer(t)
	ks, err := oidckeyset.NewRemoteKeySet(context.Background(), ts.issuer, oidckeyset.WithHTTPClient(ts.client()))
	if err != nil {
		t.Fatalf("NewRemoteKeySet: %v", err)
	}

	t.Run("valid signature returns the payload", func(t *testing.T) {
		token := ts.sign(t, ts.baseClaims())
		payload, err := ks.VerifySignature(context.Background(), token)
		if err != nil {
			t.Fatalf("VerifySignature: %v", err)
		}
		var claims map[string]any
		if err := json.Unmarshal(payload, &claims); err != nil {
			t.Fatalf("payload is not the signed JSON: %v", err)
		}
		if claims["iss"] != ts.issuer {
			t.Errorf("payload iss = %v, want %q", claims["iss"], ts.issuer)
		}
	})

	t.Run("malformed token is rejected", func(t *testing.T) {
		if _, err := ks.VerifySignature(context.Background(), "not-a-jwt"); err == nil {
			t.Fatal("expected error for malformed token")
		}
	})
}

// TestRemoteKeySet_DiscoveryIssuerMismatch confirms go-oidc's issuer-confusion
// guard: if the discovery document advertises a different issuer than requested,
// construction fails.
func TestRemoteKeySet_DiscoveryIssuerMismatch(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer":   "https://someone-else.example.com",
			"jwks_uri": "https://someone-else.example.com/keys",
		})
	})
	server := httptest.NewTLSServer(mux)
	defer server.Close()

	_, err := oidckeyset.NewRemoteKeySet(context.Background(), server.URL, oidckeyset.WithHTTPClient(server.Client()))
	if err == nil {
		t.Fatal("expected discovery issuer mismatch to fail construction")
	}
}

// TestRemoteKeySet_SkipDiscovery exercises the TEST-ONLY discovery-skip path:
// keys are fetched directly from a JWKS URL, bypassing the well-known document.
func TestRemoteKeySet_SkipDiscovery(t *testing.T) {
	ts := newOIDCTestServer(t)

	ks, err := oidckeyset.NewRemoteKeySet(
		context.Background(),
		ts.issuer,
		oidckeyset.WithHTTPClient(ts.client()),
		oidckeyset.WithInsecureSkipDiscovery(ts.issuer+"/keys"),
	)
	if err != nil {
		t.Fatalf("NewRemoteKeySet(skip discovery): %v", err)
	}
	if _, err := ks.VerifySignature(context.Background(), ts.sign(t, ts.baseClaims())); err != nil {
		t.Fatalf("VerifySignature over skip-discovery key set: %v", err)
	}
}
