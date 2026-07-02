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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"testing"
	"time"

	jose "gopkg.in/go-jose/go-jose.v2"

	"k8s.io/client-go/webhook/authentication/verify"
)

const (
	testAudience = "webhook.example.com"
	testGroup    = "apps"
	testIssuer   = "https://issuer.example.com"
	keyID        = "test-key-1"
)

// signingHarness holds a key pair, its published JWKS, and a signer.
type signingHarness struct {
	priv   *rsa.PrivateKey
	signer jose.Signer
	jwks   jose.JSONWebKeySet
}

func newSigningHarness(t *testing.T) *signingHarness {
	t.Helper()
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	signer, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.RS256, Key: jose.JSONWebKey{Key: priv, KeyID: keyID}},
		(&jose.SignerOptions{}).WithType("JWT"),
	)
	if err != nil {
		t.Fatalf("new signer: %v", err)
	}
	jwks := jose.JSONWebKeySet{Keys: []jose.JSONWebKey{
		{Key: priv.Public(), KeyID: keyID, Algorithm: string(jose.RS256), Use: "sig"},
	}}
	return &signingHarness{priv: priv, signer: signer, jwks: jwks}
}

// mint signs the given claims map into a compact JWS.
func (h *signingHarness) mint(t *testing.T, claims map[string]interface{}) string {
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

func validatingClaims() map[string]interface{} {
	return map[string]interface{}{
		"iss": "https://issuer.example.com",
		"sub": "system:serviceaccount:kube-system:webhook-auth",
		"aud": []string{testAudience},
		"exp": time.Now().Add(5 * time.Minute).Unix(),
		"kubernetes.io": map[string]interface{}{
			"validatingWebhookConfiguration": map[string]string{"name": "vwc", "uid": "vwc-uid"},
			"attestationClaims": map[string][]string{
				verify.AllowedAPIGroupClaimKey: {testGroup},
			},
		},
	}
}

func newVerifier(t *testing.T, h *signingHarness) *verify.Verifier {
	t.Helper()
	v, err := verify.NewVerifier(NewStaticKeySet(h.jwks), testIssuer, []string{testAudience})
	if err != nil {
		t.Fatalf("NewVerifier: %v", err)
	}
	return v
}

// TestEndToEnd mints a real RS256-signed token and verifies it end to end
// through StaticKeySet + the core verifier, covering both bound-object kinds.
func TestEndToEnd(t *testing.T) {
	tests := []struct {
		name string
		// mutate customizes the base (validating) claims; nil leaves them as-is.
		mutate    func(claims map[string]interface{})
		wantKind  string
		wantName  string
		wantGroup string
	}{
		{
			name:      "validating webhook config bound token -> accepted",
			wantKind:  verify.KindValidatingWebhookConfiguration,
			wantName:  "vwc",
			wantGroup: testGroup,
		},
		{
			name: "mutating webhook config bound token -> accepted",
			mutate: func(claims map[string]interface{}) {
				k8s := claims["kubernetes.io"].(map[string]interface{})
				delete(k8s, "validatingWebhookConfiguration")
				k8s["mutatingWebhookConfiguration"] = map[string]string{"name": "mwc", "uid": "mwc-uid"}
			},
			wantKind: verify.KindMutatingWebhookConfiguration,
			wantName: "mwc",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h := newSigningHarness(t)
			v := newVerifier(t, h)

			claims := validatingClaims()
			if tc.mutate != nil {
				tc.mutate(claims)
			}

			tok := h.mint(t, claims)
			res, err := v.Verify(context.Background(), tok, testGroup)
			if err != nil {
				t.Fatalf("Verify: %v", err)
			}
			if res.BoundObjectKind != tc.wantKind {
				t.Errorf("BoundObjectKind = %q, want %q", res.BoundObjectKind, tc.wantKind)
			}
			if res.BoundObjectName != tc.wantName {
				t.Errorf("BoundObjectName = %q, want %q", res.BoundObjectName, tc.wantName)
			}
			if tc.wantGroup != "" && res.AllowedAPIGroup != tc.wantGroup {
				t.Errorf("AllowedAPIGroup = %q, want %q", res.AllowedAPIGroup, tc.wantGroup)
			}
		})
	}
}

// TestEndToEnd_WrongKey_Rejected proves signature verification actually happens:
// a token signed by a different key must not verify against the published JWKS.
// This stays a focused test because it needs two independent signing harnesses.
func TestEndToEnd_WrongKey_Rejected(t *testing.T) {
	issuer := newSigningHarness(t)
	attacker := newSigningHarness(t)

	// Verifier trusts only the issuer's JWKS, but the token is signed by attacker.
	v := newVerifier(t, issuer)
	tok := attacker.mint(t, validatingClaims())

	if _, err := v.Verify(context.Background(), tok, testGroup); err == nil {
		t.Fatal("expected signature verification to fail for a foreign key")
	}
}

// TestStaticKeySet_VerifySignature covers the keyset-level contract directly:
// a well-formed JWS returns the decoded payload, while a non-JWS input errors.
func TestStaticKeySet_VerifySignature(t *testing.T) {
	tests := []struct {
		name string
		// token builds the input from the harness; the returned value is verified.
		token   func(t *testing.T, h *signingHarness) string
		wantErr bool
	}{
		{
			name: "well-formed signed JWS -> payload returned",
			token: func(t *testing.T, h *signingHarness) string {
				return h.mint(t, validatingClaims())
			},
			wantErr: false,
		},
		{
			name: "non-JWS input -> error",
			token: func(t *testing.T, h *signingHarness) string {
				return "not-a-jws"
			},
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h := newSigningHarness(t)
			ks := NewStaticKeySet(h.jwks)

			payload, err := ks.VerifySignature(context.Background(), tc.token(t, h))
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("VerifySignature: %v", err)
			}
			if len(payload) == 0 {
				t.Error("expected non-empty payload")
			}
		})
	}
}

// TestStaticKeySet_AlgorithmAllowlist proves the explicit accepted-algorithm
// allowlist: alg:none and HS256 (symmetric MAC) tokens are rejected even when
// crafted to target the trusted RSA key, while normal RS256 and ES256 tokens are
// accepted.
func TestStaticKeySet_AlgorithmAllowlist(t *testing.T) {
	t.Run("alg none -> rejected", func(t *testing.T) {
		// A token with no signature and alg:none must never verify.
		header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none","typ":"JWT"}`))
		payload, err := json.Marshal(validatingClaims())
		if err != nil {
			t.Fatalf("marshal claims: %v", err)
		}
		body := base64.RawURLEncoding.EncodeToString(payload)
		tok := header + "." + body + "."

		h := newSigningHarness(t)
		ks := NewStaticKeySet(h.jwks)
		if _, err := ks.VerifySignature(context.Background(), tok); err == nil {
			t.Fatal("expected alg:none token to be rejected, got nil")
		}
	})

	t.Run("HS256 keyed with RSA public bytes -> rejected", func(t *testing.T) {
		// Classic key-confusion attack: sign HS256 using the RSA public key bytes
		// as the shared secret, then present it against the RSA JWK. The allowlist
		// rejects HS256 before any verification is attempted.
		h := newSigningHarness(t)
		pubDER, err := x509.MarshalPKIXPublicKey(h.priv.Public())
		if err != nil {
			t.Fatalf("marshal public key: %v", err)
		}
		hmacSigner, err := jose.NewSigner(jose.SigningKey{Algorithm: jose.HS256, Key: pubDER}, nil)
		if err != nil {
			t.Fatalf("new HMAC signer: %v", err)
		}
		payload, err := json.Marshal(validatingClaims())
		if err != nil {
			t.Fatalf("marshal claims: %v", err)
		}
		jws, err := hmacSigner.Sign(payload)
		if err != nil {
			t.Fatalf("sign HS256: %v", err)
		}
		tok, err := jws.CompactSerialize()
		if err != nil {
			t.Fatalf("serialize: %v", err)
		}

		ks := NewStaticKeySet(h.jwks)
		if _, err := ks.VerifySignature(context.Background(), tok); err == nil {
			t.Fatal("expected HS256 key-confusion token to be rejected, got nil")
		}
	})

	t.Run("RS256 -> accepted", func(t *testing.T) {
		h := newSigningHarness(t)
		ks := NewStaticKeySet(h.jwks)
		if _, err := ks.VerifySignature(context.Background(), h.mint(t, validatingClaims())); err != nil {
			t.Fatalf("expected RS256 token to verify: %v", err)
		}
	})

	t.Run("ES256 -> accepted", func(t *testing.T) {
		priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
		if err != nil {
			t.Fatalf("generate EC key: %v", err)
		}
		signer, err := jose.NewSigner(
			jose.SigningKey{Algorithm: jose.ES256, Key: jose.JSONWebKey{Key: priv, KeyID: "ec-kid"}},
			(&jose.SignerOptions{}).WithType("JWT"),
		)
		if err != nil {
			t.Fatalf("new EC signer: %v", err)
		}
		payload, err := json.Marshal(validatingClaims())
		if err != nil {
			t.Fatalf("marshal claims: %v", err)
		}
		jws, err := signer.Sign(payload)
		if err != nil {
			t.Fatalf("sign ES256: %v", err)
		}
		tok, err := jws.CompactSerialize()
		if err != nil {
			t.Fatalf("serialize: %v", err)
		}
		jwks := jose.JSONWebKeySet{Keys: []jose.JSONWebKey{
			{Key: priv.Public(), KeyID: "ec-kid", Algorithm: string(jose.ES256), Use: "sig"},
		}}
		ks := NewStaticKeySet(jwks)
		if _, err := ks.VerifySignature(context.Background(), tok); err != nil {
			t.Fatalf("expected ES256 token to verify: %v", err)
		}
	})
}

// TestStaticKeySet_FiltersNonSignatureKeys proves the JWKS is filtered to
// signature-usable keys: a symmetric ("oct") key present in the set is dropped
// so its shared secret can never be trusted to verify a token.
func TestStaticKeySet_FiltersNonSignatureKeys(t *testing.T) {
	h := newSigningHarness(t)
	octKey := jose.JSONWebKey{Key: []byte("a-shared-secret-value"), KeyID: "oct-kid", Use: "sig"}

	mixed := jose.JSONWebKeySet{Keys: append([]jose.JSONWebKey{octKey}, h.jwks.Keys...)}
	ks := NewStaticKeySet(mixed)

	if len(ks.keys.Keys) != 1 {
		t.Fatalf("expected the oct key to be filtered out, got %d keys", len(ks.keys.Keys))
	}
	if _, symmetric := ks.keys.Keys[0].Key.([]byte); symmetric {
		t.Fatal("retained key is symmetric; oct key was not filtered")
	}
	// The remaining RSA key still verifies a genuine token.
	if _, err := ks.VerifySignature(context.Background(), h.mint(t, validatingClaims())); err != nil {
		t.Fatalf("expected RSA key to still verify after filtering: %v", err)
	}
}
