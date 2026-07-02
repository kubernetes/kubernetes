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
	"testing"
	"time"

	jose "gopkg.in/go-jose/go-jose.v2"

	"k8s.io/client-go/webhook/authentication/verify"
)

const (
	testAudience = "webhook.example.com"
	testGroup    = "apps"
	keyID        = "test-key-1"
)

// signingHarness holds a key pair, its published JWKS, and a signer.
type signingHarness struct {
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
	return &signingHarness{signer: signer, jwks: jwks}
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
	v, err := verify.NewVerifier(NewStaticKeySet(h.jwks), []string{testAudience})
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
