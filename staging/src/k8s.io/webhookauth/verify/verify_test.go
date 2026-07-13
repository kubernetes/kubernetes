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

package verify

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// TODO(kep-6060): rebuild fuller coverage after review. The Commit-2 slimming
// dropped the bound-object cases (both/neither), the "len>1 -> reject" case
// (multi-value lists are now allowed), and the per-branch Reason() assertions
// with the error taxonomy. See kep-6060-review-2.2-actions.md ("Tests to rebuild").

const (
	testAudience = "webhook.example.com"
	testGroup    = "apps"
	testIssuer   = "https://issuer.example.com"
	testSubject  = "system:serviceaccount:kube-system:webhook-auth"
)

// fakeAuthenticator is a stand-in TokenAuthenticator that exercises the policy
// layer without any real crypto. When err is non-nil it simulates a
// signature/standard-claim failure; otherwise it returns claims verbatim.
type fakeAuthenticator struct {
	claims *VerifiedClaims
	err    error
}

func (f fakeAuthenticator) AuthenticateToken(_ context.Context, _ string) (*VerifiedClaims, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.claims, nil
}

// baseClaims returns a valid, already-authenticated claim set authorizing
// testGroup. Tests mutate it to drive each policy branch.
func baseClaims() *VerifiedClaims {
	return &VerifiedClaims{
		Issuer:   testIssuer,
		Subject:  testSubject,
		Audience: []string{testAudience},
		Kubernetes: kubernetesClaims{
			AttestationClaims: map[string][]string{
				allowedAPIGroupClaimKey: {testGroup},
			},
		},
	}
}

func mustVerifier(t *testing.T, auth TokenAuthenticator) *Verifier {
	t.Helper()
	v, err := NewVerifier(auth)
	if err != nil {
		t.Fatalf("NewVerifier: %v", err)
	}
	return v
}

func TestNewVerifier_Validation(t *testing.T) {
	_, err := NewVerifier(nil)
	if err == nil {
		t.Fatal("expected error for nil authenticator")
	}
	// A nil-authenticator error must NOT masquerade as a verification failure.
	if errors.Is(err, ErrVerificationFailed) {
		t.Fatal("construction error must not satisfy ErrVerificationFailed")
	}
	if _, err := NewVerifier(fakeAuthenticator{claims: baseClaims()}); err != nil {
		t.Fatalf("unexpected error for valid authenticator: %v", err)
	}
}

// TestVerify covers the slimmed policy: the allowedAPIGroup claim's list must
// contain the review group or "*"; any failure returns the generic
// ErrVerificationFailed.
func TestVerify(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(c *VerifiedClaims)
		group   string
		wantErr bool
	}{
		{
			name: "exact group -> accepted",
		},
		{
			name: "wildcard -> matches any group -> accepted",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.AttestationClaims = map[string][]string{allowedAPIGroupClaimKey: {wildcardAPIGroup}}
			},
			group: "any.group.example.com",
		},
		{
			name: "multi-value list containing the group -> accepted",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.AttestationClaims = map[string][]string{allowedAPIGroupClaimKey: {"extensions", testGroup}}
			},
		},
		{
			name: "allowedAPIGroup claim absent -> rejected",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.AttestationClaims = map[string][]string{}
			},
			wantErr: true,
		},
		{
			name: "bare allowedAPIGroup key -> treated as missing -> rejected",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.AttestationClaims = map[string][]string{"allowedAPIGroup": {testGroup}}
			},
			wantErr: true,
		},
		{
			name:    "review group not authorized -> rejected",
			group:   "batch",
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			claims := baseClaims()
			if tc.mutate != nil {
				tc.mutate(claims)
			}
			group := tc.group
			if group == "" {
				group = testGroup
			}
			v := mustVerifier(t, fakeAuthenticator{claims: claims})
			err := v.Verify(context.Background(), "raw-token", group)

			if tc.wantErr {
				if err == nil {
					t.Fatal("expected verification failure, got nil")
				}
				if !errors.Is(err, ErrVerificationFailed) {
					t.Fatalf("error must satisfy ErrVerificationFailed, got %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

// TestVerify_AuthenticatorFailureIsGeneric confirms an authenticator failure
// collapses into the generic ErrVerificationFailed and the caller-facing error
// never contains the authenticator's descriptive detail (anti-enumeration).
func TestVerify_AuthenticatorFailureIsGeneric(t *testing.T) {
	secret := "aud claim mismatch: expected https://victim got https://attacker"
	v := mustVerifier(t, fakeAuthenticator{err: errors.New(secret)})

	err := v.Verify(context.Background(), "raw-token", testGroup)
	if err == nil {
		t.Fatal("expected failure")
	}
	if !errors.Is(err, ErrVerificationFailed) {
		t.Fatalf("error must satisfy ErrVerificationFailed, got %v", err)
	}
	if strings.Contains(err.Error(), secret) || strings.Contains(err.Error(), "attacker") {
		t.Fatalf("caller-facing error leaked authenticator detail: %q", err.Error())
	}
}
