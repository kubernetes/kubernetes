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

const (
	testAudience = "webhook.example.com"
	testGroup    = "apps"
	testIssuer   = "https://issuer.example.com"
	testSubject  = "system:serviceaccount:kube-system:webhook-auth"
)

// fakeAuthenticator is a stand-in TokenAuthenticator that exercises the policy
// layer without any real crypto. When err is non-nil it simulates a
// signature/standard-claim failure; otherwise it returns claims verbatim. This
// keeps the core policy tests pure-stdlib and offline.
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

// baseClaims returns a valid, already-authenticated claim set. Tests mutate it
// to drive each policy branch.
func baseClaims() *VerifiedClaims {
	return &VerifiedClaims{
		Issuer:   testIssuer,
		Subject:  testSubject,
		Audience: []string{testAudience},
		Kubernetes: kubernetesClaims{
			ValidatingWebhookConfiguration: &objectRef{Name: "vwc", UID: "vwc-uid"},
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

// TestVerify exercises the KEP-6060 policy the core Verifier owns: the
// bound-object exactly-one rule and the namespaced allowedAPIGroup match. Each
// case starts from baseClaims(), optionally mutates it, and asserts either a
// successful Result or a generic failure carrying the expected log reason.
func TestVerify(t *testing.T) {
	tests := []struct {
		name string
		// mutate customizes the base claims before verification; nil leaves them.
		mutate func(c *VerifiedClaims)
		// group is the reviewAPIGroup argument; empty defaults to testGroup.
		group string
		// wantReason is the expected log reason; empty means the happy path.
		wantReason string
		// check runs extra assertions on the Result of a happy-path case.
		check func(t *testing.T, res *Result)
	}{
		{
			name: "validating webhook config, exact group -> accepted",
			check: func(t *testing.T, res *Result) {
				if res.BoundObjectKind != kindValidatingWebhookConfiguration {
					t.Errorf("BoundObjectKind = %q, want %q", res.BoundObjectKind, kindValidatingWebhookConfiguration)
				}
				if res.BoundObjectName != "vwc" || res.BoundObjectUID != "vwc-uid" {
					t.Errorf("bound object = %s/%s, want vwc/vwc-uid", res.BoundObjectName, res.BoundObjectUID)
				}
				if res.AllowedAPIGroup != testGroup {
					t.Errorf("AllowedAPIGroup = %q, want %q", res.AllowedAPIGroup, testGroup)
				}
				if res.Subject != testSubject {
					t.Errorf("Subject = %q", res.Subject)
				}
				if res.Issuer != testIssuer {
					t.Errorf("Issuer = %q", res.Issuer)
				}
				if len(res.Audience) != 1 || res.Audience[0] != testAudience {
					t.Errorf("Audience = %v", res.Audience)
				}
			},
		},
		{
			name: "mutating webhook config, exact group -> accepted",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.ValidatingWebhookConfiguration = nil
				c.Kubernetes.MutatingWebhookConfiguration = &objectRef{Name: "mwc", UID: "mwc-uid"}
			},
			check: func(t *testing.T, res *Result) {
				if res.BoundObjectKind != kindMutatingWebhookConfiguration {
					t.Errorf("BoundObjectKind = %q, want %q", res.BoundObjectKind, kindMutatingWebhookConfiguration)
				}
				if res.BoundObjectName != "mwc" {
					t.Errorf("BoundObjectName = %q, want mwc", res.BoundObjectName)
				}
			},
		},
		{
			name: "wildcard allowedAPIGroup -> matches any review group -> accepted",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.AttestationClaims = map[string][]string{allowedAPIGroupClaimKey: {wildcardAPIGroup}}
			},
			group: "any.group.example.com",
			check: func(t *testing.T, res *Result) {
				if res.AllowedAPIGroup != wildcardAPIGroup {
					t.Errorf("AllowedAPIGroup = %q, want %q", res.AllowedAPIGroup, wildcardAPIGroup)
				}
			},
		},
		{
			name: "both bound objects set -> reasonBothBoundObjects",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.MutatingWebhookConfiguration = &objectRef{Name: "mwc", UID: "mwc-uid"}
			},
			wantReason: reasonBothBoundObjects,
		},
		{
			name: "neither bound object set -> reasonNoBoundObject",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.ValidatingWebhookConfiguration = nil
			},
			wantReason: reasonNoBoundObject,
		},
		{
			name: "allowedAPIGroup claim absent -> reasonMissingAllowedAPIGroup",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.AttestationClaims = map[string][]string{}
			},
			wantReason: reasonMissingAllowedAPIGroup,
		},
		{
			name: "bare allowedAPIGroup key -> treated as missing -> reasonMissingAllowedAPIGroup",
			// A spec-violating issuer emitting the bare "allowedAPIGroup" key (not
			// the namespaced form) must be treated as missing, never accepted.
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.AttestationClaims = map[string][]string{"allowedAPIGroup": {testGroup}}
			},
			wantReason: reasonMissingAllowedAPIGroup,
		},
		{
			name: "allowedAPIGroup with more than one value -> reasonMissingAllowedAPIGroup",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.AttestationClaims = map[string][]string{allowedAPIGroupClaimKey: {testGroup, "extensions"}}
			},
			wantReason: reasonMissingAllowedAPIGroup,
		},
		{
			name:       "review group not authorized by claim -> reasonAPIGroupNotAuthorized",
			group:      "batch",
			wantReason: reasonAPIGroupNotAuthorized,
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
			res, err := v.Verify(context.Background(), "raw-token", group)

			if tc.wantReason == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if res == nil {
					t.Fatal("expected a Result, got nil")
				}
				if tc.check != nil {
					tc.check(t, res)
				}
				return
			}

			if err == nil {
				t.Fatalf("expected failure, got result %+v", res)
			}
			if !errors.Is(err, ErrVerificationFailed) {
				t.Fatalf("error must satisfy ErrVerificationFailed, got %v", err)
			}
			if got := Reason(err); got != tc.wantReason {
				t.Fatalf("Reason = %q, want %q", got, tc.wantReason)
			}
			// The generic error string must never leak the reason or claim values.
			if strings.Contains(err.Error(), tc.wantReason) {
				t.Fatalf("error string leaked the reason: %q", err.Error())
			}
		})
	}
}

// TestVerify_AntiEnumeration asserts that a rejection never echoes the bound
// webhook name, UID, or the rejected group value in the caller-facing error.
func TestVerify_AntiEnumeration(t *testing.T) {
	v := mustVerifier(t, fakeAuthenticator{claims: baseClaims()})
	_, err := v.Verify(context.Background(), "raw-token", "batch")
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, ErrVerificationFailed) {
		t.Errorf("error %v does not satisfy ErrVerificationFailed", err)
	}
	msg := err.Error()
	for _, leak := range []string{"vwc", "vwc-uid", "batch", testGroup} {
		if strings.Contains(msg, leak) {
			t.Errorf("error message %q leaks %q", msg, leak)
		}
	}
}

// TestVerify_AuthenticatorFailureIsGeneric confirms the anti-enumeration
// contract at the authenticator seam: any error the authenticator reports
// (signature, issuer, audience, expiry) collapses into the single generic
// ErrVerificationFailed. The descriptive text survives ONLY as a log reason and
// is never present in the returned error's string.
func TestVerify_AuthenticatorFailureIsGeneric(t *testing.T) {
	secret := "aud claim mismatch: expected https://victim got https://attacker"
	v := mustVerifier(t, fakeAuthenticator{err: errors.New(secret)})

	res, err := v.Verify(context.Background(), "raw-token", testGroup)
	if err == nil {
		t.Fatalf("expected failure, got result %+v", res)
	}
	if !errors.Is(err, ErrVerificationFailed) {
		t.Fatalf("error must satisfy ErrVerificationFailed, got %v", err)
	}
	// Anti-enumeration: the descriptive authenticator text must NOT appear in
	// the error surfaced to the caller.
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("caller-facing error leaked authenticator detail: %q", err.Error())
	}
	if strings.Contains(err.Error(), "attacker") || strings.Contains(err.Error(), "victim") {
		t.Fatalf("caller-facing error leaked claim values: %q", err.Error())
	}
	// The descriptive text is retained for operator logging only.
	if reason := Reason(err); !strings.Contains(reason, secret) {
		t.Fatalf("Reason should retain the authenticator detail for logging, got %q", reason)
	}
}
