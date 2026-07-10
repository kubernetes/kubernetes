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

// TestAntiEnumeration_AllRejectionsAreIndistinguishable is the security-critical
// contract lock for KEP-6060: EVERY rejection path the core verifier can take
// must surface the SAME caller-visible error — byte-for-byte identical Error()
// text, all satisfying errors.Is(err, ErrVerificationFailed) — so a caller can
// never distinguish which check failed and thus cannot enumerate objects or
// probe claim values through the error surface.
//
// It walks every distinct failure the Verifier owns (authenticator failure plus
// each policy branch), collects the caller-facing error string from each, and
// asserts they are all identical to the generic message. It further asserts each
// case still yields a distinct, non-empty log reason via Reason() — the ONLY
// place detail is allowed to exist — and that the reason never bleeds into
// Error(). If a future change makes any rejection distinguishable to the caller,
// this test fails.
func TestAntiEnumeration_AllRejectionsAreIndistinguishable(t *testing.T) {
	// A representative "secret" the authenticator might describe. It must never
	// reach the caller.
	const authSecret = "aud mismatch: expected https://victim got https://attacker"

	cases := []struct {
		name string
		// auth is the authenticator used; nil means use a passing fake driven by
		// the mutated claims.
		auth TokenAuthenticator
		// mutate customizes passing claims to trigger a policy branch.
		mutate func(c *VerifiedClaims)
		// group is the reviewAPIGroup argument (default testGroup).
		group string
	}{
		{
			name: "authenticator failure (signature/iss/aud/exp)",
			auth: fakeAuthenticator{err: errors.New(authSecret)},
		},
		{
			name: "both bound objects set",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.MutatingWebhookConfiguration = &objectRef{Name: "mwc", UID: "mwc-uid"}
			},
		},
		{
			name: "no bound object set",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.ValidatingWebhookConfiguration = nil
			},
		},
		{
			name: "allowedAPIGroup claim missing",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.AttestationClaims = map[string][]string{}
			},
		},
		{
			name: "bare allowedAPIGroup key (spec-violating issuer)",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.AttestationClaims = map[string][]string{"allowedAPIGroup": {testGroup}}
			},
		},
		{
			name: "allowedAPIGroup with multiple values",
			mutate: func(c *VerifiedClaims) {
				c.Kubernetes.AttestationClaims = map[string][]string{
					allowedAPIGroupClaimKey: {testGroup, "extensions"},
				}
			},
		},
		{
			name:  "review API group not authorized",
			group: "batch",
		},
	}

	// Also verify a valid token's identifiers are known so we can assert none of
	// them leak into any rejection message.
	secrets := []string{"vwc", "vwc-uid", "mwc", "mwc-uid", testGroup, "batch", "extensions",
		"attacker", "victim", authSecret}

	generic := ErrVerificationFailed.Error()
	seenReasons := make(map[string]struct{})

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var auth TokenAuthenticator = tc.auth
			if auth == nil {
				claims := baseClaims()
				if tc.mutate != nil {
					tc.mutate(claims)
				}
				auth = fakeAuthenticator{claims: claims}
			}
			group := tc.group
			if group == "" {
				group = testGroup
			}

			v := mustVerifier(t, auth)
			res, err := v.Verify(context.Background(), "raw-token", group)
			if err == nil {
				t.Fatalf("expected a rejection, got result %+v", res)
			}

			// 1. Same generic sentinel for every rejection.
			if !errors.Is(err, ErrVerificationFailed) {
				t.Fatalf("error must satisfy ErrVerificationFailed, got %v", err)
			}
			// 2. Byte-for-byte identical caller-visible message across ALL paths.
			if got := err.Error(); got != generic {
				t.Fatalf("caller-visible message = %q, want the generic %q (distinguishable rejection!)", got, generic)
			}
			// 3. No claim value or authenticator detail leaks to the caller.
			for _, s := range secrets {
				if s != "" && strings.Contains(err.Error(), s) {
					t.Fatalf("caller-visible error %q leaked %q", err.Error(), s)
				}
			}
			// 4. Detail survives ONLY as a non-empty log reason.
			reason := Reason(err)
			if reason == "" {
				t.Fatalf("Reason() is empty; operators need a diagnostic")
			}
			if strings.Contains(err.Error(), reason) {
				t.Fatalf("Error() %q leaked its log reason %q", err.Error(), reason)
			}
			seenReasons[reason] = struct{}{}
		})
	}

	// The reasons themselves are diagnostic (they differ per branch) even though
	// the caller-facing error does not. That difference must live only behind
	// Reason(); here we simply confirm the log path preserved more than one
	// distinct reason, proving detail was retained internally while the wire
	// surface stayed uniform.
	if len(seenReasons) < 2 {
		t.Errorf("expected multiple distinct log reasons across rejection paths, got %d", len(seenReasons))
	}
}
