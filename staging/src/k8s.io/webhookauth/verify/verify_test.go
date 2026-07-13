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

const testGroup = "apps"

// TODO(kep-6060): rebuild after review. The claims-decode tests (namespaced vs.
// bare allowedAPIGroup key) moved to the oidc package along with the claim
// decoding and should be rebuilt there. The removed policy cases (bound-object,
// len>1 reject) do not apply. See kep-6060-review-2.2-actions.md.

// fakeAuthenticator is a stand-in TokenAuthenticator returning a fixed set of
// allowedAPIGroups (or an error), so the policy layer is tested without crypto.
type fakeAuthenticator struct {
	groups []string
	err    error
}

func (f fakeAuthenticator) AuthenticateToken(_ context.Context, _ string) ([]string, error) {
	return f.groups, f.err
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
	if _, err := NewVerifier(fakeAuthenticator{groups: []string{testGroup}}); err != nil {
		t.Fatalf("unexpected error for valid authenticator: %v", err)
	}
}

// TestVerify covers the policy: the authenticator's allowedAPIGroup list must
// contain the review group or "*"; any failure returns ErrVerificationFailed.
func TestVerify(t *testing.T) {
	tests := []struct {
		name    string
		groups  []string
		group   string
		wantErr bool
	}{
		{name: "exact group -> accepted", groups: []string{testGroup}},
		{name: "wildcard -> matches any group -> accepted", groups: []string{wildcardAPIGroup}, group: "any.group.example.com"},
		{name: "multi-value list containing the group -> accepted", groups: []string{"extensions", testGroup}},
		{name: "empty list -> rejected", groups: []string{}, wantErr: true},
		{name: "group not authorized -> rejected", groups: []string{"extensions"}, wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			group := tc.group
			if group == "" {
				group = testGroup
			}
			v := mustVerifier(t, fakeAuthenticator{groups: tc.groups})
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
