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
	"testing"
)

// TODO(kep-6060): rebuild the full anti-enumeration matrix after review. This
// slimmed version only checks that two representative rejection paths (a policy
// rejection and an authenticator failure carrying secret detail) return the
// identical generic ErrVerificationFailed. The removed suite asserted EVERY
// rejection path is byte-for-byte indistinguishable to the caller and that the
// specific reason lives only in logs. See kep-6060-review-2.2-actions.md.
func TestAntiEnumeration_RejectionsAreGeneric(t *testing.T) {
	generic := ErrVerificationFailed.Error()

	// Policy rejection: the review group is not authorized by the claim.
	policyErr := mustVerifier(t, fakeAuthenticator{claims: baseClaims()}).
		Verify(context.Background(), "raw-token", "batch")

	// Authenticator rejection carrying descriptive (secret) detail that must
	// never reach the caller.
	secret := "aud mismatch: expected https://victim got https://attacker"
	authErr := mustVerifier(t, fakeAuthenticator{err: errors.New(secret)}).
		Verify(context.Background(), "raw-token", testGroup)

	for _, err := range []error{policyErr, authErr} {
		if err == nil {
			t.Fatal("expected a rejection")
		}
		if !errors.Is(err, ErrVerificationFailed) {
			t.Fatalf("error must satisfy ErrVerificationFailed, got %v", err)
		}
		if err.Error() != generic {
			t.Fatalf("caller-visible message = %q, want the generic %q", err.Error(), generic)
		}
	}
	if policyErr.Error() != authErr.Error() {
		t.Fatal("rejection messages are distinguishable across paths")
	}
}
