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
	"errors"
	"fmt"
	"strings"
	"testing"
)

// TestFail_GenericSurface pins the core error contract Fail must satisfy: the
// returned error is the single generic failure (errors.Is succeeds), its
// stringified form is the generic message ONLY, and the descriptive reason is
// reachable exclusively through Reason — never through Error().
func TestFail_GenericSurface(t *testing.T) {
	const reason = "some very specific internal reason"
	err := Fail(reason)

	if err == nil {
		t.Fatal("Fail returned nil")
	}
	if !errors.Is(err, ErrVerificationFailed) {
		t.Errorf("Fail(...) must satisfy errors.Is(err, ErrVerificationFailed)")
	}
	if got := err.Error(); got != ErrVerificationFailed.Error() {
		t.Errorf("Error() = %q, want the generic message %q", got, ErrVerificationFailed.Error())
	}
	if strings.Contains(err.Error(), reason) {
		t.Errorf("Error() %q leaked the reason %q", err.Error(), reason)
	}
	if got := Reason(err); got != reason {
		t.Errorf("Reason() = %q, want %q", got, reason)
	}
}

// TestReason_NonVerificationErrors confirms Reason returns "" for inputs it does
// not own: nil and any error that is not a verification failure. A caller must
// not be able to coax a log reason out of an unrelated error.
func TestReason_NonVerificationErrors(t *testing.T) {
	if got := Reason(nil); got != "" {
		t.Errorf("Reason(nil) = %q, want \"\"", got)
	}
	if got := Reason(errors.New("unrelated")); got != "" {
		t.Errorf("Reason(unrelated) = %q, want \"\"", got)
	}
	// errNilAuthenticator is a construction error, not a verification failure, so
	// it carries no reason.
	if got := Reason(errNilAuthenticator); got != "" {
		t.Errorf("Reason(errNilAuthenticator) = %q, want \"\"", got)
	}
}

// TestReason_UnwrapsWrappedFailure confirms Reason (via errors.As) still finds
// the verification failure when it has been wrapped with additional context by
// an intermediate layer, so operators keep the log reason even after wrapping.
func TestReason_UnwrapsWrappedFailure(t *testing.T) {
	const reason = "bound-object rule violated"
	wrapped := fmt.Errorf("adapter context: %w", Fail(reason))

	if !errors.Is(wrapped, ErrVerificationFailed) {
		t.Errorf("wrapped error must still satisfy ErrVerificationFailed")
	}
	if got := Reason(wrapped); got != reason {
		t.Errorf("Reason(wrapped) = %q, want %q", got, reason)
	}
}

// TestErrNilAuthenticator_IsNotVerificationFailure guards the boundary between a
// startup misconfiguration and a runtime verification failure: NewVerifier(nil)
// must NOT masquerade as ErrVerificationFailed, so callers that (correctly)
// treat verification failures as an authn decision never confuse them with a
// programming error.
func TestErrNilAuthenticator_IsNotVerificationFailure(t *testing.T) {
	_, err := NewVerifier(nil)
	if err == nil {
		t.Fatal("NewVerifier(nil) returned no error")
	}
	if errors.Is(err, ErrVerificationFailed) {
		t.Errorf("nil-authenticator construction error must not satisfy ErrVerificationFailed")
	}
}

// TestAuthenticationFailedReason covers both branches of the log-reason builder:
// a nil authenticator error yields the base reason, and a non-nil error appends
// the authenticator's descriptive text (for logging only).
func TestAuthenticationFailedReason(t *testing.T) {
	if got := authenticationFailedReason(nil); got != reasonTokenAuthenticationFailed {
		t.Errorf("authenticationFailedReason(nil) = %q, want %q", got, reasonTokenAuthenticationFailed)
	}

	detail := "oidc: token is expired"
	got := authenticationFailedReason(errors.New(detail))
	if !strings.HasPrefix(got, reasonTokenAuthenticationFailed) {
		t.Errorf("authenticationFailedReason(err) = %q, want it to start with the base reason", got)
	}
	if !strings.Contains(got, detail) {
		t.Errorf("authenticationFailedReason(err) = %q, want it to append %q for logging", got, detail)
	}
}

// TestReasonStrings_CarryNoClaimValues asserts the internal reason constants are
// non-empty (useful to operators) and free of anything that looks like an
// interpolated claim value. This is a belt-and-suspenders check on the
// anti-enumeration posture: the reasons are log-only, but they must not become a
// side channel even in logs.
func TestReasonStrings_CarryNoClaimValues(t *testing.T) {
	reasons := []string{
		reasonBothBoundObjects,
		reasonNoBoundObject,
		reasonMissingAllowedAPIGroup,
		reasonAPIGroupNotAuthorized,
		reasonTokenAuthenticationFailed,
	}
	for _, r := range reasons {
		if r == "" {
			t.Error("reason string is empty; operators need a diagnostic")
		}
		// A "%" would betray an accidental fmt.Sprintf of a claim value into a
		// reason constant.
		if strings.ContainsAny(r, "%") {
			t.Errorf("reason %q appears to interpolate a value; reasons must be static", r)
		}
	}
}
