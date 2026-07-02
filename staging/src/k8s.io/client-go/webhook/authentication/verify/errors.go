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
)

// ErrVerificationFailed is the single, generic sentinel that every verification
// failure satisfies (errors.Is(err, ErrVerificationFailed) == true). It is the
// ONLY error value callers may inspect.
//
// The verifier deliberately exposes no per-check error taxonomy: clients must
// not branch on why verification failed (that is an authentication-level
// concern). The specific reason is available ONLY as a non-sensitive log string
// via [Reason], for operators and debugging — never as a typed value to switch
// on. None of the reason strings interpolate claim values (webhook name, UID,
// subject, or API group), so logging them still does not reveal whether a
// specific webhook, group, or object exists.
var ErrVerificationFailed = errors.New("webhook token verification failed")

// reason strings describe which check failed. They are surfaced ONLY through
// [Reason] for logging; they are never returned as distinct error values and
// never embed claim values.
const (
	reasonInvalidSignature       = "signature verification failed"
	reasonMalformedToken         = "token payload is not valid JSON"
	reasonAudienceMismatch       = "token audience does not match the expected audience"
	reasonIssuerMismatch         = "token issuer does not match the expected issuer"
	reasonExpired                = "token is expired"
	reasonMissingExpiry          = "token is missing the exp claim"
	reasonNotYetValid            = "token is not yet valid (nbf is in the future)"
	reasonBothBoundObjects       = "token is bound to both a validating and a mutating webhook configuration"
	reasonNoBoundObject          = "token is not bound to any webhook configuration"
	reasonMissingAllowedAPIGroup = "token is missing a single allowedAPIGroup attestation claim"
	reasonAPIGroupNotAuthorized  = "token allowedAPIGroup does not authorize the review's API group"
)

// verificationError is the internal error every failure returns. Its Error()
// yields only the generic message, so stringifying it never leaks the reason or
// any claim value; the reason is reachable solely through [Reason].
type verificationError struct {
	reason string
}

func (e *verificationError) Error() string { return ErrVerificationFailed.Error() }

// Unwrap makes errors.Is(err, ErrVerificationFailed) succeed for every failure.
func (e *verificationError) Unwrap() error { return ErrVerificationFailed }

// Fail returns a generic verification failure carrying reason for logging. The
// returned error satisfies errors.Is(err, ErrVerificationFailed) and its
// Error() is the generic message. Adapters use this to report pre-verification
// denials (for example a missing bearer token or an undecodable body) with the
// same generic surface, and the same log-only reason mechanism, as the core
// contract checks. reason MUST NOT contain claim values.
func Fail(reason string) error {
	return &verificationError{reason: reason}
}

// Reason returns the non-sensitive log string describing why verification
// failed, or "" if err is nil or is not a verification failure. The reason is
// for operator logging and metrics ONLY; callers MUST NOT branch on it.
func Reason(err error) string {
	var ve *verificationError
	if errors.As(err, &ve) {
		return ve.reason
	}
	return ""
}
