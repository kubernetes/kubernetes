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
)

// ErrVerificationFailed is the generic, anti-enumeration sentinel that every
// verification failure satisfies (errors.Is(err, ErrVerificationFailed) == true).
//
// Callers surfacing an error to an external client SHOULD return only this
// message (or a fixed 401/403) and log the specific typed sentinel internally.
// None of the sentinels below interpolate claim values (webhook name, UID,
// subject, or API group), so leaking their raw strings still does not reveal
// whether a specific webhook, group, or object exists.
var ErrVerificationFailed = errors.New("webhook token verification failed")

// Typed sentinels. Each wraps ErrVerificationFailed, so errors.Is against either
// the specific sentinel or the generic one works. Note that ErrBothBoundObjects
// and ErrNoBoundObject deliberately share the same human-readable category
// ("invalid webhook binding") so the external string does not reveal which side
// of the bound-object rule was violated; they remain distinct values for
// programmatic branching.
var (
	// ErrInvalidSignature indicates the JWS signature did not verify against any
	// known issuer key. It also stands in for malformed compact-serialized JWTs
	// so a caller cannot distinguish "bad signature" from "not a JWT".
	ErrInvalidSignature = fmt.Errorf("%w: invalid signature", ErrVerificationFailed)

	// ErrInvalidToken indicates the verified payload was not well-formed JSON.
	ErrInvalidToken = fmt.Errorf("%w: malformed token", ErrVerificationFailed)

	// ErrAudienceMismatch indicates the token audience did not include any of the
	// verifier's expected audiences.
	ErrAudienceMismatch = fmt.Errorf("%w: audience mismatch", ErrVerificationFailed)

	// ErrIssuerMismatch indicates the token "iss" claim was absent, or did not
	// equal the verifier's expected issuer. RFC 7519 / OIDC require the issuer to
	// be validated; a token from an unexpected (or unstated) issuer is rejected.
	ErrIssuerMismatch = fmt.Errorf("%w: issuer mismatch", ErrVerificationFailed)

	// ErrExpired indicates the token is past its exp instant (accounting for leeway).
	ErrExpired = fmt.Errorf("%w: token expired", ErrVerificationFailed)

	// ErrMissingExpiry indicates the token carried no "exp" claim. A token without
	// an expiry would never expire, so its absence is rejected rather than treated
	// as "valid forever".
	ErrMissingExpiry = fmt.Errorf("%w: token missing expiry", ErrVerificationFailed)

	// ErrNotYetValid indicates the token is before its nbf instant (accounting for leeway).
	ErrNotYetValid = fmt.Errorf("%w: token not yet valid", ErrVerificationFailed)

	// ErrBothBoundObjects indicates both validating- and mutating-webhook refs
	// were present, violating the exactly-one bound-object rule.
	ErrBothBoundObjects = fmt.Errorf("%w: invalid webhook binding", ErrVerificationFailed)

	// ErrNoBoundObject indicates neither webhook ref was present, violating the
	// exactly-one bound-object rule.
	ErrNoBoundObject = fmt.Errorf("%w: invalid webhook binding", ErrVerificationFailed)

	// ErrMissingAllowedAPIGroup indicates the allowedAPIGroup attestation claim
	// was absent, or did not carry exactly one value.
	ErrMissingAllowedAPIGroup = fmt.Errorf("%w: invalid attestation claim", ErrVerificationFailed)

	// ErrAPIGroupNotAuthorized indicates the allowedAPIGroup claim neither
	// wildcarded nor matched the AdmissionReview resource's API group.
	ErrAPIGroupNotAuthorized = fmt.Errorf("%w: not authorized", ErrVerificationFailed)
)
