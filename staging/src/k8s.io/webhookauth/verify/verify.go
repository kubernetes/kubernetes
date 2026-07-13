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
	"slices"

	"k8s.io/klog/v2"
)

// wildcardAPIGroup is the allowedAPIGroup value that authorizes every API group.
const wildcardAPIGroup = "*"

// ErrVerificationFailed is the single, generic error every verification failure
// returns. Callers check errors.Is(err, ErrVerificationFailed) (or simply
// err != nil) and MUST NOT branch on any finer taxonomy: the specific reason is
// logged inline via the context logger for operators and is never surfaced to
// the caller, so a rejection cannot be used to enumerate objects or probe claim
// values.
var ErrVerificationFailed = errors.New("webhook token verification failed")

// TokenAuthenticator verifies a token's signature and standard claims and
// returns the KEP-6060 allowedAPIGroup values it carries. It is the seam through
// which the core verifier obtains a verified token's authorized API groups
// without importing any JOSE/OIDC library: an implementation (see the oidc
// package, which uses go-oidc) performs OIDC discovery, signature verification,
// and the iss/aud/exp checks, then decodes the "kubernetes.io" private claims
// and returns just the allowedAPIGroup list for the policy layer here to match.
type TokenAuthenticator interface {
	// AuthenticateToken verifies rawToken's signature and standard claims
	// (issuer, audience, expiry) and returns the token's allowedAPIGroup values.
	// It returns a non-nil error if the token is not well-formed, not signed by a
	// trusted key, or fails any standard-claim check; the returned groups MUST
	// NOT be trusted unless err is nil.
	//
	// The returned error's text is used ONLY as a non-sensitive log reason: the
	// Verifier collapses every authentication failure into the single generic
	// ErrVerificationFailed, so a caller can never learn which check failed.
	AuthenticateToken(ctx context.Context, rawToken string) (allowedAPIGroups []string, err error)
}

// Verifier applies the KEP-6060 policy on top of a TokenAuthenticator. Signature
// and standard-claim verification are delegated to the authenticator; this type
// carries only the check go-oidc has no concept of: the namespaced
// allowedAPIGroup match.
type Verifier struct {
	authenticator TokenAuthenticator
}

// NewVerifier constructs a Verifier that delegates signature and standard-claim
// verification to authenticator and layers the KEP-6060 policy on top. It
// returns an error if authenticator is nil so callers fail fast at startup.
func NewVerifier(authenticator TokenAuthenticator) (*Verifier, error) {
	if authenticator == nil {
		return nil, errors.New("verify: TokenAuthenticator must not be nil")
	}
	return &Verifier{authenticator: authenticator}, nil
}

// Verify authenticates rawToken and applies the KEP-6060 policy: the token's
// allowedAPIGroup attestation claim must authorize reviewAPIGroup (an exact
// match or the "*" wildcard). reviewAPIGroup is the API group of the resource in
// the AdmissionReview being processed.
//
// It returns nil on success or ErrVerificationFailed on any failure. The
// specific reason is logged via klog.FromContext(ctx) for operators and is never
// returned: callers get one generic error and cannot branch on why verification
// failed (anti-enumeration).
func (v *Verifier) Verify(ctx context.Context, rawToken string, reviewAPIGroup string) error {
	logger := klog.FromContext(ctx)

	// The authenticator owns everything go-oidc verifies (signature, iss/aud/exp)
	// and returns just the token's allowedAPIGroup values. Its descriptive error
	// text is logged for operators only, never returned.
	groups, err := v.authenticator.AuthenticateToken(ctx, rawToken)
	if err != nil {
		logger.V(2).Info("Webhook token verification denied",
			"reason", "token authentication failed",
			"detail", err.Error())
		return ErrVerificationFailed
	}

	// The webhook cares only about the API group: the token's allowedAPIGroup
	// list must contain either the review's group or the "*" wildcard. The API
	// server decides how many groups the list holds; the webhook only checks
	// membership (an empty list authorizes nothing).
	if !slices.Contains(groups, reviewAPIGroup) && !slices.Contains(groups, wildcardAPIGroup) {
		logger.V(2).Info("Webhook token verification denied",
			"reason", "token allowedAPIGroup does not authorize the review's API group")
		return ErrVerificationFailed
	}

	return nil
}
