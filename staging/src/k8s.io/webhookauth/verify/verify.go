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

// ErrVerificationFailed is the single generic error every verification failure
// returns. Callers check errors.Is(err, ErrVerificationFailed) (or just
// err != nil) and MUST NOT branch on any finer taxonomy: the reason is logged,
// never surfaced, so a rejection cannot be used to enumerate objects or claim
// values.
var ErrVerificationFailed = errors.New("webhook token verification failed")

// TokenAuthenticator verifies a token's signature and standard claims and returns
// the allowedAPIGroup values it carries. It is the seam through which the core
// verifier gets a verified token's authorized groups without importing any
// JOSE/OIDC library: an implementation (see the oidc package) does OIDC
// discovery, signature and iss/aud/exp verification, then decodes the
// "kubernetes.io" claims and returns the allowedAPIGroup list.
type TokenAuthenticator interface {
	// AuthenticateToken verifies rawToken (signature, issuer, audience, expiry)
	// and returns its allowedAPIGroup values, or a non-nil error. The groups MUST
	// NOT be trusted unless err is nil. The error text is used only as a log
	// reason; the Verifier collapses every failure into ErrVerificationFailed.
	AuthenticateToken(ctx context.Context, rawToken string) (allowedAPIGroups []string, err error)
}

// Verifier applies the KEP-6060 policy on top of a TokenAuthenticator: signature
// and standard-claim verification are delegated to the authenticator, and this
// type adds only the allowedAPIGroup match go-oidc has no concept of.
type Verifier struct {
	authenticator TokenAuthenticator
}

// NewVerifier returns a Verifier backed by authenticator, or an error if
// authenticator is nil so callers fail fast at startup.
func NewVerifier(authenticator TokenAuthenticator) (*Verifier, error) {
	if authenticator == nil {
		return nil, errors.New("verify: TokenAuthenticator must not be nil")
	}
	return &Verifier{authenticator: authenticator}, nil
}

// Verify authenticates rawToken and applies the KEP-6060 policy: the token's
// allowedAPIGroup must authorize reviewAPIGroup (an exact match or "*"). It
// returns nil on success or ErrVerificationFailed on any failure; the reason is
// logged via klog.FromContext(ctx), never returned (anti-enumeration).
func (v *Verifier) Verify(ctx context.Context, rawToken string, reviewAPIGroup string) error {
	logger := klog.FromContext(ctx)

	// The authenticator owns what go-oidc verifies (signature, iss/aud/exp) and
	// returns the allowedAPIGroup values; its error text is log-only.
	groups, err := v.authenticator.AuthenticateToken(ctx, rawToken)
	if err != nil {
		logger.V(2).Info("Webhook token verification denied",
			"reason", "token authentication failed",
			"detail", err.Error())
		return ErrVerificationFailed
	}

	// The webhook only checks membership: the allowedAPIGroup list must contain
	// the review's group or "*" (an empty list authorizes nothing).
	if !slices.Contains(groups, reviewAPIGroup) && !slices.Contains(groups, wildcardAPIGroup) {
		logger.V(2).Info("Webhook token verification denied",
			"reason", "token allowedAPIGroup does not authorize the review's API group")
		return ErrVerificationFailed
	}

	return nil
}
