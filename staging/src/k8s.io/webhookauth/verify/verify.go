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

// AudienceBinder is optionally implemented by a TokenAuthenticator whose expected
// audience is not known at construction and must be supplied once at runtime (the
// in-cluster case, where the audience is derived from the first admission
// request). Until an audience is bound such an authenticator denies every token.
type AudienceBinder interface {
	// BindAudience sets the single expected audience. It is idempotent: the first
	// successful bind wins and later calls with the same audience are no-ops; a
	// call with a different audience is an error, so the audience cannot be
	// silently repointed after it is frozen.
	BindAudience(audience string) error
}

// HealthChecker is optionally implemented by a TokenAuthenticator to report
// readiness — for an AudienceBinder, whether an audience has been bound. It maps
// onto a controller-runtime health/readiness check so an authenticator that can
// never become ready (for example a scheduling race that leaves the audience
// underivable) surfaces as an unhealthy pod rather than a silent deny-all.
type HealthChecker interface {
	// HealthCheck returns nil when the authenticator is ready to verify tokens,
	// or a non-nil error describing why it is not.
	HealthCheck() error
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

// BindAudience supplies the expected audience to an authenticator that derives it
// at runtime (see [AudienceBinder]). It returns nil for an authenticator that
// already knows its audience (nothing to bind), and an error only if the backing
// authenticator supports late binding and rejects this audience (for example a
// second, conflicting bind). Callers that never use the in-cluster deferred path
// can ignore it.
func (v *Verifier) BindAudience(audience string) error {
	binder, ok := v.authenticator.(AudienceBinder)
	if !ok {
		return nil
	}
	return binder.BindAudience(audience)
}

// HealthCheck reports whether the backing authenticator is ready to verify tokens
// (see [HealthChecker]). An authenticator that does not implement HealthChecker is
// always considered ready. This is the seam a webhook wires into a
// controller-runtime health/readiness check.
func (v *Verifier) HealthCheck() error {
	checker, ok := v.authenticator.(HealthChecker)
	if !ok {
		return nil
	}
	return checker.HealthCheck()
}
