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

// wildcardAPIGroup is the admissionReviewAPIGroups value that authorizes every API group.
const wildcardAPIGroup = "*"

// ErrVerificationFailed is the single generic error every verification failure
// returns. Callers check errors.Is(err, ErrVerificationFailed) (or just
// err != nil) and MUST NOT branch on any finer taxonomy: the reason is logged,
// never surfaced, so a rejection cannot be used to enumerate objects or claim
// values.
var ErrVerificationFailed = errors.New("webhook token verification failed")

// TokenAuthenticator verifies a token's signature and standard claims and returns
// the admissionReviewAPIGroups values it carries. It is the seam through which the
// core verifier gets a verified token's authorized groups without importing any
// JOSE/OIDC library: an implementation (see the oidc package) does OIDC
// discovery, signature and iss/aud/exp verification, then decodes the
// "kubernetes.io" claims and returns the admissionReviewAPIGroups list.
//
// Audience binding and readiness are part of the interface rather than optional
// runtime type assertions: the Verifier delegates BindAudience and HealthCheck
// directly, so an authenticator's audience-binding and health behavior is
// explicit at compile time instead of a silent no-op when a method is absent. An
// authenticator that knows its audience at construction (the out-of-cluster case)
// implements BindAudience as an idempotent check and HealthCheck as always-ready.
type TokenAuthenticator interface {
	// AuthenticateToken verifies rawToken (signature, issuer, audience, expiry)
	// and returns its admissionReviewAPIGroups values, or a non-nil error. The groups
	// MUST NOT be trusted unless err is nil. The error text is used only as a log
	// reason; the Verifier collapses every failure into ErrVerificationFailed.
	AuthenticateToken(ctx context.Context, rawToken string) (admissionReviewAPIGroups []string, err error)

	// BindAudience sets the single expected audience. It is idempotent: the first
	// successful bind wins and later calls with the same audience are no-ops; a
	// call with a different audience is an error, so the audience cannot be
	// silently repointed after it is frozen. An authenticator whose audience is
	// fixed at construction accepts a matching audience and rejects a conflicting
	// one; a deferred (in-cluster) authenticator denies every token until an
	// audience is bound.
	BindAudience(audience string) error

	// HealthCheck returns nil when the authenticator is ready to verify tokens, or
	// a non-nil error describing why it is not. It maps onto a controller-runtime
	// health/readiness check so a deferred authenticator that can never become
	// ready (for example a scheduling race that leaves the audience underivable)
	// surfaces as an unhealthy pod rather than a silent deny-all.
	HealthCheck() error
}

// Verifier applies the KEP-6060 policy on top of a TokenAuthenticator: signature
// and standard-claim verification are delegated to the authenticator, and this
// type adds only the admissionReviewAPIGroups match go-oidc has no concept of.
type Verifier struct {
	authenticator TokenAuthenticator
}

// NewVerifier returns a Verifier backed by authenticator.
//
// It performs no nil check: the audience-binding and health methods are now part
// of TokenAuthenticator, so the in-module constructors (oidc.NewRemoteVerifier,
// oidc.NewLocalKeySetVerifier) can only pass a concrete authenticator, and an
// external caller passing an explicit nil is a programming error caught at the
// first call. The (*Verifier, error) signature is retained for API stability
// across the staging module boundary and so a future validation can be added
// without a breaking change; the error is currently always nil.
func NewVerifier(authenticator TokenAuthenticator) (*Verifier, error) {
	return &Verifier{authenticator: authenticator}, nil
}

// Verify authenticates rawToken and applies the KEP-6060 policy: the token's
// admissionReviewAPIGroups must authorize reviewAPIGroup (an exact match or "*"). It
// returns nil on success or ErrVerificationFailed on any failure; the reason is
// logged via klog.FromContext(ctx), never returned (anti-enumeration).
func (v *Verifier) Verify(ctx context.Context, rawToken string, reviewAPIGroup string) error {
	logger := klog.FromContext(ctx)

	// The authenticator owns what go-oidc verifies (signature, iss/aud/exp) and
	// returns the admissionReviewAPIGroups values; its error text is log-only.
	groups, err := v.authenticator.AuthenticateToken(ctx, rawToken)
	if err != nil {
		logger.V(2).Info("Webhook token verification denied",
			"reason", "token authentication failed",
			"detail", err.Error())
		return ErrVerificationFailed
	}

	// The webhook only checks membership: the admissionReviewAPIGroups list must contain
	// the review's group or "*" (an empty list authorizes nothing).
	if !slices.Contains(groups, reviewAPIGroup) && !slices.Contains(groups, wildcardAPIGroup) {
		logger.V(2).Info("Webhook token verification denied",
			"reason", "token admissionReviewAPIGroups does not authorize the review's API group")
		return ErrVerificationFailed
	}

	return nil
}

// BindAudience supplies the expected audience to the backing authenticator. For
// an authenticator that derives its audience at runtime (the in-cluster deferred
// path) this is the one-time bind; for one whose audience is fixed at
// construction it is an idempotent check that rejects a conflicting audience. The
// call is delegated directly — there is no optional-interface type assertion —
// so the behavior is explicit rather than a silent no-op.
func (v *Verifier) BindAudience(audience string) error {
	return v.authenticator.BindAudience(audience)
}

// HealthCheck reports whether the backing authenticator is ready to verify
// tokens. The call is delegated directly to the authenticator. This is the seam a
// webhook wires into a controller-runtime health/readiness check.
func (v *Verifier) HealthCheck() error {
	return v.authenticator.HealthCheck()
}
