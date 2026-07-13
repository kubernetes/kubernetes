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
)

const (
	// allowedAPIGroupClaimKey is the fully-namespaced key under
	// kubernetes.io.attestationClaims that carries the API group(s) a token is
	// authorized for. Per KEP-6060 this key is namespaced; the bare
	// "allowedAPIGroup" form is a known issuer bug and MUST NOT be used.
	allowedAPIGroupClaimKey = "webhook-authentication.k8s.io/allowedAPIGroup"

	// wildcardAPIGroup is the allowedAPIGroup value that authorizes every API group.
	wildcardAPIGroup = "*"

	// kindValidatingWebhookConfiguration and kindMutatingWebhookConfiguration are
	// the values Result.BoundObjectKind takes, identifying which webhook
	// configuration the token was bound to.
	kindValidatingWebhookConfiguration = "ValidatingWebhookConfiguration"
	kindMutatingWebhookConfiguration   = "MutatingWebhookConfiguration"
)

// TokenAuthenticator verifies a token's signature and standard claims and
// returns the decoded claims for policy evaluation. It is the seam through
// which the core verifier obtains a signature- and standard-claim-verified
// token without importing any JOSE/OIDC library: an implementation (see the
// oidc package, which uses go-oidc) performs OIDC discovery, signature
// verification, and the iss/aud/exp checks, then hands back a VerifiedClaims for
// the pure-stdlib policy layer here to finish.
type TokenAuthenticator interface {
	// AuthenticateToken verifies rawToken's signature and standard claims
	// (issuer, audience, expiry) and returns the decoded claims. It returns a
	// non-nil error if the token is not well-formed, not signed by a trusted
	// key, or fails any standard-claim check; the returned claims MUST NOT be
	// trusted unless err is nil.
	//
	// The returned error's text is used ONLY as a non-sensitive log reason: the
	// Verifier collapses every authentication failure into the single generic
	// ErrVerificationFailed, so a caller can never learn which check failed.
	//
	// The 2-value (claims, error) shape is intentional and does NOT mirror the
	// k8s authenticator.Token (response, ok, error) triple: any failure — including
	// an unauthenticated token — is a non-nil error, and callers must not branch
	// on an ok-versus-error distinction, which anti-enumeration deliberately
	// denies them.
	AuthenticateToken(ctx context.Context, rawToken string) (*VerifiedClaims, error)
}

// Result is the validated identity extracted from a token that passed every
// contract check.
type Result struct {
	// BoundObjectKind is "ValidatingWebhookConfiguration" or
	// "MutatingWebhookConfiguration".
	BoundObjectKind string
	// BoundObjectName and BoundObjectUID identify the webhook configuration the
	// token is bound to.
	BoundObjectName string
	BoundObjectUID  string
	// AllowedAPIGroup is the single value from the allowedAPIGroup attestation
	// claim ("*" for all groups).
	AllowedAPIGroup string
	// Audience is the token's audience list.
	Audience []string
	// Subject is the token's "sub" claim (typically the service account identity).
	Subject string
	// Issuer is the token's "iss" claim.
	Issuer string
}

// Verifier applies the KEP-6060 policy on top of a TokenAuthenticator. Signature
// and standard-claim verification are delegated to the authenticator; this type
// carries only the checks go-oidc has no concept of (the bound-object exactly-one
// rule and the namespaced allowedAPIGroup match) and builds the Result.
type Verifier struct {
	authenticator TokenAuthenticator
}

// NewVerifier constructs a Verifier that delegates signature and standard-claim
// verification to authenticator and layers the KEP-6060 policy on top. It
// returns an error if authenticator is nil so callers fail fast at startup.
func NewVerifier(authenticator TokenAuthenticator) (*Verifier, error) {
	if authenticator == nil {
		return nil, errNilAuthenticator
	}
	return &Verifier{authenticator: authenticator}, nil
}

// Verify authenticates rawToken and, on success, applies the KEP-6060 policy and
// returns the bound identity. reviewAPIGroup is the API group of the resource in
// the AdmissionReview being processed; it is matched against the token's
// allowedAPIGroup claim.
//
// On any failure Verify returns a single generic error that satisfies
// errors.Is(err, ErrVerificationFailed) and nothing else: there is no per-check
// error taxonomy for callers to branch on. The specific reason is available only
// as a non-sensitive log string via Reason(err). Authentication failures
// surfaced by the authenticator (for example go-oidc's descriptive "expired
// token" or "audience mismatch" errors) are likewise collapsed into the generic
// failure so rejections cannot be used to enumerate objects or probe claim
// values; the descriptive text is retained only for logging.
func (v *Verifier) Verify(ctx context.Context, rawToken string, reviewAPIGroup string) (*Result, error) {
	// 1. Signature + standard claims (iss/aud/exp) are the authenticator's job.
	// Any failure collapses to the single generic error; the authenticator's
	// descriptive text is kept only as a log-only reason, never returned.
	claims, err := v.authenticator.AuthenticateToken(ctx, rawToken)
	if err != nil {
		return nil, Fail(authenticationFailedReason(err))
	}

	k := &claims.Kubernetes

	// 2. Bound-object rule: exactly one of validating / mutating.
	kind, name, uid, err := boundObject(k)
	if err != nil {
		return nil, err
	}

	// 3. allowedAPIGroup attestation claim must be present with exactly one
	// value, under the fully-namespaced key.
	groups, ok := k.AttestationClaims[allowedAPIGroupClaimKey]
	if !ok || len(groups) != 1 {
		return nil, Fail(reasonMissingAllowedAPIGroup)
	}
	allowed := groups[0]

	// 4. Wildcard matches all; otherwise it must exactly equal the review group.
	if allowed != wildcardAPIGroup && allowed != reviewAPIGroup {
		return nil, Fail(reasonAPIGroupNotAuthorized)
	}

	return &Result{
		BoundObjectKind: kind,
		BoundObjectName: name,
		BoundObjectUID:  uid,
		AllowedAPIGroup: allowed,
		Audience:        claims.Audience,
		Subject:         claims.Subject,
		Issuer:          claims.Issuer,
	}, nil
}

// boundObject enforces the exactly-one bound-object rule and returns the kind,
// name, and UID of the single bound webhook configuration.
func boundObject(k *kubernetesClaims) (kind, name, uid string, err error) {
	hasValidating := k.ValidatingWebhookConfiguration != nil
	hasMutating := k.MutatingWebhookConfiguration != nil

	switch {
	case hasValidating && hasMutating:
		return "", "", "", Fail(reasonBothBoundObjects)
	case !hasValidating && !hasMutating:
		return "", "", "", Fail(reasonNoBoundObject)
	case hasValidating:
		ref := k.ValidatingWebhookConfiguration
		return kindValidatingWebhookConfiguration, ref.Name, ref.UID, nil
	default:
		ref := k.MutatingWebhookConfiguration
		return kindMutatingWebhookConfiguration, ref.Name, ref.UID, nil
	}
}
