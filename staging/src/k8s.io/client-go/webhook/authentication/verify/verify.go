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
	"encoding/json"
	"errors"
	"time"
)

const (
	// AllowedAPIGroupClaimKey is the fully-namespaced key under
	// kubernetes.io.attestationClaims that carries the API group(s) a token is
	// authorized for. Per KEP-6060 this key is namespaced; the bare
	// "allowedAPIGroup" form is a known issuer bug and MUST NOT be used.
	AllowedAPIGroupClaimKey = "webhook-authentication.k8s.io/allowedAPIGroup"

	// WildcardAPIGroup is the allowedAPIGroup value that authorizes every API group.
	WildcardAPIGroup = "*"

	// KindValidatingWebhookConfiguration and KindMutatingWebhookConfiguration are
	// the values Result.BoundObjectKind takes, identifying which webhook
	// configuration the token was bound to.
	KindValidatingWebhookConfiguration = "ValidatingWebhookConfiguration"
	KindMutatingWebhookConfiguration   = "MutatingWebhookConfiguration"
)

// KeySet abstracts signature-key retrieval and JWS verification. Keeping this as
// the sole seam through which signatures are checked means the core verifier
// carries no third-party crypto dependency: it can be relocated freely, and
// tests can exercise every contract check with a trivial in-memory fake.
//
// Implementations back this with a static key set (see the josekeyset
// subpackage) or, in a follow-up, an OIDC-discovery / JWKS-fetching client.
type KeySet interface {
	// VerifySignature verifies the JWS signature of rawToken and returns the
	// decoded, signature-verified JWT payload bytes. It returns a non-nil error
	// if the token is not a well-formed JWS or no known key verifies it. The
	// returned payload MUST NOT be trusted unless err is nil.
	VerifySignature(ctx context.Context, rawToken string) (payload []byte, err error)
}

// Result is the validated identity extracted from a token that passed every
// contract check.
type Result struct {
	// BoundObjectKind is KindValidatingWebhookConfiguration or
	// KindMutatingWebhookConfiguration.
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

// Option configures a Verifier.
type Option func(*Verifier)

// WithClock overrides the time source used for exp/nbf checks. Primarily for tests.
func WithClock(now func() time.Time) Option {
	return func(v *Verifier) {
		if now != nil {
			v.now = now
		}
	}
}

// WithLeeway sets a symmetric tolerance applied to exp/nbf comparisons to absorb
// clock skew between the issuer and the verifying webhook.
func WithLeeway(d time.Duration) Option {
	return func(v *Verifier) {
		if d >= 0 {
			v.leeway = d
		}
	}
}

// Verifier performs offline verification of webhook authentication tokens.
type Verifier struct {
	keySet    KeySet
	audiences []string
	now       func() time.Time
	leeway    time.Duration
}

// NewVerifier constructs a Verifier that checks signatures via keySet and
// requires the token audience to include at least one of expectedAudiences.
//
// It returns an error for obvious misconfiguration (nil key set or empty
// audience list) so callers fail fast at startup rather than silently rejecting
// every token at request time.
func NewVerifier(keySet KeySet, expectedAudiences []string, opts ...Option) (*Verifier, error) {
	if keySet == nil {
		return nil, errors.New("verify: KeySet must not be nil")
	}
	if len(expectedAudiences) == 0 {
		return nil, errors.New("verify: at least one expected audience is required")
	}
	auds := make([]string, len(expectedAudiences))
	copy(auds, expectedAudiences)

	v := &Verifier{
		keySet:    keySet,
		audiences: auds,
		now:       time.Now,
		leeway:    0,
	}
	for _, opt := range opts {
		opt(v)
	}
	return v, nil
}

// Verify validates rawToken against the KEP-6060 token contract and, on success,
// returns the bound identity. reviewAPIGroup is the API group of the resource in
// the AdmissionReview being processed; it is matched against the token's
// allowedAPIGroup claim.
//
// On any failure Verify returns an error that satisfies
// errors.Is(err, ErrVerificationFailed); the specific typed sentinel is
// available via errors.Is for internal logging. Errors never embed claim values,
// so they are safe to surface (or, preferably, collapse to a single generic
// response) without enabling enumeration.
func (v *Verifier) Verify(ctx context.Context, rawToken string, reviewAPIGroup string) (*Result, error) {
	// 1. Signature. Any failure (bad signature or malformed JWS) collapses to a
	// single generic error so a caller cannot distinguish the two.
	payload, err := v.keySet.VerifySignature(ctx, rawToken)
	if err != nil {
		return nil, ErrInvalidSignature
	}

	var claims tokenClaims
	if err := json.Unmarshal(payload, &claims); err != nil {
		return nil, ErrInvalidToken
	}

	// 2. Audience: the token must be intended for this webhook.
	if !v.audienceMatches(claims.Audience) {
		return nil, ErrAudienceMismatch
	}

	// 6. Time validity (checked early; ordering relative to other checks is not
	// security-significant because every failure returns the same generic error).
	if err := v.checkTimeValidity(claims); err != nil {
		return nil, err
	}

	k := claims.Kubernetes
	if k == nil {
		return nil, ErrNoBoundObject
	}

	// 3. Bound-object rule: exactly one of validating / mutating.
	kind, name, uid, err := boundObject(k)
	if err != nil {
		return nil, err
	}

	// 4. allowedAPIGroup attestation claim must be present with exactly one value,
	// under the fully-namespaced key.
	groups, ok := k.AttestationClaims[AllowedAPIGroupClaimKey]
	if !ok || len(groups) != 1 {
		return nil, ErrMissingAllowedAPIGroup
	}
	allowed := groups[0]

	// 5. Wildcard matches all; otherwise it must exactly equal the review group.
	if allowed != WildcardAPIGroup && allowed != reviewAPIGroup {
		return nil, ErrAPIGroupNotAuthorized
	}

	return &Result{
		BoundObjectKind: kind,
		BoundObjectName: name,
		BoundObjectUID:  uid,
		AllowedAPIGroup: allowed,
		Audience:        []string(claims.Audience),
		Subject:         claims.Subject,
		Issuer:          claims.Issuer,
	}, nil
}

// audienceMatches reports whether the token audience includes any expected audience.
func (v *Verifier) audienceMatches(tokenAud audience) bool {
	for _, want := range v.audiences {
		if tokenAud.contains(want) {
			return true
		}
	}
	return false
}

// checkTimeValidity enforces exp and nbf with the configured leeway.
func (v *Verifier) checkTimeValidity(claims tokenClaims) error {
	now := v.now()
	if claims.Expiry != nil && now.After(claims.Expiry.time().Add(v.leeway)) {
		return ErrExpired
	}
	if claims.NotBefore != nil && now.Add(v.leeway).Before(claims.NotBefore.time()) {
		return ErrNotYetValid
	}
	return nil
}

// boundObject enforces the exactly-one bound-object rule and returns the kind,
// name, and UID of the single bound webhook configuration.
func boundObject(k *kubernetesClaims) (kind, name, uid string, err error) {
	hasValidating := k.ValidatingWebhookConfiguration != nil
	hasMutating := k.MutatingWebhookConfiguration != nil

	switch {
	case hasValidating && hasMutating:
		return "", "", "", ErrBothBoundObjects
	case !hasValidating && !hasMutating:
		return "", "", "", ErrNoBoundObject
	case hasValidating:
		ref := k.ValidatingWebhookConfiguration
		return KindValidatingWebhookConfiguration, ref.Name, ref.UID, nil
	default:
		ref := k.MutatingWebhookConfiguration
		return KindMutatingWebhookConfiguration, ref.Name, ref.UID, nil
	}
}
