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

// Package oidc builds a [verify.Verifier] whose signature and standard-claim
// checks (issuer, audience, expiry) are delegated to github.com/coreos/go-oidc.
//
// It is the only package in this module that imports go-oidc; consumers who
// supply their own [verify.TokenAuthenticator] never pull it in.
package oidc // import "k8s.io/webhookauth/internal/oidc"

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"

	coreosoidc "github.com/coreos/go-oidc"
	"k8s.io/webhookauth/internal/verify"
)

// config holds the resolved options for the verifier constructors.
type config struct {
	// httpClient, when non-nil, is used for discovery and JWKS fetches (including
	// go-oidc's background key refreshes).
	httpClient *http.Client
}

// Option configures the verifier constructors.
type Option func(*config)

// WithHTTPClient sets the [http.Client] used for discovery and JWKS fetches; a
// nil client is ignored. In-cluster this supplies a transport that trusts the
// cluster CA.
func WithHTTPClient(c *http.Client) Option {
	return func(cfg *config) {
		if c != nil {
			cfg.httpClient = c
		}
	}
}

// oidcAuthenticator implements [verify.TokenAuthenticator] using go-oidc, with
// the verifier built lazily by BindAudience (fail-closed until an audience is
// bound).
type oidcAuthenticator struct {
	issuer string
	keySet coreosoidc.KeySet

	// verifier is built lazily by BindAudience; nil until an audience is bound.
	verifier atomic.Pointer[coreosoidc.IDTokenVerifier]
	// mu guards BindAudience; verifier reads are lock-free.
	mu sync.Mutex
	// audience is retained to reject a conflicting rebind.
	audience string
}

// admissionReviewAPIGroupsClaimKey is the "attestations" claim key carrying the
// API groups a webhook token is authorized for. The literal duplicates the
// server-side constant (staging cannot import it) and is guarded by test
// tripwires.
const admissionReviewAPIGroupsClaimKey = "admissionReviewAPIGroups"

// webhookPrivateClaims decodes the "kubernetes.io" attestation claims the policy
// needs.
type webhookPrivateClaims struct {
	Kubernetes struct {
		Attestations map[string][]string `json:"attestations,omitempty"`
	} `json:"kubernetes.io"`
}

// AuthenticateToken verifies rawToken via go-oidc and returns its
// admissionReviewAPIGroups values.
func (a *oidcAuthenticator) AuthenticateToken(ctx context.Context, rawToken string) ([]string, error) {
	verifier := a.verifier.Load()
	if verifier == nil {
		return nil, errors.New("oidc: no audience bound yet; verifier not ready")
	}
	return verifyTokenGroups(ctx, verifier, a.issuer, rawToken)
}

// verifyTokenGroups verifies rawToken with the supplied go-oidc verifier
// (signature, audience, expiry) and returns its admissionReviewAPIGroups values.
// Shared by the deferred (in-cluster) and eagerly-bound (out-of-cluster)
// authenticators so both decode the KEP-6060 attestation claims identically.
func verifyTokenGroups(ctx context.Context, verifier *coreosoidc.IDTokenVerifier, issuer, rawToken string) ([]string, error) {
	// Cheap unverified issuer pre-check before the expensive signature work.
	if parsed, err := parseUnverifiedClaims(rawToken); err != nil {
		return nil, fmt.Errorf("oidc: parsing token issuer: %w", err)
	} else if parsed.Issuer != issuer {
		return nil, fmt.Errorf("oidc: token issuer %q is not the expected issuer", parsed.Issuer)
	}

	idToken, err := verifier.Verify(ctx, rawToken)
	if err != nil {
		return nil, err
	}

	var claims webhookPrivateClaims
	if err := idToken.Claims(&claims); err != nil {
		return nil, fmt.Errorf("oidc: decoding token claims: %w", err)
	}
	// A verified token must carry at least one authorized API group; an empty
	// attestation authorizes nothing and is treated as a malformed token rather
	// than a silent deny-all.
	groups := claims.Kubernetes.Attestations[admissionReviewAPIGroupsClaimKey]
	if len(groups) == 0 {
		return nil, fmt.Errorf("oidc: token carries no %q attestation", admissionReviewAPIGroupsClaimKey)
	}
	return groups, nil
}

// BindAudience builds the go-oidc verifier for audience and makes the
// authenticator ready. It is idempotent: the first bind wins, a matching rebind
// is a no-op, and a conflicting rebind is rejected.
func (a *oidcAuthenticator) BindAudience(audience string) error {
	if audience == "" {
		return errors.New("oidc: audience must not be empty")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.audience != "" {
		if a.audience != audience {
			return fmt.Errorf("oidc: audience already bound to %q, refusing to rebind to %q", a.audience, audience)
		}
		return nil
	}
	// The audience is the go-oidc ClientID (enforced natively). Store the verifier
	// before recording the audience so HealthCheck never sees a bound audience
	// without a visible verifier.
	a.verifier.Store(coreosoidc.NewVerifier(a.issuer, a.keySet, &coreosoidc.Config{ClientID: audience}))
	a.audience = audience
	return nil
}

// HealthCheck reports readiness: nil once an audience is bound and the verifier
// built, else an error.
func (a *oidcAuthenticator) HealthCheck() error {
	if a.verifier.Load() == nil {
		return errors.New("oidc: audience not yet derived; verifier not ready")
	}
	return nil
}

// boundAuthenticator implements [verify.TokenAuthenticator] for the
// out-of-cluster path: the audience is fixed at construction and the go-oidc
// verifier is built eagerly, so it is ready immediately and never fail-closes
// for a missing audience.
type boundAuthenticator struct {
	issuer   string
	audience string
	verifier *coreosoidc.IDTokenVerifier
}

// AuthenticateToken verifies rawToken via the pre-built go-oidc verifier and
// returns its admissionReviewAPIGroups values.
func (b *boundAuthenticator) AuthenticateToken(ctx context.Context, rawToken string) ([]string, error) {
	return verifyTokenGroups(ctx, b.verifier, b.issuer, rawToken)
}

// BindAudience is an idempotent check: the audience is fixed at construction, so
// a matching audience is a no-op and a conflicting one is rejected. It never
// rebuilds the verifier.
func (b *boundAuthenticator) BindAudience(audience string) error {
	if audience == "" {
		return errors.New("oidc: audience must not be empty")
	}
	if audience != b.audience {
		return fmt.Errorf("oidc: audience already bound to %q, refusing to rebind to %q", b.audience, audience)
	}
	return nil
}

// HealthCheck always reports ready: the verifier is built at construction.
func (b *boundAuthenticator) HealthCheck() error { return nil }

// NewRemoteVerifier returns a [verify.Verifier] for the out-of-cluster path: it
// checks tokens against issuer's OIDC discovery document and requires the token
// audience to equal audience, enforced natively as the go-oidc ClientID. The
// resulting verifier is long-lived and concurrency-safe.
//
// ctx governs discovery and the key set's background refreshes, so pass a
// process-lifetime context.
func NewRemoteVerifier(ctx context.Context, issuer, audience string, opts ...Option) (*verify.Verifier, error) {
	if issuer == "" {
		return nil, errors.New("oidc: issuer must not be empty")
	}
	if audience == "" {
		return nil, errors.New("oidc: audience must not be empty")
	}
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}
	// go-oidc reads the HTTP client from the context for discovery and background
	// key refreshes.
	if cfg.httpClient != nil {
		ctx = coreosoidc.ClientContext(ctx, cfg.httpClient)
	}
	// NewProvider performs discovery and verifies the discovered issuer matches
	// (issuer-confusion guard).
	provider, err := coreosoidc.NewProvider(ctx, issuer)
	if err != nil {
		return nil, fmt.Errorf("oidc: OIDC discovery for issuer %q failed: %w", issuer, err)
	}
	// The audience is the go-oidc ClientID (enforced natively).
	verifier := provider.Verifier(&coreosoidc.Config{ClientID: audience})
	return verify.NewVerifier(&boundAuthenticator{issuer: issuer, audience: audience, verifier: verifier})
}
