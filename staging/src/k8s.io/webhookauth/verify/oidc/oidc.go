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
// checks (issuer, audience, expiry) are performed by github.com/coreos/go-oidc
// via OpenID Connect discovery and a remote JWKS.
//
// It is the ONLY package in this module that imports go-oidc (and transitively
// JOSE); consumers who supply their own [verify.TokenAuthenticator] never pull
// it in. go-oidc owns discovery (the well-known + jwks_uri lookups, JWKS fetch,
// caching, rotation) and the signature/iss/aud/exp checks, so this package
// hardcodes no discovery URL. The layering mirrors kube-apiserver's own OIDC
// authenticator: discovery + signature + iss/aud/exp = go-oidc; the KEP-6060
// allowedAPIGroup match = the core verify package.
package oidc // import "k8s.io/webhookauth/verify/oidc"

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"

	coreosoidc "github.com/coreos/go-oidc"
	"k8s.io/webhookauth/verify"
)

// config holds the resolved options for NewRemoteVerifier.
type config struct {
	// httpClient, when non-nil, is used for both OIDC discovery and JWKS
	// fetches. go-oidc reads it from the constructor context, so it governs the
	// long-lived background key refreshes, not just the first request.
	httpClient *http.Client
}

// Option configures NewRemoteVerifier.
type Option func(*config)

// WithHTTPClient sets the *http.Client used for OIDC discovery and JWKS fetches;
// a nil client is ignored (go-oidc uses http.DefaultClient). In-cluster this is
// the seam for a client whose transport trusts the cluster CA.
func WithHTTPClient(c *http.Client) Option {
	return func(cfg *config) {
		if c != nil {
			cfg.httpClient = c
		}
	}
}

// oidcAuthenticator implements [verify.TokenAuthenticator] on top of a go-oidc
// *oidc.IDTokenVerifier. It performs the entire signature + iss/aud/exp check and
// returns the token's allowedAPIGroup values to the core policy layer.
//
// The key set is fetched at construction, but the go-oidc verifier is built
// lazily by BindAudience once the expected audience is known (out-of-cluster:
// immediately; in-cluster: from the first admission request). Until then verifier
// is nil and every token is denied. This mirrors kube-apiserver's own OIDC
// authenticator, which likewise defers verifier construction and gates readiness
// on it.
type oidcAuthenticator struct {
	// issuer is the expected token issuer, used for a cheap unverified pre-check
	// before the expensive signature/JWKS verification.
	issuer string
	// keySet is fetched at construction and reused for every verifier build.
	keySet coreosoidc.KeySet

	// verifier is built by BindAudience; nil until an audience is bound.
	verifier atomic.Pointer[coreosoidc.IDTokenVerifier]
	// mu guards audience during BindAudience; reads of verifier are lock-free.
	mu sync.Mutex
	// audience is the bound audience, retained to reject a conflicting rebind.
	audience string
}

// admissionReviewAPIGroupsClaimKey is the key, within the "kubernetes.io"
// "attestations" claim, that carries the API group(s) a webhook token is
// authorized for. It matches the server-side KEP-6060 contract
// (authentication.AttestationAdmissionReviewAPIGroups in
// k8s.io/kubernetes/pkg/apis/authentication, which staging cannot import, so the
// literal is duplicated here and kept honest by the test tripwires).
const admissionReviewAPIGroupsClaimKey = "admissionReviewAPIGroups"

// webhookPrivateClaims decodes the subset of the "kubernetes.io" private claims
// the policy needs: the admissionReviewAPIGroups attestation values.
type webhookPrivateClaims struct {
	Kubernetes struct {
		Attestations map[string][]string `json:"attestations,omitempty"`
	} `json:"kubernetes.io"`
}

// AuthenticateToken verifies rawToken and returns its allowedAPIGroup values. It
// first does a cheap unverified issuer pre-check — bailing before the expensive
// signature/JWKS work if the token's "iss" is not the expected issuer — then
// verifies via go-oidc and decodes the "kubernetes.io" claims.
//
// go-oidc owns the standard-claim verification; any error (including the issuer
// pre-check) is returned as-is and collapsed into the generic failure by the
// Verifier, so its text never reaches the caller.
func (a *oidcAuthenticator) AuthenticateToken(ctx context.Context, rawToken string) ([]string, error) {
	// The verifier is nil until an audience is bound (in-cluster: not until the
	// first request derives it). Deny fail-closed until then.
	verifier := a.verifier.Load()
	if verifier == nil {
		return nil, errors.New("oidc: no audience bound yet; verifier not ready")
	}

	// Cheap pre-check: if the token's unverified issuer is not ours, it was not
	// minted for us — fail before the expensive signature/JWKS work.
	if parsed, err := parseUnverifiedClaims(rawToken); err != nil {
		return nil, fmt.Errorf("oidc: parsing token issuer: %w", err)
	} else if parsed.Issuer != a.issuer {
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
	return claims.Kubernetes.Attestations[admissionReviewAPIGroupsClaimKey], nil
}

// BindAudience builds the go-oidc verifier for the given audience and makes the
// authenticator ready. It is idempotent: the first successful bind wins; a repeat
// bind with the same audience is a no-op, and a bind with a different audience is
// rejected so the frozen audience cannot be silently repointed.
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
	// A single audience is the go-oidc ClientID, so audience is enforced natively
	// (no SkipClientIDCheck). Store the verifier before recording the audience so
	// that once HealthCheck sees a bound audience the verifier is already visible.
	a.verifier.Store(coreosoidc.NewVerifier(a.issuer, a.keySet, &coreosoidc.Config{ClientID: audience}))
	a.audience = audience
	return nil
}

// HealthCheck reports readiness: nil once an audience has been bound and the
// verifier built, else an error. It is the readiness seam a webhook wires into a
// controller-runtime health check.
func (a *oidcAuthenticator) HealthCheck() error {
	if a.verifier.Load() == nil {
		return errors.New("oidc: audience not yet derived; verifier not ready")
	}
	return nil
}

// keySetFromDiscovery performs OIDC discovery for issuer and returns a rotating
// remote key set built from the discovery document's jwks_uri. ctx (carrying any
// injected HTTP client) governs both discovery and the key set's background
// refreshes, so pass the process-lifetime context, not a per-request one.
func keySetFromDiscovery(ctx context.Context, issuer string) (coreosoidc.KeySet, error) {
	// NewProvider fetches the well-known doc and verifies its issuer matches
	// (issuer-confusion guard).
	provider, err := coreosoidc.NewProvider(ctx, issuer)
	if err != nil {
		return nil, fmt.Errorf("oidc: OIDC discovery for issuer %q failed: %w", issuer, err)
	}
	var meta struct {
		JWKSURL string `json:"jwks_uri"`
	}
	if err := provider.Claims(&meta); err != nil {
		return nil, fmt.Errorf("oidc: reading discovery metadata for issuer %q: %w", issuer, err)
	}
	if meta.JWKSURL == "" {
		return nil, fmt.Errorf("oidc: discovery document for issuer %q has no jwks_uri", issuer)
	}
	return coreosoidc.NewRemoteKeySet(ctx, meta.JWKSURL), nil
}

// newAuthenticator fetches the key set for issuer and returns an authenticator
// whose audience is not yet bound.
func newAuthenticator(ctx context.Context, issuer string, opts ...Option) (*oidcAuthenticator, error) {
	if issuer == "" {
		return nil, errors.New("oidc: issuer must not be empty")
	}
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}
	// go-oidc reads the HTTP client from the context and uses it for discovery
	// AND the key set's background refreshes.
	if cfg.httpClient != nil {
		ctx = coreosoidc.ClientContext(ctx, cfg.httpClient)
	}
	keySet, err := keySetFromDiscovery(ctx, issuer)
	if err != nil {
		return nil, err
	}
	return &oidcAuthenticator{issuer: issuer, keySet: keySet}, nil
}

// NewRemoteVerifier returns a [verify.Verifier] that checks tokens against
// issuer's OIDC discovery document, requiring the token audience to equal
// audience. Construction performs discovery (go-oidc fetches the well-known doc,
// verifies its issuer, and builds a rotating key set from jwks_uri) and binds the
// audience immediately — the out-of-cluster path, where the audience is known up
// front.
//
// The verifier is long-lived and concurrency-safe; construct one per
// (issuer, audience). ctx governs discovery AND the key set's background fetches,
// so pass the process-lifetime context, not a per-request one.
func NewRemoteVerifier(ctx context.Context, issuer, audience string, opts ...Option) (*verify.Verifier, error) {
	if audience == "" {
		return nil, errors.New("oidc: audience must not be empty")
	}
	auth, err := newAuthenticator(ctx, issuer, opts...)
	if err != nil {
		return nil, err
	}
	if err := auth.BindAudience(audience); err != nil {
		return nil, err
	}
	return verify.NewVerifier(auth)
}

// newDeferredVerifier returns a [verify.Verifier] whose key set is fetched now but
// whose audience is bound later via [verify.Verifier.BindAudience] — the
// in-cluster path, where the audience is derived from the first admission
// request. Until an audience is bound it denies every token and reports unhealthy.
func newDeferredVerifier(ctx context.Context, issuer string, opts ...Option) (*verify.Verifier, error) {
	auth, err := newAuthenticator(ctx, issuer, opts...)
	if err != nil {
		return nil, err
	}
	return verify.NewVerifier(auth)
}
