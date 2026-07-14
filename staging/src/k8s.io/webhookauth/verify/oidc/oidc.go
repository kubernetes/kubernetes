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
// *oidc.IDTokenVerifier. It performs the entire signature + iss/aud/exp check
// and returns the token's allowedAPIGroup values to the core policy layer.
type oidcAuthenticator struct {
	verifier *coreosoidc.IDTokenVerifier
	// issuer is the expected token issuer, used for a cheap unverified pre-check
	// before the expensive signature/JWKS verification.
	issuer string
}

// allowedAPIGroupClaimKey is the fully-namespaced key under
// kubernetes.io.attestationClaims that carries the API group(s) a token is
// authorized for. Per KEP-6060 this key is namespaced; the bare "allowedAPIGroup"
// form is a known issuer bug and MUST NOT be matched.
//
// TODO(kep-6060): source this from the server-side PR once published; the final
// key will not carry the "webhook-authentication.k8s.io" prefix.
const allowedAPIGroupClaimKey = "webhook-authentication.k8s.io/allowedAPIGroup"

// webhookPrivateClaims decodes the subset of the "kubernetes.io" private claims
// the policy needs: the allowedAPIGroup attestation values.
type webhookPrivateClaims struct {
	Kubernetes struct {
		AttestationClaims map[string][]string `json:"attestationClaims,omitempty"`
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
	// Cheap pre-check: if the token's unverified issuer is not ours, it was not
	// minted for us — fail before the expensive signature/JWKS work.
	if parsed, err := parseUnverifiedClaims(rawToken); err != nil {
		return nil, fmt.Errorf("oidc: parsing token issuer: %w", err)
	} else if parsed.Issuer != a.issuer {
		return nil, fmt.Errorf("oidc: token issuer %q is not the expected issuer", parsed.Issuer)
	}

	idToken, err := a.verifier.Verify(ctx, rawToken)
	if err != nil {
		return nil, err
	}

	var claims webhookPrivateClaims
	if err := idToken.Claims(&claims); err != nil {
		return nil, fmt.Errorf("oidc: decoding token claims: %w", err)
	}
	return claims.Kubernetes.AttestationClaims[allowedAPIGroupClaimKey], nil
}

// NewRemoteVerifier returns a [verify.Verifier] that checks tokens against
// issuer's OIDC discovery document, requiring the token audience to contain
// audience. Construction performs discovery (go-oidc fetches the well-known doc,
// verifies its issuer, and builds a rotating key set from jwks_uri).
//
// The verifier is long-lived and concurrency-safe; construct one per
// (issuer, audience). ctx governs discovery AND the key set's background
// fetches, so pass the process lifetime context, not a per-request one.
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

	// go-oidc reads the HTTP client from the context and uses it for discovery
	// AND the key set's background refreshes.
	if cfg.httpClient != nil {
		ctx = coreosoidc.ClientContext(ctx, cfg.httpClient)
	}

	// Discovery: NewProvider fetches the well-known doc and verifies its issuer
	// matches (issuer-confusion guard).
	provider, err := coreosoidc.NewProvider(ctx, issuer)
	if err != nil {
		return nil, fmt.Errorf("oidc: OIDC discovery for issuer %q failed: %w", issuer, err)
	}

	// provider.Verifier enforces the algorithm allowlist, signature, issuer,
	// audience (ClientID), and expiry.
	idv := provider.Verifier(&coreosoidc.Config{ClientID: audience})

	return verify.NewVerifier(&oidcAuthenticator{verifier: idv, issuer: issuer})
}
