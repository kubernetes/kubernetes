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

// Package oidckeyset provides a [verify.KeySet] backed by OpenID Connect
// discovery and a remote JWKS, implemented on top of github.com/coreos/go-oidc.
//
// It is the ONLY package in this module that imports go-oidc (and, transitively,
// JOSE): the core verify package stays pure-stdlib, and consumers who supply
// their own KeySet never pull in go-oidc. go-oidc owns the OIDC discovery flow
// (the /.well-known/openid-configuration path, the jwks_uri lookup, JWKS fetch,
// caching and key rotation), so this package hardcodes no discovery URL —
// resolving the long-standing "OIDC discovery is at the wrong layer, reuse
// existing code" review item.
//
// The canonical precedent for reusing go-oidc this way is kube-apiserver's own
// OIDC authenticator, which likewise takes an oidc.KeySet and layers its claim
// policy on top. Here the layering is: signature/keyset = go-oidc; the KEP-6060
// claim policy (issuer/audience/exp/bound-object/allowedAPIGroup) = the core
// verify package.
package oidckeyset // import "k8s.io/webhook-auth/verify/oidckeyset"

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	oidc "github.com/coreos/go-oidc"
	"k8s.io/webhook-auth/verify"
)

// config holds the resolved options for NewRemoteKeySet.
type config struct {
	// httpClient, when non-nil, is used for both OIDC discovery and JWKS
	// fetches. go-oidc reads it from the constructor context, so it governs the
	// long-lived background key refreshes, not just the first request.
	httpClient *http.Client

	// insecureJWKSURL, when non-empty, bypasses OIDC discovery entirely and
	// fetches keys directly from this URL. This skips go-oidc's
	// issuer-in-discovery match (NewProvider verifies that the discovery
	// document's "issuer" equals the configured issuer). It exists ONLY so tests
	// can point a key set at an httptest JWKS endpoint without standing up a
	// self-consistent discovery document; it MUST NOT be used in production,
	// where the issuer<->discovery binding is a security check.
	insecureJWKSURL string
}

// Option configures NewRemoteKeySet.
type Option func(*config)

// WithHTTPClient sets the *http.Client used for OIDC discovery and JWKS fetches.
// A nil client is ignored and go-oidc's default (http.DefaultClient) is used.
//
// In-cluster this is the seam for injecting a client whose transport trusts the
// cluster CA (for example the apiserver serving-CA bundle mounted into the pod)
// so discovery and JWKS retrieval succeed over the in-cluster HTTPS endpoint.
func WithHTTPClient(c *http.Client) Option {
	return func(cfg *config) {
		if c != nil {
			cfg.httpClient = c
		}
	}
}

// WithInsecureSkipDiscovery bypasses OIDC discovery and its issuer-match check,
// fetching keys directly from jwksURL. TEST-ONLY: it removes the binding between
// the configured issuer and the discovery document, so it must never be used
// outside tests. A non-nil error is still returned for an empty jwksURL.
func WithInsecureSkipDiscovery(jwksURL string) Option {
	return func(cfg *config) {
		cfg.insecureJWKSURL = jwksURL
	}
}

// keySet adapts an oidc.KeySet to the core verify.KeySet. The two interfaces
// have identical method signatures, so this is a thin, named pass-through that
// keeps go-oidc's concrete type out of the returned API.
type keySet struct {
	inner oidc.KeySet
}

// VerifySignature implements verify.KeySet by delegating to go-oidc, which
// parses the JWS, selects the signing key by "kid" from the cached/rotated JWKS
// (refreshing on a miss), verifies the signature, and returns the raw payload
// bytes. It intentionally performs no claim validation — issuer, audience,
// expiry, bound-object and allowedAPIGroup are all enforced by the core Verifier.
func (k *keySet) VerifySignature(ctx context.Context, rawToken string) (payload []byte, err error) {
	return k.inner.VerifySignature(ctx, rawToken)
}

// NewRemoteKeySet returns a verify.KeySet that verifies token signatures against
// the JWKS advertised by issuer's OIDC discovery document.
//
// Construction performs OIDC discovery: go-oidc fetches
// {issuer}/.well-known/openid-configuration, verifies that the document's
// "issuer" field equals issuer (an issuer-confusion guard), and reads its
// "jwks_uri". The returned key set then fetches and caches keys from that
// jwks_uri, refreshing on signature-key misses to follow rotation. Because
// go-oidc owns these paths, no discovery or JWKS URL is hardcoded here.
//
// The returned key set is long-lived and safe for concurrent use; construct one
// per issuer and share it. The provided ctx governs discovery AND the key set's
// subsequent background fetches, so it should be the lifetime context of the
// verifying process, not a per-request context.
//
// The returned KeySet only checks signatures. The token's issuer must still be
// validated against a trusted value by verify.NewVerifier (pass the SAME issuer
// used here), so the keys that verified the signature and the issuer asserted in
// the token cannot disagree.
func NewRemoteKeySet(ctx context.Context, issuer string, opts ...Option) (verify.KeySet, error) {
	if issuer == "" {
		return nil, errors.New("oidckeyset: issuer must not be empty")
	}

	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	// Thread a custom client through go-oidc via the context it uses for all
	// HTTP. go-oidc's remote key set fetches keys using the constructor context,
	// so the client set here also governs background key refreshes.
	if cfg.httpClient != nil {
		ctx = oidc.ClientContext(ctx, cfg.httpClient)
	}

	// Test-only shortcut: skip discovery (and its issuer match) and go straight
	// to a remote JWKS. See WithInsecureSkipDiscovery.
	if cfg.insecureJWKSURL != "" {
		return &keySet{inner: oidc.NewRemoteKeySet(ctx, cfg.insecureJWKSURL)}, nil
	}

	// Discovery. NewProvider fetches the well-known document and verifies that
	// its "issuer" matches the configured issuer; a mismatch (issuer confusion)
	// fails here.
	provider, err := oidc.NewProvider(ctx, issuer)
	if err != nil {
		return nil, fmt.Errorf("oidckeyset: OIDC discovery for issuer %q failed: %w", issuer, err)
	}

	// Read jwks_uri from the discovery document go-oidc already fetched, rather
	// than assuming any path. The provider derives jwks_uri from the document
	// itself; we extract it to build an explicitly shareable remote key set.
	var meta struct {
		JWKSURL string `json:"jwks_uri"`
	}
	if err := provider.Claims(&meta); err != nil {
		return nil, fmt.Errorf("oidckeyset: reading discovery document for issuer %q failed: %w", issuer, err)
	}
	if meta.JWKSURL == "" {
		return nil, fmt.Errorf("oidckeyset: discovery document for issuer %q has no jwks_uri", issuer)
	}

	return &keySet{inner: oidc.NewRemoteKeySet(ctx, meta.JWKSURL)}, nil
}
