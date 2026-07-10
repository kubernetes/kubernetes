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

// Package oidc builds a [verify.Verifier] whose signature and standard
// claim checks (issuer, audience, expiry) are performed by
// github.com/coreos/go-oidc via OpenID Connect discovery and a remote JWKS.
//
// It is the ONLY package in this module that imports go-oidc (and, transitively,
// JOSE): the core verify package stays pure-stdlib, and consumers who supply
// their own [verify.TokenAuthenticator] never pull in go-oidc. go-oidc owns the
// OIDC discovery flow (the /.well-known/openid-configuration path, the jwks_uri
// lookup, JWKS fetch, caching and key rotation) and the signature / issuer /
// audience / expiry verification, so this package hardcodes no discovery URL and
// re-implements none of that machinery.
//
// The canonical precedent for reusing go-oidc this way is kube-apiserver's own
// OIDC authenticator, which likewise layers its claim policy on top of a go-oidc
// verifier. Here the layering is: discovery + signature + iss/aud/exp = go-oidc;
// the KEP-6060 policy (bound-object exactly-one, allowedAPIGroup match) = the
// core verify package.
package oidc // import "k8s.io/webhook-auth/verify/oidc"

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	coreosoidc "github.com/coreos/go-oidc"
	"k8s.io/webhook-auth/verify"
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

// oidcAuthenticator implements [verify.TokenAuthenticator] on top of a go-oidc
// *oidc.IDTokenVerifier. It performs the entire signature + iss/aud/exp check
// and hands the decoded claims back to the core policy layer.
type oidcAuthenticator struct {
	verifier *coreosoidc.IDTokenVerifier
}

// AuthenticateToken verifies rawToken via go-oidc — parsing the JWS, enforcing
// the signing-algorithm allowlist, verifying the signature against the
// discovered/rotated JWKS, and checking issuer, audience and expiry — and then
// decodes the webhook private claims.
//
// The standard claims returned in VerifiedClaims are taken from the values
// go-oidc validated (idToken.Issuer/Subject/Audience), NOT re-read from the raw
// payload, so a token cannot smuggle a different issuer/subject/audience past
// the checks by adding extra top-level fields. Only the "kubernetes.io" private
// claims are decoded from the raw payload.
//
// Any verification error (go-oidc's descriptive "expired", "audience mismatch",
// bad-signature, or issuer-mismatch messages) is returned as-is; the core
// Verifier collapses it into the single generic failure, so the descriptive
// text never reaches the caller.
func (a *oidcAuthenticator) AuthenticateToken(ctx context.Context, rawToken string) (*verify.VerifiedClaims, error) {
	idToken, err := a.verifier.Verify(ctx, rawToken)
	if err != nil {
		return nil, err
	}

	claims := &verify.VerifiedClaims{}
	// Decode the raw payload for the "kubernetes.io" private claims. This also
	// touches the exported standard-claim fields, so overwrite them afterwards
	// with the values go-oidc actually validated.
	if err := idToken.Claims(claims); err != nil {
		return nil, fmt.Errorf("oidc: decoding token claims: %w", err)
	}
	claims.Issuer = idToken.Issuer
	claims.Subject = idToken.Subject
	claims.Audience = append([]string(nil), idToken.Audience...)
	return claims, nil
}

// NewRemoteVerifier returns a [verify.Verifier] whose signatures and standard
// claims are checked against issuer's OIDC discovery document, with the token
// audience required to contain audience.
//
// Construction performs OIDC discovery: go-oidc fetches
// {issuer}/.well-known/openid-configuration, verifies that the document's
// "issuer" field equals issuer (an issuer-confusion guard), reads its
// "jwks_uri", and builds a key set that fetches and caches keys from there,
// refreshing on rotation. Because go-oidc owns these paths, no discovery or JWKS
// URL is hardcoded here.
//
// The returned verifier is long-lived and safe for concurrent use; construct one
// per (issuer, audience) and share it. The provided ctx governs discovery AND
// the key set's subsequent background fetches, so it should be the lifetime
// context of the verifying process, not a per-request context.
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

	// Thread a custom client through go-oidc via the context it uses for all
	// HTTP. go-oidc's remote key set fetches keys using the constructor context,
	// so the client set here also governs background key refreshes.
	if cfg.httpClient != nil {
		ctx = coreosoidc.ClientContext(ctx, cfg.httpClient)
	}

	// Discovery. NewProvider fetches the well-known document and verifies that
	// its "issuer" matches the configured issuer; a mismatch (issuer confusion)
	// fails here.
	provider, err := coreosoidc.NewProvider(ctx, issuer)
	if err != nil {
		return nil, fmt.Errorf("oidc: OIDC discovery for issuer %q failed: %w", issuer, err)
	}

	// provider.Verifier enforces the signing-algorithm allowlist advertised by
	// discovery, the signature, the issuer, the audience (ClientID) and expiry.
	idv := provider.Verifier(&coreosoidc.Config{ClientID: audience})

	return verify.NewVerifier(&oidcAuthenticator{verifier: idv})
}
