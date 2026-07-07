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

package oidckeyset

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"

	"k8s.io/webhook-auth/verify"
	"k8s.io/webhook-auth/verify/admissionhttp"
)

// defaultServiceAccountTokenPath is where the kubelet projects the pod's
// service-account token. An in-cluster webhook always holds one; its "iss" claim
// is the cluster's SA-token issuer, which is exactly the issuer whose OIDC
// discovery and JWKS verify KEP-6060 tokens.
const defaultServiceAccountTokenPath = "/var/run/secrets/kubernetes.io/serviceaccount/token"

// inClusterConfig holds the resolved options for NewInClusterVerifier.
type inClusterConfig struct {
	issuer      string
	saTokenPath string
	audiences   []string
	serviceURL  string
	keySet      verify.KeySet
	httpClient  *http.Client
	verifyOpts  []verify.Option
}

// InClusterOption configures NewInClusterVerifier. Every default is overridable;
// the strict verify.NewVerifier path is never bypassed.
type InClusterOption func(*inClusterConfig)

// WithIssuer overrides the token issuer. By default the issuer is discovered
// from the "iss" claim of the pod's own projected service-account token. The
// issuer is still validated end to end: it is matched against the OIDC discovery
// document (by go-oidc) AND against every token's "iss" (by the core Verifier),
// so this only selects WHICH issuer to trust, it does not relax validation.
func WithIssuer(issuer string) InClusterOption {
	return func(c *inClusterConfig) {
		if issuer != "" {
			c.issuer = issuer
		}
	}
}

// WithServiceAccountTokenPath overrides the path of the projected
// service-account token read to discover the issuer (and, absent other
// configuration, the provisional audience). Primarily for tests.
func WithServiceAccountTokenPath(path string) InClusterOption {
	return func(c *inClusterConfig) {
		if path != "" {
			c.saTokenPath = path
		}
	}
}

// WithAudiences sets the expected audiences explicitly, overriding all audience
// inference. This is the recommended path once the expected audience is known.
func WithAudiences(audiences ...string) InClusterOption {
	return func(c *inClusterConfig) {
		if len(audiences) > 0 {
			c.audiences = append([]string(nil), audiences...)
		}
	}
}

// WithServiceURL derives the PROVISIONAL expected audience from the webhook's
// own service URL (host+path), using the same formula as the request-time
// admissionhttp.DeriveExpectedAudience. Overridden by WithAudiences.
//
// PROVISIONAL — must match the issuer-side (kube-apiserver) audience derivation
// once KEP-6060 finalizes it.
func WithServiceURL(rawURL string) InClusterOption {
	return func(c *inClusterConfig) {
		if rawURL != "" {
			c.serviceURL = rawURL
		}
	}
}

// WithKeySet supplies a pre-built verify.KeySet, skipping OIDC discovery. Use
// this to share one remote key set across verifiers, or to inject a fake in
// tests.
func WithKeySet(ks verify.KeySet) InClusterOption {
	return func(c *inClusterConfig) {
		if ks != nil {
			c.keySet = ks
		}
	}
}

// WithDiscoveryHTTPClient sets the *http.Client used for OIDC discovery and
// JWKS fetches when this constructor builds the key set (i.e. when WithKeySet is
// not used). In-cluster this is where a client trusting the cluster CA is
// injected.
func WithDiscoveryHTTPClient(client *http.Client) InClusterOption {
	return func(c *inClusterConfig) {
		if client != nil {
			c.httpClient = client
		}
	}
}

// WithVerifierOptions forwards options (for example verify.WithLeeway) to the
// underlying verify.NewVerifier.
func WithVerifierOptions(opts ...verify.Option) InClusterOption {
	return func(c *inClusterConfig) {
		c.verifyOpts = append(c.verifyOpts, opts...)
	}
}

// NewInClusterVerifier builds a *verify.Verifier with working defaults so an
// in-cluster webhook can construct one with minimal or no explicit
// configuration, while every default remains overridable.
//
// Defaults:
//
//   - Issuer / KeySet: the issuer is read from the "iss" claim of the pod's own
//     projected service-account token (WithServiceAccountTokenPath overrides the
//     path; WithIssuer overrides the value). A remote KeySet is then built via
//     OIDC discovery against that issuer (WithKeySet supplies one instead). The
//     issuer is validated, not merely used: go-oidc matches it against the
//     discovery document, and the core Verifier matches every token's "iss"
//     against it — so this does not weaken issuer validation.
//
//   - Audience: resolved in order — WithAudiences (explicit), else WithServiceURL
//     (derive from the webhook's own service URL), else a PROVISIONAL default
//     taken from the "aud" of the pod's own service-account token. The audience
//     derivation format is an OPEN QUESTION on the issuer side and is NOT
//     finalized in KEP-6060; the provisional defaults let a webhook construct
//     with zero audience config, but deployments that know their audience should
//     set WithAudiences. See admissionhttp.DeriveExpectedAudience for the
//     request-time counterpart.
//
// The returned Verifier goes through the strict verify.NewVerifier, so a nil
// key set, empty issuer, or empty audience list still fails fast here rather
// than silently at request time.
func NewInClusterVerifier(ctx context.Context, opts ...InClusterOption) (*verify.Verifier, error) {
	cfg := &inClusterConfig{saTokenPath: defaultServiceAccountTokenPath}
	for _, opt := range opts {
		opt(cfg)
	}

	// Read the pod's own SA token once for whatever defaults still need it
	// (issuer and/or provisional audience). If everything needed was supplied
	// explicitly, skip the read entirely so the constructor works off-cluster.
	needToken := cfg.issuer == "" || (len(cfg.audiences) == 0 && cfg.serviceURL == "")
	var selfClaims serviceAccountTokenClaims
	if needToken {
		claims, err := readServiceAccountTokenClaims(cfg.saTokenPath)
		if err != nil {
			return nil, fmt.Errorf("oidckeyset: resolving in-cluster defaults: %w", err)
		}
		selfClaims = claims
	}

	// Resolve issuer.
	issuer := cfg.issuer
	if issuer == "" {
		issuer = selfClaims.Issuer
	}
	if issuer == "" {
		return nil, errors.New("oidckeyset: could not determine issuer; the service-account token has no \"iss\" claim — set WithIssuer")
	}

	// Resolve audiences.
	audiences, err := cfg.resolveAudiences(selfClaims)
	if err != nil {
		return nil, err
	}

	// Resolve key set (discovery unless one was supplied).
	keySet := cfg.keySet
	if keySet == nil {
		var ksOpts []Option
		if cfg.httpClient != nil {
			ksOpts = append(ksOpts, WithHTTPClient(cfg.httpClient))
		}
		keySet, err = NewRemoteKeySet(ctx, issuer, ksOpts...)
		if err != nil {
			return nil, err
		}
	}

	return verify.NewVerifier(keySet, issuer, audiences, cfg.verifyOpts...)
}

// resolveAudiences applies the audience precedence: explicit > service URL >
// provisional default from the SA token's own "aud".
func (c *inClusterConfig) resolveAudiences(selfClaims serviceAccountTokenClaims) ([]string, error) {
	if len(c.audiences) > 0 {
		return c.audiences, nil
	}
	if c.serviceURL != "" {
		aud, err := admissionhttp.AudienceFromServiceURL(c.serviceURL)
		if err != nil {
			return nil, fmt.Errorf("oidckeyset: deriving audience from service URL: %w", err)
		}
		return []string{aud}, nil
	}
	// PROVISIONAL zero-config default: fall back to the audience(s) the cluster
	// stamped into the webhook's OWN service-account token. This lets a webhook
	// construct with no audience configuration at all; it is a placeholder, NOT
	// the finalized KEP-6060 audience, and should be replaced with WithAudiences
	// (or WithServiceURL) in any deployment that knows its expected audience.
	if len(selfClaims.Audience) > 0 {
		return append([]string(nil), selfClaims.Audience...), nil
	}
	return nil, errors.New("oidckeyset: could not determine an expected audience; set WithAudiences or WithServiceURL")
}

// serviceAccountTokenClaims is the minimal, UNVERIFIED view of a JWT payload used
// only to discover the in-cluster issuer and a provisional audience. The token
// is never trusted on the strength of this parse: signatures and all claims are
// re-verified per request by the core Verifier against the discovered keys.
type serviceAccountTokenClaims struct {
	Issuer   string       `json:"iss"`
	Audience audienceList `json:"aud"`
}

// audienceList decodes the JWT "aud" claim, which may be a single string or a
// list of strings (RFC 7519 §4.1.3).
type audienceList []string

func (a *audienceList) UnmarshalJSON(b []byte) error {
	var list []string
	if err := json.Unmarshal(b, &list); err == nil {
		*a = list
		return nil
	}
	var single string
	if err := json.Unmarshal(b, &single); err != nil {
		return err
	}
	*a = audienceList{single}
	return nil
}

// readServiceAccountTokenClaims reads the projected SA token at path and returns
// its UNVERIFIED "iss"/"aud" claims. It performs no signature check — it exists
// only to discover configuration defaults, and the returned issuer/audience are
// re-validated for every request by the core Verifier.
func readServiceAccountTokenClaims(path string) (serviceAccountTokenClaims, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return serviceAccountTokenClaims{}, fmt.Errorf("reading service-account token %q: %w", path, err)
	}
	return parseUnverifiedClaims(strings.TrimSpace(string(raw)))
}

// parseUnverifiedClaims decodes the payload segment of a compact JWS WITHOUT
// verifying its signature. Used solely to read defaults; never for trust.
func parseUnverifiedClaims(token string) (serviceAccountTokenClaims, error) {
	parts := strings.Split(token, ".")
	if len(parts) < 2 {
		return serviceAccountTokenClaims{}, errors.New("service-account token is not a JWT")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return serviceAccountTokenClaims{}, fmt.Errorf("decoding service-account token payload: %w", err)
	}
	var claims serviceAccountTokenClaims
	if err := json.Unmarshal(payload, &claims); err != nil {
		return serviceAccountTokenClaims{}, fmt.Errorf("parsing service-account token payload: %w", err)
	}
	return claims, nil
}
