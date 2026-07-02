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

package josekeyset

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	jose "gopkg.in/go-jose/go-jose.v2"
)

const (
	// defaultMaxBodyBytes bounds how much of a discovery or JWKS response body is
	// read into memory, guarding against a hostile or misconfigured issuer
	// serving an unbounded body.
	defaultMaxBodyBytes = 1 << 20 // 1 MiB

	// defaultMinRefreshInterval rate-limits JWKS refetches so that a burst of
	// unverifiable tokens cannot stampede the issuer's discovery/JWKS endpoints.
	defaultMinRefreshInterval = time.Minute

	// defaultHTTPTimeout is a safety net applied only when the caller does not
	// supply its own *http.Client. Per-request cancellation still flows from the
	// context passed to VerifySignature.
	defaultHTTPTimeout = 30 * time.Second

	discoveryPath = "/.well-known/openid-configuration"
)

// Coarse, non-descriptive errors. The core verifier collapses any failure into a
// generic ErrVerificationFailed, so these strings must not leak issuer state or
// help an attacker enumerate valid keys/tokens.
var (
	errRemoteNoKey          = errors.New("josekeyset: no key verified the token signature")
	errRemoteIssuerMismatch = errors.New("josekeyset: discovery document issuer mismatch")
	errRemoteInsecureScheme = errors.New("josekeyset: refusing non-HTTPS URL")
	errRemoteMalformed      = errors.New("josekeyset: malformed discovery or JWKS document")
)

// RemoteKeySet is a [verify.KeySet] that discovers an issuer's signing keys over
// the network via OIDC discovery and JWKS, then verifies compact JWS tokens
// against them. It is the real-world counterpart to StaticKeySet.
//
// Keys are fetched lazily on first use and cached. On a verification miss (for
// example, after the issuer rotates its signing key) the key set is refreshed at
// most once per WithMinRefreshInterval and the token is retried a single time.
//
// RemoteKeySet is safe for concurrent use.
//
// Dependency caveat: like the rest of this package, RemoteKeySet imports
// gopkg.in/go-jose/go-jose.v2, which is not yet a direct dependency of the
// k8s.io/client-go module. Publishing it requires hack/pin-dependency.sh +
// hack/update-vendor.sh and sig-auth sign-off; do not hand-edit go.mod/go.work.
type RemoteKeySet struct {
	issuer             string
	httpClient         *http.Client
	allowHTTP          bool
	minRefreshInterval time.Duration
	maxBodyBytes       int64

	// mu guards the mutable cache below.
	mu          sync.RWMutex
	jwksURI     string // discovered once from the issuer's discovery document
	keys        *jose.JSONWebKeySet
	lastRefresh time.Time

	// refreshMu single-flights network refreshes so concurrent callers do not
	// stampede the issuer.
	refreshMu sync.Mutex
}

// RemoteOption configures a RemoteKeySet.
type RemoteOption func(*RemoteKeySet)

// WithHTTPClient sets the HTTP client used for discovery and JWKS requests. The
// caller controls transport-level security (TLS roots, proxies, timeouts)
// through this client. A nil client is ignored.
func WithHTTPClient(c *http.Client) RemoteOption {
	return func(r *RemoteKeySet) {
		if c != nil {
			r.httpClient = c
		}
	}
}

// WithMinRefreshInterval sets the minimum time between JWKS refetches triggered
// by verification misses. Smaller values pick up key rotation faster at the cost
// of more requests to the issuer. Non-positive values are ignored.
func WithMinRefreshInterval(d time.Duration) RemoteOption {
	return func(r *RemoteKeySet) {
		if d > 0 {
			r.minRefreshInterval = d
		}
	}
}

// WithInsecureAllowHTTP permits plain-http discovery and JWKS URLs. It exists
// only for tests against a local server and MUST NOT be used in production:
// without TLS, an on-path attacker can substitute signing keys.
func WithInsecureAllowHTTP() RemoteOption {
	return func(r *RemoteKeySet) {
		r.allowHTTP = true
	}
}

// NewRemoteKeySet returns a RemoteKeySet for the given issuer. The issuer must be
// an absolute HTTPS URL (http is permitted only under WithInsecureAllowHTTP).
// Signing keys are not fetched until the first VerifySignature call.
func NewRemoteKeySet(issuer string, opts ...RemoteOption) (*RemoteKeySet, error) {
	if issuer == "" {
		return nil, errors.New("josekeyset: issuer must not be empty")
	}
	r := &RemoteKeySet{
		issuer:             issuer,
		httpClient:         &http.Client{Timeout: defaultHTTPTimeout},
		minRefreshInterval: defaultMinRefreshInterval,
		maxBodyBytes:       defaultMaxBodyBytes,
	}
	for _, opt := range opts {
		opt(r)
	}
	if err := r.checkScheme(issuer); err != nil {
		return nil, err
	}
	return r, nil
}

// VerifySignature implements verify.KeySet. It parses rawToken as a compact JWS,
// verifies it against the issuer's cached keys, and on a miss refreshes the keys
// once (subject to the min-refresh interval) before retrying. Errors are
// intentionally coarse.
func (r *RemoteKeySet) VerifySignature(ctx context.Context, rawToken string) ([]byte, error) {
	jws, err := jose.ParseSigned(rawToken)
	if err != nil {
		return nil, err
	}

	keys, err := r.currentKeys(ctx)
	if err != nil {
		return nil, err
	}
	if payload, ok := verifyWithKeys(jws, keys); ok {
		return payload, nil
	}

	// Miss: the issuer may have rotated keys. Refresh at most once and retry.
	keys, err = r.refreshOnMiss(ctx)
	if err != nil {
		return nil, err
	}
	if payload, ok := verifyWithKeys(jws, keys); ok {
		return payload, nil
	}
	return nil, errRemoteNoKey
}

// verifyWithKeys tries to verify jws against keys, preferring the key whose kid
// matches the JWS header and falling back to trying every key.
func verifyWithKeys(jws *jose.JSONWebSignature, keys *jose.JSONWebKeySet) ([]byte, bool) {
	if keys == nil {
		return nil, false
	}
	var kid string
	if len(jws.Signatures) > 0 {
		kid = jws.Signatures[0].Header.KeyID
	}
	candidates := keys.Keys
	if kid != "" {
		if matched := keys.Key(kid); len(matched) > 0 {
			candidates = matched
		}
	}
	for i := range candidates {
		if payload, err := jws.Verify(&candidates[i]); err == nil {
			return payload, true
		}
	}
	return nil, false
}

// currentKeys returns the cached key set, fetching it if it has not been loaded.
func (r *RemoteKeySet) currentKeys(ctx context.Context) (*jose.JSONWebKeySet, error) {
	r.mu.RLock()
	keys := r.keys
	r.mu.RUnlock()
	if keys != nil {
		return keys, nil
	}
	return r.refresh(ctx, time.Time{})
}

// refreshOnMiss refreshes the key set after a verification miss, honoring the
// minimum refresh interval to prevent refetch storms.
func (r *RemoteKeySet) refreshOnMiss(ctx context.Context) (*jose.JSONWebKeySet, error) {
	r.mu.RLock()
	keys := r.keys
	last := r.lastRefresh
	r.mu.RUnlock()

	if keys != nil && !last.IsZero() && time.Since(last) < r.minRefreshInterval {
		// Rate-limited: reuse the cached keys. Verification will fail again and
		// the core verifier reports a generic failure, but the issuer is spared
		// a stampede of refetches on a burst of bad tokens.
		return keys, nil
	}
	return r.refresh(ctx, last)
}

// refresh fetches the JWKS (discovering the jwks_uri first, once) and updates the
// cache. It single-flights via refreshMu: if another goroutine already refreshed
// past the caller's observed lastRefresh (since), the freshly cached keys are
// returned without another network round-trip.
func (r *RemoteKeySet) refresh(ctx context.Context, since time.Time) (*jose.JSONWebKeySet, error) {
	r.refreshMu.Lock()
	defer r.refreshMu.Unlock()

	r.mu.RLock()
	keys := r.keys
	last := r.lastRefresh
	jwksURI := r.jwksURI
	r.mu.RUnlock()
	if keys != nil && last.After(since) {
		return keys, nil
	}

	if jwksURI == "" {
		discovered, err := r.discover(ctx)
		if err != nil {
			return nil, err
		}
		jwksURI = discovered
	}

	fetched, err := r.fetchJWKS(ctx, jwksURI)
	if err != nil {
		return nil, err
	}

	r.mu.Lock()
	r.jwksURI = jwksURI
	r.keys = fetched
	r.lastRefresh = time.Now()
	r.mu.Unlock()
	return fetched, nil
}

// discover fetches and validates the issuer's OIDC discovery document, returning
// its jwks_uri. It requires the document's issuer to exactly match the configured
// issuer (preventing issuer confusion) and the jwks_uri to be HTTPS.
func (r *RemoteKeySet) discover(ctx context.Context) (string, error) {
	discoveryURL := strings.TrimSuffix(r.issuer, "/") + discoveryPath
	var doc struct {
		Issuer  string `json:"issuer"`
		JWKSURI string `json:"jwks_uri"`
	}
	if err := r.getJSON(ctx, discoveryURL, &doc); err != nil {
		return "", err
	}
	if doc.Issuer != r.issuer {
		return "", errRemoteIssuerMismatch
	}
	if err := r.checkScheme(doc.JWKSURI); err != nil {
		return "", err
	}
	return doc.JWKSURI, nil
}

// fetchJWKS retrieves and parses the JSON Web Key Set at jwksURI.
func (r *RemoteKeySet) fetchJWKS(ctx context.Context, jwksURI string) (*jose.JSONWebKeySet, error) {
	var ks jose.JSONWebKeySet
	if err := r.getJSON(ctx, jwksURI, &ks); err != nil {
		return nil, err
	}
	if len(ks.Keys) == 0 {
		return nil, errRemoteMalformed
	}
	return &ks, nil
}

// checkScheme enforces the HTTPS-only policy, allowing http only when the
// test-only WithInsecureAllowHTTP option is set.
func (r *RemoteKeySet) checkScheme(rawURL string) error {
	u, err := url.Parse(rawURL)
	if err != nil || u.Host == "" {
		return errRemoteMalformed
	}
	switch u.Scheme {
	case "https":
		return nil
	case "http":
		if r.allowHTTP {
			return nil
		}
		return errRemoteInsecureScheme
	default:
		return errRemoteInsecureScheme
	}
}

// getJSON performs a bounded GET and decodes the body as JSON into v.
func (r *RemoteKeySet) getJSON(ctx context.Context, rawURL string, v interface{}) error {
	body, err := r.get(ctx, rawURL)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(body, v); err != nil {
		return errRemoteMalformed
	}
	return nil
}

// get issues a context-bound GET and returns the response body, capped at
// maxBodyBytes.
func (r *RemoteKeySet) get(ctx context.Context, rawURL string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := r.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("josekeyset: unexpected response status %d", resp.StatusCode)
	}
	return io.ReadAll(io.LimitReader(resp.Body, r.maxBodyBytes))
}
