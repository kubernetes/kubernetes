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

// Package oidc constructs admission-token verifiers using OIDC discovery and JWKS.
package oidc // import "k8s.io/component-helpers/auth/admission/internal/oidc"

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"
	"time"

	coreosoidc "github.com/coreos/go-oidc/v3/oidc"
	authenticationv1 "k8s.io/api/authentication/v1"
	"k8s.io/apimachinery/pkg/util/cache"
	"k8s.io/component-helpers/auth/admission/internal/verify"
)

const (
	admissionReviewAPIGroupsClaimKey = authenticationv1.AttestationAdmissionReviewAPIGroups
	maxCachedTokens                  = 1024
	maxVerifiedTokenCacheAge         = 30 * time.Second
	// Keys removed at the issuer can authenticate tokens only until this hard
	// deadline, even when a signature or positive token result was cached.
	signingKeyMaxAge     = time.Minute
	signingKeyRetryDelay = time.Second
)

type verifiedToken struct {
	groups []string
	expiry time.Time
}

func newTokenCache() *cache.LRUExpireCache {
	return cache.NewLRUExpireCache(maxCachedTokens)
}

type webhookBinding struct {
	Name string `json:"name"`
	UID  string `json:"uid"`
}

type webhookPrivateClaims struct {
	Kubernetes struct {
		Pod                            *webhookBinding     `json:"pod"`
		Secret                         *webhookBinding     `json:"secret"`
		Node                           *webhookBinding     `json:"node"`
		ValidatingWebhookConfiguration *webhookBinding     `json:"validatingwebhookconfiguration"`
		MutatingWebhookConfiguration   *webhookBinding     `json:"mutatingwebhookconfiguration"`
		Attestations                   map[string][]string `json:"attestations"`
	} `json:"kubernetes.io"`
}

// verifyTokenGroups caches only successfully verified token claims. Every
// AdmissionReview still undergoes API-group and object checks separately.
func verifyTokenGroups(ctx context.Context, verifier *coreosoidc.IDTokenVerifier, issuer, rawToken string, tokenCache *cache.LRUExpireCache) ([]string, error) {
	var tokenKey [sha256.Size]byte
	if tokenCache != nil {
		// Hash the bearer so its raw contents are never retained as a cache key.
		tokenKey = sha256.Sum256([]byte(rawToken))
		if value, found := tokenCache.Get(tokenKey); found {
			cached := value.(verifiedToken)
			if time.Now().Before(cached.expiry) {
				return append([]string(nil), cached.groups...), nil
			}
			tokenCache.Remove(tokenKey)
		}
	}
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
	// Ordinary ServiceAccount tokens must not authenticate the API server to a
	// webhook. The issuer binds a webhook token to exactly one configuration.
	vwc, mwc := claims.Kubernetes.ValidatingWebhookConfiguration, claims.Kubernetes.MutatingWebhookConfiguration
	if (vwc == nil) == (mwc == nil) || claims.Kubernetes.Pod != nil || claims.Kubernetes.Secret != nil || claims.Kubernetes.Node != nil {
		return nil, errors.New("oidc: token is not bound exclusively to one webhook configuration")
	}
	binding := vwc
	if binding == nil {
		binding = mwc
	}
	if binding.Name == "" || binding.UID == "" {
		return nil, errors.New("oidc: webhook configuration binding must include a name and UID")
	}
	groups := claims.Kubernetes.Attestations[admissionReviewAPIGroupsClaimKey]
	if len(groups) != 1 || groups[0] == "" {
		return nil, fmt.Errorf("oidc: token must carry exactly one nonempty %q attestation", admissionReviewAPIGroupsClaimKey)
	}
	if tokenCache != nil {
		ttl := time.Until(idToken.Expiry)
		if ttl > maxVerifiedTokenCacheAge {
			ttl = maxVerifiedTokenCacheAge
		}
		if ttl > 0 {
			tokenCache.Add(tokenKey, verifiedToken{groups: append([]string(nil), groups...), expiry: idToken.Expiry}, ttl)
		}
	}
	return groups, nil
}

type verifierSnapshot struct {
	verifier   *coreosoidc.IDTokenVerifier
	tokenCache *cache.LRUExpireCache
	validUntil time.Time
}

type signingKeyRefresh struct {
	done chan struct{}
	err  error
}

// refreshingAuthenticator replaces the verifier and its positive token cache
// together. Reads of a fresh snapshot never contact the issuer, including on a
// JWT with an unknown kid or a bad signature.
type refreshingAuthenticator struct {
	issuer   string
	audience string
	loadKeys func(context.Context) (*jwksKeySet, []string, error)
	now      func() time.Time
	current  atomic.Pointer[verifierSnapshot]

	mu         sync.Mutex
	inflight   *signingKeyRefresh
	retryAfter time.Time
	lastErr    error
}

// The local and remote paths share expiry, refresh, and fail-closed behavior;
// only their discovery and signing-algorithm policies differ.
type oidcAuthenticator struct{ *refreshingAuthenticator }
type boundAuthenticator struct{ *refreshingAuthenticator }

func newRefreshingAuthenticator(issuer, audience string, keys *jwksKeySet, algorithms []string, loadKeys func(context.Context) (*jwksKeySet, []string, error), now func() time.Time, refreshStartedAt time.Time) *refreshingAuthenticator {
	a := &refreshingAuthenticator{issuer: issuer, audience: audience, loadKeys: loadKeys, now: now}
	a.current.Store(a.snapshot(keys, algorithms, refreshStartedAt))
	return a
}

func (a *refreshingAuthenticator) snapshot(keys *jwksKeySet, algorithms []string, refreshStartedAt time.Time) *verifierSnapshot {
	return &verifierSnapshot{
		verifier: coreosoidc.NewVerifier(a.issuer, keys, &coreosoidc.Config{
			ClientID:             a.audience,
			SupportedSigningAlgs: append([]string(nil), algorithms...),
		}),
		tokenCache: newTokenCache(),
		// Start the lifetime before network I/O, so a delayed response cannot
		// extend acceptance of a key removed while the refresh was in flight.
		validUntil: refreshStartedAt.Add(signingKeyMaxAge),
	}
}

// AuthenticateToken checks the snapshot deadline before even consulting a
// positive token cache. If a refresh races verification, recheck against the
// replacement rather than returning a result signed by removed keys.
func (a *refreshingAuthenticator) AuthenticateToken(ctx context.Context, rawToken string) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for range 2 {
		snapshot, err := a.freshSnapshot(ctx, false)
		if err != nil {
			return nil, err
		}
		groups, verifyErr := verifyTokenGroups(ctx, snapshot.verifier, a.issuer, rawToken, snapshot.tokenCache)
		if a.current.Load() == snapshot && a.now().Before(snapshot.validUntil) {
			return groups, verifyErr
		}
	}
	return nil, errors.New("oidc: signing keys changed during token verification")
}

func (a *refreshingAuthenticator) freshSnapshot(ctx context.Context, force bool) (*verifierSnapshot, error) {
	if !force {
		if snapshot := a.current.Load(); snapshot != nil && a.now().Before(snapshot.validUntil) {
			return snapshot, nil
		}
	}

	a.mu.Lock()
	if !force {
		if snapshot := a.current.Load(); snapshot != nil && a.now().Before(snapshot.validUntil) {
			a.mu.Unlock()
			return snapshot, nil
		}
	}
	attempt := a.inflight
	if attempt == nil {
		if !force && a.now().Before(a.retryAfter) {
			err := a.lastErr
			a.mu.Unlock()
			return nil, fmt.Errorf("oidc: signing keys unavailable until refresh retry: %w", err)
		}
		attempt = &signingKeyRefresh{done: make(chan struct{})}
		a.inflight = attempt
		// A canceled request may stop waiting, but must not cancel the one
		// bounded refresh that other requests are sharing.
		go a.refresh(attempt)
	}
	a.mu.Unlock()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-attempt.done:
		if attempt.err != nil {
			return nil, attempt.err
		}
		if snapshot := a.current.Load(); snapshot != nil && a.now().Before(snapshot.validUntil) {
			return snapshot, nil
		}
		return nil, errors.New("oidc: signing keys expired during refresh")
	}
}

func (a *refreshingAuthenticator) refresh(attempt *signingKeyRefresh) {
	ctx, cancel := context.WithTimeout(context.Background(), signingKeyRefreshTimeout)
	defer cancel()
	refreshStartedAt := a.now()
	keys, algorithms, err := a.loadKeys(ctx)

	a.mu.Lock()
	if err != nil {
		// HealthCheck uses this same path. A failed probe cannot leave an old
		// verifier live while reporting its signing keys unavailable.
		a.current.Store(nil)
		a.lastErr = fmt.Errorf("oidc: refresh signing keys: %w", err)
		a.retryAfter = a.now().Add(signingKeyRetryDelay)
		attempt.err = a.lastErr
	} else {
		a.current.Store(a.snapshot(keys, algorithms, refreshStartedAt))
		a.lastErr = nil
		a.retryAfter = time.Time{}
	}
	a.inflight = nil
	close(attempt.done)
	a.mu.Unlock()
}

func (a *refreshingAuthenticator) BindAudience(audience string) error {
	if audience == "" {
		return errors.New("oidc: audience must not be empty")
	}
	if audience != a.audience {
		return fmt.Errorf("oidc: audience already bound to %q, refusing to rebind to %q", a.audience, audience)
	}
	return nil
}

// HealthCheck fetches a fresh snapshot through the same path as verification.
// It cannot report a different JWKS ready while verification keeps stale keys.
func (a *refreshingAuthenticator) HealthCheck() error {
	_, err := a.freshSnapshot(context.Background(), true)
	return err
}

func approvedRemoteSigningAlgs(advertised []string) []string {
	if len(advertised) == 0 {
		return []string{"RS256"} // go-oidc's default when discovery omits algorithms.
	}
	allowed := make(map[string]bool, len(remoteSigningAlgs))
	for _, alg := range remoteSigningAlgs {
		allowed[alg] = true
	}
	seen := make(map[string]bool, len(advertised))
	var selected []string
	for _, alg := range advertised {
		if allowed[alg] && !seen[alg] {
			selected = append(selected, alg)
			seen[alg] = true
		}
	}
	return selected
}

// NewProvider reads discovery responses without a size limit. Bound its body
// without changing the TLS transport or redirect policy used for JWKS requests.
type discoveryBodyLimitTransport struct{ http.RoundTripper }

func (t discoveryBodyLimitTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.RoundTripper.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	if resp.Body != nil {
		resp.Body = http.MaxBytesReader(nil, resp.Body, maxDiscoveryBytes)
	}
	return resp, nil
}

func boundedDiscoveryClient(client *http.Client) *http.Client {
	bounded := *client
	transport := bounded.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	bounded.Transport = discoveryBodyLimitTransport{transport}
	return &bounded
}

func loadRemoteSigningKeys(ctx context.Context, issuer string, client *http.Client) (*jwksKeySet, []string, error) {
	// NewProvider checks the pinned issuer on every refresh, not just at
	// construction. Check the claim explicitly too: an injected OIDC context
	// must not be able to disable the issuer pin.
	provider, err := coreosoidc.NewProvider(coreosoidc.ClientContext(ctx, boundedDiscoveryClient(client)), issuer)
	if err != nil {
		return nil, nil, fmt.Errorf("oidc: OIDC discovery for issuer %q failed: %w", issuer, err)
	}
	var metadata struct {
		Issuer      string   `json:"issuer"`
		JWKSURI     string   `json:"jwks_uri"`
		SigningAlgs []string `json:"id_token_signing_alg_values_supported"`
	}
	if err := provider.Claims(&metadata); err != nil {
		return nil, nil, fmt.Errorf("oidc: reading signing-key metadata from discovery: %w", err)
	}
	if metadata.Issuer != issuer {
		return nil, nil, fmt.Errorf("oidc: discovered issuer %q is not the pinned issuer %q", metadata.Issuer, issuer)
	}
	algorithms := approvedRemoteSigningAlgs(metadata.SigningAlgs)
	if len(algorithms) == 0 {
		return nil, nil, errors.New("oidc: discovery advertises no supported asymmetric signing algorithms")
	}
	readiness, err := newKeySetReadiness(metadata.JWKSURI, client, algorithms)
	if err != nil {
		return nil, nil, err
	}
	keys, err := readiness.loadKeys(ctx)
	if err != nil {
		return nil, nil, err
	}
	return keys, algorithms, nil
}

func newRemoteAuthenticator(ctx context.Context, issuer, audience string, httpClient *http.Client, now func() time.Time) (*boundAuthenticator, error) {
	if issuer == "" {
		return nil, errors.New("oidc: issuer must not be empty")
	}
	if audience == "" {
		return nil, errors.New("oidc: audience must not be empty")
	}
	parsedIssuer, err := url.Parse(issuer)
	if err != nil || parsedIssuer.Scheme != "https" || parsedIssuer.Host == "" || parsedIssuer.User != nil || parsedIssuer.RawQuery != "" || parsedIssuer.Fragment != "" {
		return nil, errors.New("oidc: remote issuer must be an HTTPS URL without credentials, query, or fragment")
	}
	client := boundedHTTPClient(httpClient)
	loadKeys := func(ctx context.Context) (*jwksKeySet, []string, error) {
		return loadRemoteSigningKeys(ctx, issuer, client)
	}
	ctx, cancel := context.WithTimeout(ctx, signingKeyRefreshTimeout)
	defer cancel()
	refreshStartedAt := now()
	keys, algorithms, err := loadKeys(ctx)
	if err != nil {
		return nil, fmt.Errorf("oidc: remote signing keys are not ready: %w", err)
	}
	return &boundAuthenticator{newRefreshingAuthenticator(issuer, audience, keys, algorithms, loadKeys, now, refreshStartedAt)}, nil
}

// NewRemoteVerifier returns a [verify.Verifier] for the out-of-cluster path: it
// pins the discovered issuer and configured audience, validates asymmetric
// signatures, and refreshes its JWKS and token cache after at most 60 seconds.
// A removed key may remain usable until that deadline; refresh failures deny.
func NewRemoteVerifier(ctx context.Context, issuer, audience string, httpClient *http.Client) (*verify.Verifier, error) {
	authenticator, err := newRemoteAuthenticator(ctx, issuer, audience, httpClient, time.Now)
	if err != nil {
		return nil, err
	}
	return verify.NewVerifier(authenticator)
}
