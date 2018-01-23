/*
Copyright 2016 The Kubernetes Authors.

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

package oidc

import (
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/oauth2"
	"github.com/coreos/go-oidc/oidc"
	"github.com/golang/glog"

	"k8s.io/client-go/rest"
)

const (
	cfgIssuerUrl                = "idp-issuer-url"
	cfgClientID                 = "client-id"
	cfgClientSecret             = "client-secret"
	cfgCertificateAuthority     = "idp-certificate-authority"
	cfgCertificateAuthorityData = "idp-certificate-authority-data"
	cfgExtraScopes              = "extra-scopes"
	cfgIDToken                  = "id-token"
	cfgRefreshToken             = "refresh-token"
)

func init() {
	if err := rest.RegisterAuthProviderPlugin("oidc", newOIDCAuthProvider); err != nil {
		glog.Fatalf("Failed to register oidc auth plugin: %v", err)
	}
}

// expiryDelta determines how earlier a token should be considered
// expired than its actual expiration time. It is used to avoid late
// expirations due to client-server time mismatches.
//
// NOTE(ericchiang): this is take from golang.org/x/oauth2
const expiryDelta = 10 * time.Second

var cache = newClientCache()

// Like TLS transports, keep a cache of OIDC clients indexed by issuer URL.
type clientCache struct {
	mu    sync.RWMutex
	cache map[cacheKey]*oidcAuthProvider
}

func newClientCache() *clientCache {
	return &clientCache{cache: make(map[cacheKey]*oidcAuthProvider)}
}

type cacheKey struct {
	// Canonical issuer URL string of the provider.
	issuerURL string

	clientID     string
	clientSecret string

	// Don't use CA as cache key because we only add a cache entry if we can connect
	// to the issuer in the first place. A valid CA is a prerequisite.
}

func (c *clientCache) getClient(issuer, clientID, clientSecret string) (*oidcAuthProvider, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	client, ok := c.cache[cacheKey{issuer, clientID, clientSecret}]
	return client, ok
}

// setClient attempts to put the client in the cache but may return any clients
// with the same keys set before. This is so there's only ever one client for a provider.
func (c *clientCache) setClient(issuer, clientID, clientSecret string, client *oidcAuthProvider) *oidcAuthProvider {
	c.mu.Lock()
	defer c.mu.Unlock()
	key := cacheKey{issuer, clientID, clientSecret}

	// If another client has already initialized a client for the given provider we want
	// to use that client instead of the one we're trying to set. This is so all transports
	// share a client and can coordinate around the same mutex when refreshing and writing
	// to the kubeconfig.
	if oldClient, ok := c.cache[key]; ok {
		return oldClient
	}

	c.cache[key] = client
	return client
}

func newOIDCAuthProvider(_ string, cfg map[string]string, persister rest.AuthProviderConfigPersister) (rest.AuthProvider, error) {
	issuer := cfg[cfgIssuerUrl]
	if issuer == "" {
		return nil, fmt.Errorf("Must provide %s", cfgIssuerUrl)
	}

	clientID := cfg[cfgClientID]
	if clientID == "" {
		return nil, fmt.Errorf("Must provide %s", cfgClientID)
	}

	clientSecret := cfg[cfgClientSecret]
	if clientSecret == "" {
		return nil, fmt.Errorf("Must provide %s", cfgClientSecret)
	}

	// Check cache for existing provider.
	if provider, ok := cache.getClient(issuer, clientID, clientSecret); ok {
		return provider, nil
	}

	var certAuthData []byte
	var err error
	if cfg[cfgCertificateAuthorityData] != "" {
		certAuthData, err = base64.StdEncoding.DecodeString(cfg[cfgCertificateAuthorityData])
		if err != nil {
			return nil, err
		}
	}

	clientConfig := rest.Config{
		TLSClientConfig: rest.TLSClientConfig{
			CAFile: cfg[cfgCertificateAuthority],
			CAData: certAuthData,
		},
	}

	trans, err := rest.TransportFor(&clientConfig)
	if err != nil {
		return nil, err
	}
	hc := &http.Client{Transport: trans}

	providerCfg, err := oidc.FetchProviderConfig(hc, issuer)
	if err != nil {
		return nil, fmt.Errorf("error fetching provider config: %v", err)
	}

	scopes := strings.Split(cfg[cfgExtraScopes], ",")
	oidcCfg := oidc.ClientConfig{
		HTTPClient: hc,
		Credentials: oidc.ClientCredentials{
			ID:     clientID,
			Secret: clientSecret,
		},
		ProviderConfig: providerCfg,
		Scope:          append(scopes, oidc.DefaultScope...),
	}
	client, err := oidc.NewClient(oidcCfg)
	if err != nil {
		return nil, fmt.Errorf("error creating OIDC Client: %v", err)
	}

	provider := &oidcAuthProvider{
		client:    &oidcClient{client},
		cfg:       cfg,
		persister: persister,
		now:       time.Now,
	}

	return cache.setClient(issuer, clientID, clientSecret, provider), nil
}

type oidcAuthProvider struct {
	// Interface rather than a raw *oidc.Client for testing.
	client OIDCClient

	// Stubbed out for testing.
	now func() time.Time

	// Mutex guards persisting to the kubeconfig file and allows synchronized
	// updates to the in-memory config. It also ensures concurrent calls to
	// the RoundTripper only trigger a single refresh request.
	mu        sync.Mutex
	cfg       map[string]string
	persister rest.AuthProviderConfigPersister
}

func (p *oidcAuthProvider) WrapTransport(rt http.RoundTripper) http.RoundTripper {
	return &roundTripper{
		wrapped:  rt,
		provider: p,
	}
}

func (p *oidcAuthProvider) Login() error {
	return errors.New("not yet implemented")
}

type OIDCClient interface {
	refreshToken(rt string) (oauth2.TokenResponse, error)
	verifyJWT(jwt *jose.JWT) error
}

type roundTripper struct {
	provider *oidcAuthProvider
	wrapped  http.RoundTripper
}

func (r *roundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	token, err := r.provider.idToken()
	if err != nil {
		return nil, err
	}

	// shallow copy of the struct
	r2 := new(http.Request)
	*r2 = *req
	// deep copy of the Header so we don't modify the original
	// request's Header (as per RoundTripper contract).
	r2.Header = make(http.Header)
	for k, s := range req.Header {
		r2.Header[k] = s
	}
	r2.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))

	return r.wrapped.RoundTrip(r2)
}

func (p *oidcAuthProvider) idToken() (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if idToken, ok := p.cfg[cfgIDToken]; ok && len(idToken) > 0 {
		valid, err := verifyJWTExpiry(p.now(), idToken)
		if err != nil {
			return "", err
		}
		if valid {
			// If the cached id token is still valid use it.
			return idToken, nil
		}
	}

	// Try to request a new token using the refresh token.
	rt, ok := p.cfg[cfgRefreshToken]
	if !ok || len(rt) == 0 {
		return "", errors.New("No valid id-token, and cannot refresh without refresh-token")
	}

	tokens, err := p.client.refreshToken(rt)
	if err != nil {
		return "", fmt.Errorf("could not refresh token: %v", err)
	}
	jwt, err := jose.ParseJWT(tokens.IDToken)
	if err != nil {
		return "", err
	}

	if err := p.client.verifyJWT(&jwt); err != nil {
		return "", err
	}

	// Create a new config to persist.
	newCfg := make(map[string]string)
	for key, val := range p.cfg {
		newCfg[key] = val
	}

	if tokens.RefreshToken != "" && tokens.RefreshToken != rt {
		newCfg[cfgRefreshToken] = tokens.RefreshToken
	}

	newCfg[cfgIDToken] = tokens.IDToken
	if err = p.persister.Persist(newCfg); err != nil {
		return "", fmt.Errorf("could not perist new tokens: %v", err)
	}

	// Update the in memory config to reflect the on disk one.
	p.cfg = newCfg

	return tokens.IDToken, nil
}

// oidcClient is the real implementation of the OIDCClient interface, which is
// used for testing.
type oidcClient struct {
	client *oidc.Client
}

func (o *oidcClient) refreshToken(rt string) (oauth2.TokenResponse, error) {
	oac, err := o.client.OAuthClient()
	if err != nil {
		return oauth2.TokenResponse{}, err
	}

	return oac.RequestToken(oauth2.GrantTypeRefreshToken, rt)
}

func (o *oidcClient) verifyJWT(jwt *jose.JWT) error {
	return o.client.VerifyJWT(*jwt)
}

func verifyJWTExpiry(now time.Time, s string) (valid bool, err error) {
	jwt, err := jose.ParseJWT(s)
	if err != nil {
		return false, fmt.Errorf("invalid %q", cfgIDToken)
	}
	claims, err := jwt.Claims()
	if err != nil {
		return false, err
	}

	exp, ok, err := claims.TimeClaim("exp")
	switch {
	case err != nil:
		return false, fmt.Errorf("failed to parse 'exp' claim: %v", err)
	case !ok:
		return false, errors.New("missing required 'exp' claim")
	case exp.After(now.Add(expiryDelta)):
		return true, nil
	}

	return false, nil
}
