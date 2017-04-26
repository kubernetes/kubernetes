/*
Copyright 2017 The Kubernetes Authors.

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

package azure

import (
	"errors"
	"fmt"
	"net/http"
	"sync"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/golang/glog"

	restclient "k8s.io/client-go/rest"
)

const (
	azureTokenKey = "azureTokenKey"

	cfgClientID     = "client-id"
	cfgTenantID     = "tenant-id"
	cfgAccessToken  = "access-token"
	cfgRefreshToken = "refresh-token"
	cfgTokenType    = "token-type"
	cfgExpiresIn    = "expires-in"
	cfgExpiresOn    = "expires-on"
	cfgEnvironment  = "environment"
	cfgApiserverID  = "apiserver-id"
)

func init() {
	if err := restclient.RegisterAuthProviderPlugin("azure", newAzureAuthProvider); err != nil {
		glog.Fatalf("Failed to register azure auth plugin: %v", err)
	}
}

var cache = newAzureTokenCache()

type azureTokenCache struct {
	lock  sync.Mutex
	cache map[string]*azureToken
}

func newAzureTokenCache() *azureTokenCache {
	return &azureTokenCache{cache: make(map[string]*azureToken)}
}

func (c *azureTokenCache) getToken(tokenKey string) *azureToken {
	c.lock.Lock()
	defer c.lock.Unlock()
	return c.cache[tokenKey]
}

func (c *azureTokenCache) setToken(tokenKey string, token *azureToken) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.cache[tokenKey] = token
}

func newAzureAuthProvider(_ string, cfg map[string]string, persister restclient.AuthProviderConfigPersister) (restclient.AuthProvider, error) {
	var ts tokenSource

	environment, err := azure.EnvironmentFromName(cfg[cfgEnvironment])
	if err != nil {
		environment = azure.PublicCloud
	}
	ts, err = newAzureTokenSourceDeviceCode(environment, cfg[cfgClientID], cfg[cfgTenantID], cfg[cfgApiserverID])
	if err != nil {
		return nil, err
	}
	cacheSource := newAzureTokenSource(ts, cache, cfg, persister)

	return &azureAuthProvider{
		tokenSource: cacheSource,
	}, nil
}

type azureAuthProvider struct {
	tokenSource tokenSource
}

func (p *azureAuthProvider) Login() error {
	return errors.New("not yet implemented")
}

func (p *azureAuthProvider) WrapTransport(rt http.RoundTripper) http.RoundTripper {
	return &azureRoundTripper{
		tokenSource:  p.tokenSource,
		roundTripper: rt,
	}
}

type azureRoundTripper struct {
	tokenSource  tokenSource
	roundTripper http.RoundTripper
}

func (r *azureRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	token, err := r.tokenSource.Token()
	if err != nil {
		glog.Errorf("Failed to acquire a token: %v", err)
		return nil, err
	}

	// clone the request in order to avoid modifying the headers of the original request
	req2 := new(http.Request)
	*req2 = *req
	req2.Header = make(http.Header, len(req.Header))
	for k, s := range req.Header {
		req2.Header[k] = append([]string(nil), s...)
	}

	req2.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token.token.AccessToken))

	return r.roundTripper.RoundTrip(req2)
}

type azureToken struct {
	token    azure.Token
	clientID string
	tenantID string
}

type tokenSource interface {
	Token() (*azureToken, error)
}

type azureTokenSource struct {
	source    tokenSource
	cache     *azureTokenCache
	cfg       map[string]string
	persister restclient.AuthProviderConfigPersister
}

func newAzureTokenSource(source tokenSource, cache *azureTokenCache, cfg map[string]string, persister restclient.AuthProviderConfigPersister) tokenSource {
	return &azureTokenSource{
		source:    source,
		cache:     cache,
		cfg:       cfg,
		persister: persister,
	}
}

// Token fetches a token from the cache of configuration if present otherwise
// acquires a new token from the configured source. Automatically refreshes
// the token if expired.
func (ts *azureTokenSource) Token() (*azureToken, error) {
	var err error
	token := ts.cache.getToken(azureTokenKey)
	if token == nil {
		token, err = ts.retrieveTokenFromCfg()
		if err != nil {
			token, err = ts.source.Token()
			if err != nil {
				return nil, err
			}
		}
		if !token.token.IsExpired() {
			ts.cache.setToken(azureTokenKey, token)
			err = ts.storeTokenInCfg(token)
			if err != nil {
				return nil, err
			}
		}
	}
	if token.token.IsExpired() {
		token, err = ts.refreshToken(token)
		if err != nil {
			return nil, err
		}
		ts.cache.setToken(azureTokenKey, token)
		err = ts.storeTokenInCfg(token)
		if err != nil {
			return nil, err
		}
	}
	return token, nil
}

func (ts *azureTokenSource) retrieveTokenFromCfg() (*azureToken, error) {
	accessToken := ts.cfg[cfgAccessToken]
	if accessToken == "" {
		return nil, fmt.Errorf("no access token in cfg: %s", cfgAccessToken)
	}
	refreshToken := ts.cfg[cfgRefreshToken]
	if refreshToken == "" {
		return nil, fmt.Errorf("no refresh token in cfg: %s", cfgRefreshToken)
	}
	tokenType := ts.cfg[cfgTokenType]
	if tokenType == "" {
		tokenType = "Bearer"
	}
	clientID := ts.cfg[cfgClientID]
	if clientID == "" {
		return nil, fmt.Errorf("no client ID in cfg: %s", cfgClientID)
	}
	tenantID := ts.cfg[cfgTenantID]
	if tenantID == "" {
		return nil, fmt.Errorf("no tenant ID in cfg: %s", cfgTenantID)
	}
	apiserverID := ts.cfg[cfgApiserverID]
	if apiserverID == "" {
		return nil, fmt.Errorf("no apiserver ID in cfg: %s", apiserverID)
	}
	expiresIn := ts.cfg[cfgExpiresIn]
	if expiresIn == "" {
		return nil, fmt.Errorf("no expiresIn in cfg: %s", cfgExpiresIn)
	}
	expiresOn := ts.cfg[cfgExpiresOn]
	if expiresOn == "" {
		return nil, fmt.Errorf("no expiresOn in cfg: %s", cfgExpiresOn)
	}

	return &azureToken{
		token: azure.Token{
			AccessToken:  accessToken,
			RefreshToken: refreshToken,
			ExpiresIn:    expiresIn,
			ExpiresOn:    expiresOn,
			NotBefore:    expiresOn,
			Resource:     apiserverID,
			Type:         tokenType,
		},
		clientID: clientID,
		tenantID: tenantID,
	}, nil
}

func (ts *azureTokenSource) storeTokenInCfg(token *azureToken) error {
	newCfg := make(map[string]string)
	newCfg[cfgAccessToken] = token.token.AccessToken
	newCfg[cfgRefreshToken] = token.token.RefreshToken
	newCfg[cfgTokenType] = token.token.Type
	newCfg[cfgClientID] = token.clientID
	newCfg[cfgTenantID] = token.tenantID
	newCfg[cfgApiserverID] = token.token.Resource
	newCfg[cfgExpiresIn] = token.token.ExpiresIn
	newCfg[cfgExpiresOn] = token.token.ExpiresOn

	err := ts.persister.Persist(newCfg)
	if err != nil {
		return err
	}
	ts.cfg = newCfg
	return nil
}

func (ts *azureTokenSource) refreshToken(token *azureToken) (*azureToken, error) {
	oauthConfig, err := azure.PublicCloud.OAuthConfigForTenant(token.tenantID)
	if err != nil {
		return nil, err
	}

	callback := func(t azure.Token) error {
		return nil
	}
	spt, err := azure.NewServicePrincipalTokenFromManualToken(
		*oauthConfig,
		token.clientID,
		token.token.Resource,
		token.token,
		callback)
	if err != nil {
		return nil, err
	}

	err = spt.Refresh()
	if err != nil {
		return nil, err
	}

	return &azureToken{
		token:    spt.Token,
		clientID: token.clientID,
		tenantID: token.tenantID,
	}, nil
}

type azureTokenSourceDeviceCode struct {
	environment azure.Environment
	clientID    string
	tenantID    string
	apiserverID string
}

func newAzureTokenSourceDeviceCode(environment azure.Environment, clientID string, tenantID string, apiserverID string) (tokenSource, error) {
	if clientID == "" {
		return nil, errors.New("client-id is empty")
	}
	if tenantID == "" {
		return nil, errors.New("tenant-id is empty")
	}
	if apiserverID == "" {
		return nil, errors.New("apiserver-id is empty")
	}
	return &azureTokenSourceDeviceCode{
		environment: environment,
		clientID:    clientID,
		tenantID:    tenantID,
		apiserverID: apiserverID,
	}, nil
}

func (ts *azureTokenSourceDeviceCode) Token() (*azureToken, error) {
	oauthConfig, err := ts.environment.OAuthConfigForTenant(ts.tenantID)
	if err != nil {
		return nil, err
	}
	client := &autorest.Client{}
	deviceCode, err := azure.InitiateDeviceAuth(client, *oauthConfig, ts.clientID, ts.apiserverID)
	if err != nil {
		return nil, err
	}

	fmt.Println(*deviceCode.Message)

	token, err := azure.WaitForUserCompletion(client, deviceCode)
	if err != nil {
		return nil, err
	}

	return &azureToken{
		token:    *token,
		clientID: ts.clientID,
		tenantID: ts.tenantID,
	}, nil
}
