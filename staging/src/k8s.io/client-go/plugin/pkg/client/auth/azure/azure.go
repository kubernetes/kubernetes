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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"os/user"
	"path/filepath"
	"regexp"
	"strconv"
	"sync"
	"time"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/golang/glog"

	restclient "k8s.io/client-go/rest"
)

const (
	azureTimeFormat = "\"2006-01-02 15:04:05.99999\""
	azureAuthority  = "https://login.microsoftonline.com"
	uuidFormat      = "[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-4[a-fA-F0-9]{3}-[8|9|aA|bB][a-fA-F0-9]{3}-[a-fA-F0-9]{12}"
	azureTokenKey   = "azureTokenKey"

	cfgClientID = "client-id"
	cfgTenantID = "tenant-id"
	cfgAudience = "audience"
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

	clientID := cfg[cfgClientID]
	tenantID := cfg[cfgTenantID]
	audience := cfg[cfgAudience]
	if clientID != "" && tenantID != "" && audience != "" {
		ts = newAzureTokenSourceDeviceCode(clientID, tenantID, audience)
	} else {
		azFilename, err := azureTokensFile()
		if err != nil {
			return nil, err
		}
		ts, err = newAzureTokenSourceFromFile(azFilename)
		if err != nil {
			return nil, err
		}
	}

	cacheSource := newAzureTokenSourceFromCache(ts, cache)

	return &azureAuthProvider{
		tokenSource: cacheSource,
		persister:   persister,
	}, nil
}

type azureAuthProvider struct {
	tokenSource tokenSource
	persister   restclient.AuthProviderConfigPersister
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

type azureTokenSourceFromCache struct {
	source tokenSource
	cache  *azureTokenCache
}

func newAzureTokenSourceFromCache(source tokenSource, cache *azureTokenCache) tokenSource {
	return &azureTokenSourceFromCache{
		source: source,
		cache:  cache,
	}
}

// Token fetches a token from the cache if present otherwise acquires a new token from
// the configured source. Automatically refreshes the token if expired.
func (ts *azureTokenSourceFromCache) Token() (*azureToken, error) {
	var err error
	token := ts.cache.getToken(azureTokenKey)
	if token == nil {
		token, err = ts.source.Token()
		if err != nil {
			return nil, err
		}
		if !token.token.IsExpired() {
			ts.cache.setToken(azureTokenKey, token)
		}
	}
	if token.token.IsExpired() {
		token, err = ts.refreshToken(token)
		if err != nil {
			return nil, err
		}
		ts.cache.setToken(azureTokenKey, token)
	}
	return token, nil
}

func (ts *azureTokenSourceFromCache) refreshToken(token *azureToken) (*azureToken, error) {
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

// azureTokensFile return the file where Azure CLI 2.0 stores the access tokens
func azureTokensFile() (string, error) {
	const tokensFile = "/.azure/accessTokens.json"
	usr, err := user.Current()
	if err != nil {
		return "", err
	}
	return filepath.Join(usr.HomeDir, tokensFile), nil
}

type azureTokenSourceFromFile struct {
	tokensReader io.Reader
}

func newAzureTokenSourceFromFile(filename string) (tokenSource, error) {
	_, err := os.Stat(filename)
	if err != nil {
		return nil, err
	}
	r, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	return &azureTokenSourceFromFile{
		tokensReader: r,
	}, nil
}

// Token reads a token from Azure CLI tokens file
func (ts *azureTokenSourceFromFile) Token() (*azureToken, error) {
	b, err := ioutil.ReadAll(ts.tokensReader)
	if err != nil {
		return nil, err
	}

	type azToken struct {
		AccessToken  string    `json:"accessToken"`
		TokenType    string    `json:"tokenType"`
		RefreshToken string    `json:"refreshToken"`
		ExpiresOn    azureTime `json:"expiresOn"`
		ExpiresIn    int64     `json:"expiresIn"`
		Authority    string    `json:"_authority"`
		Resource     string    `json:"resource"`
		ClientID     string    `json:"_clientId"`
	}
	tokens := make([]azToken, 0)
	err = json.Unmarshal(b, &tokens)
	if err != nil {
		return nil, err
	}

	// Choose the token which has a tenant. Typically in this file there is also stored
	// a token without a tenant (e.g common token).
	var tenantToken *azToken
	for _, token := range tokens {
		if ts.isTeantAuthroity(token.Authority) {
			tenantToken = &token
			break
		}
	}
	if tenantToken == nil {
		return nil, fmt.Errorf("no token issued by authority %s found in ~/.azure/accessTokens.json file",
			azureAuthority)
	}

	return &azureToken{
		token: azure.Token{
			AccessToken:  tenantToken.AccessToken,
			RefreshToken: tenantToken.RefreshToken,
			ExpiresIn:    strconv.FormatInt(tenantToken.ExpiresIn, 10),
			ExpiresOn:    strconv.FormatInt(tenantToken.ExpiresOn.Time.Unix(), 10),
			NotBefore:    strconv.FormatInt(tenantToken.ExpiresOn.Time.Unix(), 10),
			Resource:     tenantToken.Resource,
			Type:         tenantToken.TokenType,
		},
		clientID: tenantToken.ClientID,
		tenantID: ts.extractTenantID(tenantToken.Authority),
	}, nil
}

func (ts *azureTokenSourceFromFile) isTeantAuthroity(authority string) bool {
	r := regexp.MustCompile(fmt.Sprintf("^%s/%s$", azureAuthority, uuidFormat))
	return r.MatchString(authority)
}

func (ts *azureTokenSourceFromFile) extractTenantID(authority string) string {
	r := regexp.MustCompile(uuidFormat)
	return r.FindString(authority)
}

type azureTime struct {
	time.Time
}

func (t *azureTime) UnmarshalJSON(b []byte) error {
	time, err := time.Parse(azureTimeFormat, string(b))
	if err != nil {
		return err
	}
	t.Time = time
	return nil
}

type azureTokenSourceDeviceCode struct {
	clinetID string
	tenantID string
	audience string
}

func newAzureTokenSourceDeviceCode(clientID string, tenantID string, audience string) tokenSource {
	return &azureTokenSourceDeviceCode{
		clinetID: clientID,
		tenantID: tenantID,
		audience: audience,
	}
}

func (ts *azureTokenSourceDeviceCode) Token() (*azureToken, error) {
	oauthConfig, err := azure.PublicCloud.OAuthConfigForTenant(ts.tenantID)
	if err != nil {
		return nil, err
	}
	client := &autorest.Client{}
	deviceCode, err := azure.InitiateDeviceAuth(client, *oauthConfig, ts.clinetID, ts.audience)
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
		clientID: ts.clinetID,
		tenantID: ts.tenantID,
	}, nil
}
