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
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Azure/go-autorest/autorest/adal"
)

func TestAzureTokenSource(t *testing.T) {
	fakeAccessToken := "fake token 1"
	fakeSource := fakeTokenSource{
		accessToken: fakeAccessToken,
		expiresOn:   strconv.FormatInt(time.Now().Add(3600*time.Second).Unix(), 10),
	}
	cfg := make(map[string]string)
	persiter := &fakePersister{cache: make(map[string]string)}
	tokenCache := newAzureTokenCache()
	tokenSource := newAzureTokenSource(&fakeSource, tokenCache, cfg, persiter)
	token, err := tokenSource.Token()
	if err != nil {
		t.Errorf("failed to retrieve the token form cache: %v", err)
	}

	wantCacheLen := 1
	if len(tokenCache.cache) != wantCacheLen {
		t.Errorf("Token() cache length error: got %v, want %v", len(tokenCache.cache), wantCacheLen)
	}

	if token != tokenCache.cache[azureTokenKey] {
		t.Error("Token() returned token != cached token")
	}

	wantCfg := token2Cfg(token)
	persistedCfg := persiter.Cache()

	wantCfgLen := len(wantCfg)
	persistedCfgLen := len(persistedCfg)
	if wantCfgLen != persistedCfgLen {
		t.Errorf("wantCfgLen and persistedCfgLen do not match, wantCfgLen=%v, persistedCfgLen=%v", wantCfgLen, persistedCfgLen)
	}

	for k, v := range persistedCfg {
		if strings.Compare(v, wantCfg[k]) != 0 {
			t.Errorf("Token() persisted cfg %s: got %v, want %v", k, v, wantCfg[k])
		}
	}

	fakeSource.accessToken = "fake token 2"
	token, err = tokenSource.Token()
	if err != nil {
		t.Errorf("failed to retrieve the cached token: %v", err)
	}

	if token.token.AccessToken != fakeAccessToken {
		t.Errorf("Token() didn't return the cached token")
	}
}

type fakePersister struct {
	lock  sync.Mutex
	cache map[string]string
}

func (p *fakePersister) Persist(cache map[string]string) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.cache = map[string]string{}
	for k, v := range cache {
		p.cache[k] = v
	}
	return nil
}

func (p *fakePersister) Cache() map[string]string {
	ret := map[string]string{}
	p.lock.Lock()
	defer p.lock.Unlock()
	for k, v := range p.cache {
		ret[k] = v
	}
	return ret
}

type fakeTokenSource struct {
	expiresOn   string
	accessToken string
}

func (ts *fakeTokenSource) Token() (*azureToken, error) {
	return &azureToken{
		token:       newFackeAzureToken(ts.accessToken, ts.expiresOn),
		environment: "testenv",
		clientID:    "fake",
		tenantID:    "fake",
		apiserverID: "fake",
	}, nil
}

func token2Cfg(token *azureToken) map[string]string {
	cfg := make(map[string]string)
	cfg[cfgAccessToken] = token.token.AccessToken
	cfg[cfgRefreshToken] = token.token.RefreshToken
	cfg[cfgEnvironment] = token.environment
	cfg[cfgClientID] = token.clientID
	cfg[cfgTenantID] = token.tenantID
	cfg[cfgApiserverID] = token.apiserverID
	cfg[cfgExpiresIn] = string(token.token.ExpiresIn)
	cfg[cfgExpiresOn] = string(token.token.ExpiresOn)
	return cfg
}

func newFackeAzureToken(accessToken string, expiresOn string) adal.Token {
	return adal.Token{
		AccessToken:  accessToken,
		RefreshToken: "fake",
		ExpiresIn:    "3600",
		ExpiresOn:    json.Number(expiresOn),
		NotBefore:    json.Number(expiresOn),
		Resource:     "fake",
		Type:         "fake",
	}
}
