package azure

import (
	"bytes"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Azure/go-autorest/autorest/azure"
)

func TestAzureTokenSourceFromFile(t *testing.T) {
	testAzureFile := `
	[
	  {
	    "expiresIn": 3599,
	    "_authority": "https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47",
	    "refreshToken": "test",
	    "tokenType": "Bearer",
	    "expiresOn": "2017-04-03 10:36:27.808196",
	    "oid": "ceb33d00-c512-4657-84fa-c6d9433291e4",
	    "userId": "test",
	    "isMRRT": true,
	    "_clientId": "04b07795-8ddb-461a-bbee-02f9e1bf7b46",
	    "resource": "https://management.core.windows.net/",
	    "accessToken": "test"
	  },
	  {
	    "expiresIn": 3599,
	    "_authority": "https://login.microsoftonline.com/common",
	    "oid": "ceb33d00-c512-4657-84fa-c6d9433291e4",
	    "refreshToken": "test",
	    "tokenType": "Bearer",
	    "expiresOn": "2017-04-03 10:36:25.657865",
	    "userId": "test",
	    "isMRRT": true,
	    "_clientId": "04b07795-8ddb-461a-bbee-02f9e1bf7b46",
	    "resource": "https://management.core.windows.net/",
	    "accessToken": "test"
	  }
	] `

	reader := bytes.NewReader([]byte(testAzureFile))
	fileSouce := &azureTokenSourceFromFile{
		tokensReader: reader,
	}

	token, err := fileSouce.Token()
	if err != nil {
		t.Errorf("failed to read the token from azure file: %v", err)
	}

	wantAccessToken := "test"
	if strings.Compare(token.token.AccessToken, wantAccessToken) != 0 {
		t.Errorf("Token() accessToken error: got %v, want %v", token.token.AccessToken, wantAccessToken)
	}
	wantRefreshToken := "test"
	if strings.Compare(token.token.RefreshToken, wantAccessToken) != 0 {
		t.Errorf("Token() accessToken error: got %v, want %v", token.token.RefreshToken, wantRefreshToken)
	}

	wantExpirationTime := "1491215787"
	if strings.Compare(token.token.ExpiresOn, wantExpirationTime) != 0 {
		t.Errorf("Token() expiresOn error: got %v, want %v", token.token.ExpiresOn, wantExpirationTime)
	}

	wantClientID := "04b07795-8ddb-461a-bbee-02f9e1bf7b46"
	if strings.Compare(token.clientID, wantClientID) != 0 {
		t.Errorf("Token() clientID error: got %v, want %v", token.clientID, wantClientID)
	}

	wantTenantID := "72f988bf-86f1-41af-91ab-2d7cd011db47"
	if strings.Compare(token.tenantID, wantTenantID) != 0 {
		t.Errorf("Token() tenantID error: got %v, want %v", token.tenantID, wantTenantID)
	}
}

func TestAzureTokenSourceFromCache(t *testing.T) {
	fakeAccessToken := "fake token 1"
	fakeSource := fakeTokenSource{
		accessToken: fakeAccessToken,
		expiresOn:   strconv.FormatInt(time.Now().Add(3600*time.Second).Unix(), 10),
	}
	cfg := make(map[string]string)
	persiter := &fakePersister{cache: make(map[string]string)}
	tokenCache := newAzureTokenCache()
	cacheSource := newAzureTokenSourceFromCache(&fakeSource, tokenCache, cfg, persiter)
	token, err := cacheSource.Token()
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
	for k, v := range persistedCfg {
		if strings.Compare(v, wantCfg[k]) != 0 {
			t.Errorf("Token() persisted cfg %s: got %v, want %v", k, v, wantCfg[k])
		}
	}

	fakeSource.accessToken = "fake token 2"
	token, err = cacheSource.Token()
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
		token:    newFackeAzureToken(ts.accessToken, ts.expiresOn),
		clientID: "fake",
		tenantID: "fake",
	}, nil
}

func token2Cfg(token *azureToken) map[string]string {
	cfg := make(map[string]string)
	cfg[cfgAccessToken] = token.token.AccessToken
	cfg[cfgRefreshToken] = token.token.RefreshToken
	cfg[cfgTokenType] = token.token.Type
	cfg[cfgClientID] = token.clientID
	cfg[cfgTenantID] = token.tenantID
	cfg[cfgAudience] = token.token.Resource
	cfg[cfgExpiresIn] = token.token.ExpiresIn
	cfg[cfgExpiresOn] = token.token.ExpiresOn
	return cfg
}

func newFackeAzureToken(accessToken string, expiresOn string) azure.Token {
	return azure.Token{
		AccessToken:  accessToken,
		RefreshToken: "fake",
		ExpiresIn:    "3600",
		ExpiresOn:    expiresOn,
		NotBefore:    expiresOn,
		Resource:     "fake",
		Type:         "fake",
	}
}
