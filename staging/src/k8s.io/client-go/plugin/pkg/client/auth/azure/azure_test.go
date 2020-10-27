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
	"net/http"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Azure/go-autorest/autorest/adal"
	"github.com/Azure/go-autorest/autorest/azure"
)

func TestAzureAuthProvider(t *testing.T) {
	t.Run("validate against invalid configurations", func(t *testing.T) {
		vectors := []struct {
			cfg           map[string]string
			expectedError string
		}{
			{
				cfg: map[string]string{
					cfgClientID:    "foo",
					cfgApiserverID: "foo",
					cfgTenantID:    "foo",
					cfgConfigMode:  "-1",
				},
				expectedError: "config-mode:-1 is not a valid mode",
			},
			{
				cfg: map[string]string{
					cfgClientID:    "foo",
					cfgApiserverID: "foo",
					cfgTenantID:    "foo",
					cfgConfigMode:  "2",
				},
				expectedError: "config-mode:2 is not a valid mode",
			},
			{
				cfg: map[string]string{
					cfgClientID:    "foo",
					cfgApiserverID: "foo",
					cfgTenantID:    "foo",
					cfgConfigMode:  "foo",
				},
				expectedError: "failed to parse config-mode, error: strconv.Atoi: parsing \"foo\": invalid syntax",
			},
		}

		for _, v := range vectors {
			persister := &fakePersister{}
			_, err := newAzureAuthProvider("", v.cfg, persister)
			if !strings.Contains(err.Error(), v.expectedError) {
				t.Errorf("cfg %v should fail with message containing '%s'. actual: '%s'", v.cfg, v.expectedError, err)
			}
		}
	})

	t.Run("it should return non-nil provider in happy cases", func(t *testing.T) {
		vectors := []struct {
			cfg                map[string]string
			expectedConfigMode configMode
		}{
			{
				cfg: map[string]string{
					cfgClientID:    "foo",
					cfgApiserverID: "foo",
					cfgTenantID:    "foo",
				},
				expectedConfigMode: configModeDefault,
			},
			{
				cfg: map[string]string{
					cfgClientID:    "foo",
					cfgApiserverID: "foo",
					cfgTenantID:    "foo",
					cfgConfigMode:  "0",
				},
				expectedConfigMode: configModeDefault,
			},
			{
				cfg: map[string]string{
					cfgClientID:    "foo",
					cfgApiserverID: "foo",
					cfgTenantID:    "foo",
					cfgConfigMode:  "1",
				},
				expectedConfigMode: configModeOmitSPNPrefix,
			},
		}

		for _, v := range vectors {
			persister := &fakePersister{}
			provider, err := newAzureAuthProvider("", v.cfg, persister)
			if err != nil {
				t.Errorf("newAzureAuthProvider should not fail with '%s'", err)
			}
			if provider == nil {
				t.Fatalf("newAzureAuthProvider should return non-nil provider")
			}
			azureProvider := provider.(*azureAuthProvider)
			if azureProvider == nil {
				t.Fatalf("newAzureAuthProvider should return an instance of type azureAuthProvider")
			}
			ts := azureProvider.tokenSource.(*azureTokenSource)
			if ts == nil {
				t.Fatalf("azureAuthProvider should be an instance of azureTokenSource")
			}
			if ts.configMode != v.expectedConfigMode {
				t.Errorf("expected configMode: %d, actual: %d", v.expectedConfigMode, ts.configMode)
			}
		}
	})
}

func TestTokenSourceDeviceCode(t *testing.T) {
	var (
		clientID    = "clientID"
		tenantID    = "tenantID"
		apiserverID = "apiserverID"
		configMode  = configModeDefault
		azureEnv    = azure.Environment{}
	)
	t.Run("validate to create azureTokenSourceDeviceCode", func(t *testing.T) {
		if _, err := newAzureTokenSourceDeviceCode(azureEnv, clientID, tenantID, apiserverID, configModeDefault); err != nil {
			t.Errorf("newAzureTokenSourceDeviceCode should not have failed. err: %s", err)
		}

		if _, err := newAzureTokenSourceDeviceCode(azureEnv, clientID, tenantID, apiserverID, configModeOmitSPNPrefix); err != nil {
			t.Errorf("newAzureTokenSourceDeviceCode should not have failed. err: %s", err)
		}

		_, err := newAzureTokenSourceDeviceCode(azureEnv, "", tenantID, apiserverID, configMode)
		actual := "client-id is empty"
		if err.Error() != actual {
			t.Errorf("newAzureTokenSourceDeviceCode should have failed. expected: %s, actual: %s", actual, err)
		}

		_, err = newAzureTokenSourceDeviceCode(azureEnv, clientID, "", apiserverID, configMode)
		actual = "tenant-id is empty"
		if err.Error() != actual {
			t.Errorf("newAzureTokenSourceDeviceCode should have failed. expected: %s, actual: %s", actual, err)
		}

		_, err = newAzureTokenSourceDeviceCode(azureEnv, clientID, tenantID, "", configMode)
		actual = "apiserver-id is empty"
		if err.Error() != actual {
			t.Errorf("newAzureTokenSourceDeviceCode should have failed. expected: %s, actual: %s", actual, err)
		}
	})
}
func TestAzureTokenSource(t *testing.T) {
	configModes := []configMode{configModeOmitSPNPrefix, configModeDefault}
	expectedConfigModes := []string{"1", "0"}

	for i, configMode := range configModes {
		t.Run(fmt.Sprintf("validate token from cfg with configMode %v", configMode), func(t *testing.T) {
			const (
				serverID     = "fakeServerID"
				clientID     = "fakeClientID"
				tenantID     = "fakeTenantID"
				accessToken  = "fakeToken"
				environment  = "fakeEnvironment"
				refreshToken = "fakeToken"
				expiresIn    = "foo"
				expiresOn    = "foo"
			)
			cfg := map[string]string{
				cfgConfigMode:   strconv.Itoa(int(configMode)),
				cfgApiserverID:  serverID,
				cfgClientID:     clientID,
				cfgTenantID:     tenantID,
				cfgEnvironment:  environment,
				cfgAccessToken:  accessToken,
				cfgRefreshToken: refreshToken,
				cfgExpiresIn:    expiresIn,
				cfgExpiresOn:    expiresOn,
			}
			fakeSource := fakeTokenSource{token: newFakeAzureToken("fakeToken", time.Now().Add(3600*time.Second))}
			persiter := &fakePersister{cache: make(map[string]string)}
			tokenCache := newAzureTokenCache()
			tokenSource := newAzureTokenSource(&fakeSource, tokenCache, cfg, configMode, persiter)
			azTokenSource := tokenSource.(*azureTokenSource)
			token, err := azTokenSource.retrieveTokenFromCfg()
			if err != nil {
				t.Errorf("failed to retrieve the token form cfg: %s", err)
			}
			if token.apiserverID != serverID {
				t.Errorf("expecting token.apiserverID: %s, actual: %s", serverID, token.apiserverID)
			}
			if token.clientID != clientID {
				t.Errorf("expecting token.clientID: %s, actual: %s", clientID, token.clientID)
			}
			if token.tenantID != tenantID {
				t.Errorf("expecting token.tenantID: %s, actual: %s", tenantID, token.tenantID)
			}
			expectedAudience := serverID
			if configMode == configModeDefault {
				expectedAudience = fmt.Sprintf("spn:%s", serverID)
			}
			if token.token.Resource != expectedAudience {
				t.Errorf("expecting adal token.Resource: %s, actual: %s", expectedAudience, token.token.Resource)
			}
		})

		t.Run("validate token against cache", func(t *testing.T) {
			fakeAccessToken := "fake token 1"
			fakeSource := fakeTokenSource{token: newFakeAzureToken(fakeAccessToken, time.Now().Add(3600*time.Second))}
			cfg := make(map[string]string)
			persiter := &fakePersister{cache: make(map[string]string)}
			tokenCache := newAzureTokenCache()
			tokenSource := newAzureTokenSource(&fakeSource, tokenCache, cfg, configMode, persiter)
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
			wantCfg[cfgConfigMode] = expectedConfigModes[i]
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

			fakeSource.token = newFakeAzureToken("fake token 2", time.Now().Add(3600*time.Second))
			token, err = tokenSource.Token()
			if err != nil {
				t.Errorf("failed to retrieve the cached token: %v", err)
			}

			if token.token.AccessToken != fakeAccessToken {
				t.Errorf("Token() didn't return the cached token")
			}
		})
	}
}

func TestAzureTokenSourceScenarios(t *testing.T) {
	expiredToken := newFakeAzureToken("expired token", time.Now().Add(-time.Second))
	extendedToken := newFakeAzureToken("extend token", time.Now().Add(1000*time.Second))
	fakeToken := newFakeAzureToken("fake token", time.Now().Add(1000*time.Second))
	wrongToken := newFakeAzureToken("wrong token", time.Now().Add(1000*time.Second))
	tests := []struct {
		name         string
		sourceToken  *azureToken
		refreshToken *azureToken
		cachedToken  *azureToken
		configToken  *azureToken
		expectToken  *azureToken
		tokenErr     error
		refreshErr   error
		expectErr    string
		tokenCalls   uint
		refreshCalls uint
		persistCalls uint
	}{
		{
			name:         "new config",
			sourceToken:  fakeToken,
			expectToken:  fakeToken,
			tokenCalls:   1,
			persistCalls: 1,
		},
		{
			name:        "load token from cache",
			sourceToken: wrongToken,
			cachedToken: fakeToken,
			configToken: wrongToken,
			expectToken: fakeToken,
		},
		{
			name:        "load token from config",
			sourceToken: wrongToken,
			configToken: fakeToken,
			expectToken: fakeToken,
		},
		{
			name:         "cached token timeout, extend success, config token should never load",
			cachedToken:  expiredToken,
			refreshToken: extendedToken,
			configToken:  wrongToken,
			expectToken:  extendedToken,
			refreshCalls: 1,
			persistCalls: 1,
		},
		{
			name:         "config token timeout, extend failure, acquire new token",
			configToken:  expiredToken,
			refreshErr:   fakeTokenRefreshError{message: "FakeError happened when refreshing"},
			sourceToken:  fakeToken,
			expectToken:  fakeToken,
			refreshCalls: 1,
			tokenCalls:   1,
			persistCalls: 1,
		},
		{
			name:         "unexpected error when extend",
			configToken:  expiredToken,
			refreshErr:   errors.New("unexpected refresh error"),
			sourceToken:  fakeToken,
			expectErr:    "unexpected refresh error",
			refreshCalls: 1,
		},
		{
			name:       "token error",
			tokenErr:   errors.New("tokenerr"),
			expectErr:  "tokenerr",
			tokenCalls: 1,
		},
		{
			name:        "Token() got expired token",
			sourceToken: expiredToken,
			expectErr:   "newly acquired token is expired",
			tokenCalls:  1,
		},
		{
			name:        "Token() got nil but no error",
			sourceToken: nil,
			expectErr:   "unable to acquire token",
			tokenCalls:  1,
		},
	}
	for _, tc := range tests {
		configModes := []configMode{configModeOmitSPNPrefix, configModeDefault}

		for _, configMode := range configModes {
			t.Run(fmt.Sprintf("%s with configMode: %v", tc.name, configMode), func(t *testing.T) {
				persister := newFakePersister()

				cfg := map[string]string{
					cfgConfigMode: strconv.Itoa(int(configMode)),
				}
				if tc.configToken != nil {
					cfg = token2Cfg(tc.configToken)
				}

				tokenCache := newAzureTokenCache()
				if tc.cachedToken != nil {
					tokenCache.setToken(azureTokenKey, tc.cachedToken)
				}

				fakeSource := fakeTokenSource{
					token:        tc.sourceToken,
					tokenErr:     tc.tokenErr,
					refreshToken: tc.refreshToken,
					refreshErr:   tc.refreshErr,
				}

				tokenSource := newAzureTokenSource(&fakeSource, tokenCache, cfg, configMode, &persister)
				token, err := tokenSource.Token()

				if token != nil && fakeSource.token != nil && token.apiserverID != fakeSource.token.apiserverID {
					t.Errorf("expecting apiservierID: %s, got: %s", fakeSource.token.apiserverID, token.apiserverID)
				}
				if fakeSource.tokenCalls != tc.tokenCalls {
					t.Errorf("expecting tokenCalls: %v, got: %v", tc.tokenCalls, fakeSource.tokenCalls)
				}

				if fakeSource.refreshCalls != tc.refreshCalls {
					t.Errorf("expecting refreshCalls: %v, got: %v", tc.refreshCalls, fakeSource.refreshCalls)
				}

				if persister.calls != tc.persistCalls {
					t.Errorf("expecting persister calls: %v, got: %v", tc.persistCalls, persister.calls)
				}

				if tc.expectErr != "" {
					if !strings.Contains(err.Error(), tc.expectErr) {
						t.Errorf("expecting error %v, got %v", tc.expectErr, err)
					}
					if token != nil {
						t.Errorf("token should be nil in err situation, got %v", token)
					}
				} else {
					if err != nil {
						t.Fatalf("error should be nil, got %v", err)
					}
					if token.token.AccessToken != tc.expectToken.token.AccessToken {
						t.Errorf("token should have accessToken %v, got %v", token.token.AccessToken, tc.expectToken.token.AccessToken)
					}
				}
			})
		}
	}
}

type fakePersister struct {
	lock  sync.Mutex
	cache map[string]string
	calls uint
}

func newFakePersister() fakePersister {
	return fakePersister{cache: make(map[string]string), calls: 0}
}

func (p *fakePersister) Persist(cache map[string]string) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.calls++
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

// a simple token source simply always returns the token property
type fakeTokenSource struct {
	token        *azureToken
	tokenCalls   uint
	tokenErr     error
	refreshToken *azureToken
	refreshCalls uint
	refreshErr   error
}

func (ts *fakeTokenSource) Token() (*azureToken, error) {
	ts.tokenCalls++
	return ts.token, ts.tokenErr
}

func (ts *fakeTokenSource) Refresh(*azureToken) (*azureToken, error) {
	ts.refreshCalls++
	return ts.refreshToken, ts.refreshErr
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

func newFakeAzureToken(accessToken string, expiresOnTime time.Time) *azureToken {
	return &azureToken{
		token:       newFakeADALToken(accessToken, strconv.FormatInt(expiresOnTime.Unix(), 10)),
		environment: "testenv",
		clientID:    "fake",
		tenantID:    "fake",
		apiserverID: "fake",
	}
}

func newFakeADALToken(accessToken string, expiresOn string) adal.Token {
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

// copied from go-autorest/adal
type fakeTokenRefreshError struct {
	message string
	resp    *http.Response
}

// Error implements the error interface which is part of the TokenRefreshError interface.
func (tre fakeTokenRefreshError) Error() string {
	return tre.message
}

// Response implements the TokenRefreshError interface, it returns the raw HTTP response from the refresh operation.
func (tre fakeTokenRefreshError) Response() *http.Response {
	return tre.resp
}
