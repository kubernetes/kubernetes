// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Package storage holds all cached token information for MSAL. This storage can be
// augmented with third-party extensions to provide persistent storage. In that case,
// reads and writes in upper packages will call Marshal() to take the entire in-memory
// representation and write it to storage and Unmarshal() to update the entire in-memory
// storage with what was in the persistent storage.  The persistent storage can only be
// accessed in this way because multiple MSAL clients written in multiple languages can
// access the same storage and must adhere to the same method that was defined
// previously.
package storage

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/json"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth/ops/accesstokens"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth/ops/authority"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/shared"
)

// aadInstanceDiscoveryer allows faking in tests.
// It is implemented in production by ops/authority.Client
type aadInstanceDiscoveryer interface {
	AADInstanceDiscovery(ctx context.Context, authorityInfo authority.Info) (authority.InstanceDiscoveryResponse, error)
}

// TokenResponse mimics a token response that was pulled from the cache.
type TokenResponse struct {
	RefreshToken accesstokens.RefreshToken
	IDToken      IDToken // *Credential
	AccessToken  AccessToken
	Account      shared.Account
}

// Manager is an in-memory cache of access tokens, accounts and meta data. This data is
// updated on read/write calls. Unmarshal() replaces all data stored here with whatever
// was given to it on each call.
type Manager struct {
	contract   *Contract
	contractMu sync.RWMutex
	requests   aadInstanceDiscoveryer // *oauth.Token

	aadCacheMu sync.RWMutex
	aadCache   map[string]authority.InstanceDiscoveryMetadata
}

// New is the constructor for Manager.
func New(requests *oauth.Client) *Manager {
	m := &Manager{requests: requests, aadCache: make(map[string]authority.InstanceDiscoveryMetadata)}
	m.contract = NewContract()
	return m
}

func checkAlias(alias string, aliases []string) bool {
	for _, v := range aliases {
		if alias == v {
			return true
		}
	}
	return false
}

func isMatchingScopes(scopesOne []string, scopesTwo string) bool {
	newScopesTwo := strings.Split(scopesTwo, scopeSeparator)
	scopeCounter := 0
	for _, scope := range scopesOne {
		for _, otherScope := range newScopesTwo {
			if strings.EqualFold(scope, otherScope) {
				scopeCounter++
				continue
			}
		}
	}
	return scopeCounter == len(scopesOne)
}

// Read reads a storage token from the cache if it exists.
func (m *Manager) Read(ctx context.Context, authParameters authority.AuthParams, account shared.Account) (TokenResponse, error) {
	tr := TokenResponse{}
	homeAccountID := authParameters.HomeAccountID
	realm := authParameters.AuthorityInfo.Tenant
	clientID := authParameters.ClientID
	scopes := authParameters.Scopes

	// fetch metadata if instanceDiscovery is enabled
	aliases := []string{authParameters.AuthorityInfo.Host}
	if !authParameters.AuthorityInfo.InstanceDiscoveryDisabled {
		metadata, err := m.getMetadataEntry(ctx, authParameters.AuthorityInfo)
		if err != nil {
			return TokenResponse{}, err
		}
		aliases = metadata.Aliases
	}

	accessToken := m.readAccessToken(homeAccountID, aliases, realm, clientID, scopes)
	tr.AccessToken = accessToken

	if account.IsZero() {
		return tr, nil
	}
	// errors returned by read* methods indicate a cache miss and are therefore non-fatal. We continue populating
	// TokenResponse fields so that e.g. lack of an ID token doesn't prevent the caller from receiving a refresh token.
	idToken, err := m.readIDToken(homeAccountID, aliases, realm, clientID)
	if err == nil {
		tr.IDToken = idToken
	}

	if appMetadata, err := m.readAppMetaData(aliases, clientID); err == nil {
		// we need the family ID to identify the correct refresh token, if any
		familyID := appMetadata.FamilyID
		refreshToken, err := m.readRefreshToken(homeAccountID, aliases, familyID, clientID)
		if err == nil {
			tr.RefreshToken = refreshToken
		}
	}

	account, err = m.readAccount(homeAccountID, aliases, realm)
	if err == nil {
		tr.Account = account
	}
	return tr, nil
}

const scopeSeparator = " "

// Write writes a token response to the cache and returns the account information the token is stored with.
func (m *Manager) Write(authParameters authority.AuthParams, tokenResponse accesstokens.TokenResponse) (shared.Account, error) {
	authParameters.HomeAccountID = tokenResponse.ClientInfo.HomeAccountID()
	homeAccountID := authParameters.HomeAccountID
	environment := authParameters.AuthorityInfo.Host
	realm := authParameters.AuthorityInfo.Tenant
	clientID := authParameters.ClientID
	target := strings.Join(tokenResponse.GrantedScopes.Slice, scopeSeparator)
	cachedAt := time.Now()

	var account shared.Account

	if len(tokenResponse.RefreshToken) > 0 {
		refreshToken := accesstokens.NewRefreshToken(homeAccountID, environment, clientID, tokenResponse.RefreshToken, tokenResponse.FamilyID)
		if err := m.writeRefreshToken(refreshToken); err != nil {
			return account, err
		}
	}

	if len(tokenResponse.AccessToken) > 0 {
		accessToken := NewAccessToken(
			homeAccountID,
			environment,
			realm,
			clientID,
			cachedAt,
			tokenResponse.ExpiresOn.T,
			tokenResponse.ExtExpiresOn.T,
			target,
			tokenResponse.AccessToken,
		)

		// Since we have a valid access token, cache it before moving on.
		if err := accessToken.Validate(); err == nil {
			if err := m.writeAccessToken(accessToken); err != nil {
				return account, err
			}
		}
	}

	idTokenJwt := tokenResponse.IDToken
	if !idTokenJwt.IsZero() {
		idToken := NewIDToken(homeAccountID, environment, realm, clientID, idTokenJwt.RawToken)
		if err := m.writeIDToken(idToken); err != nil {
			return shared.Account{}, err
		}

		localAccountID := idTokenJwt.LocalAccountID()
		authorityType := authParameters.AuthorityInfo.AuthorityType

		preferredUsername := idTokenJwt.UPN
		if idTokenJwt.PreferredUsername != "" {
			preferredUsername = idTokenJwt.PreferredUsername
		}

		account = shared.NewAccount(
			homeAccountID,
			environment,
			realm,
			localAccountID,
			authorityType,
			preferredUsername,
		)
		if err := m.writeAccount(account); err != nil {
			return shared.Account{}, err
		}
	}

	AppMetaData := NewAppMetaData(tokenResponse.FamilyID, clientID, environment)

	if err := m.writeAppMetaData(AppMetaData); err != nil {
		return shared.Account{}, err
	}
	return account, nil
}

func (m *Manager) getMetadataEntry(ctx context.Context, authorityInfo authority.Info) (authority.InstanceDiscoveryMetadata, error) {
	md, err := m.aadMetadataFromCache(ctx, authorityInfo)
	if err != nil {
		// not in the cache, retrieve it
		md, err = m.aadMetadata(ctx, authorityInfo)
	}
	return md, err
}

func (m *Manager) aadMetadataFromCache(ctx context.Context, authorityInfo authority.Info) (authority.InstanceDiscoveryMetadata, error) {
	m.aadCacheMu.RLock()
	defer m.aadCacheMu.RUnlock()
	metadata, ok := m.aadCache[authorityInfo.Host]
	if ok {
		return metadata, nil
	}
	return metadata, errors.New("not found")
}

func (m *Manager) aadMetadata(ctx context.Context, authorityInfo authority.Info) (authority.InstanceDiscoveryMetadata, error) {
	m.aadCacheMu.Lock()
	defer m.aadCacheMu.Unlock()
	discoveryResponse, err := m.requests.AADInstanceDiscovery(ctx, authorityInfo)
	if err != nil {
		return authority.InstanceDiscoveryMetadata{}, err
	}

	for _, metadataEntry := range discoveryResponse.Metadata {
		for _, aliasedAuthority := range metadataEntry.Aliases {
			m.aadCache[aliasedAuthority] = metadataEntry
		}
	}
	if _, ok := m.aadCache[authorityInfo.Host]; !ok {
		m.aadCache[authorityInfo.Host] = authority.InstanceDiscoveryMetadata{
			PreferredNetwork: authorityInfo.Host,
			PreferredCache:   authorityInfo.Host,
		}
	}
	return m.aadCache[authorityInfo.Host], nil
}

func (m *Manager) readAccessToken(homeID string, envAliases []string, realm, clientID string, scopes []string) AccessToken {
	m.contractMu.RLock()
	defer m.contractMu.RUnlock()
	// TODO: linear search (over a map no less) is slow for a large number (thousands) of tokens.
	// this shows up as the dominating node in a profile. for real-world scenarios this likely isn't
	// an issue, however if it does become a problem then we know where to look.
	for _, at := range m.contract.AccessTokens {
		if at.HomeAccountID == homeID && at.Realm == realm && at.ClientID == clientID {
			if checkAlias(at.Environment, envAliases) {
				if isMatchingScopes(scopes, at.Scopes) {
					return at
				}
			}
		}
	}
	return AccessToken{}
}

func (m *Manager) writeAccessToken(accessToken AccessToken) error {
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	key := accessToken.Key()
	m.contract.AccessTokens[key] = accessToken
	return nil
}

func (m *Manager) readRefreshToken(homeID string, envAliases []string, familyID, clientID string) (accesstokens.RefreshToken, error) {
	byFamily := func(rt accesstokens.RefreshToken) bool {
		return matchFamilyRefreshToken(rt, homeID, envAliases)
	}
	byClient := func(rt accesstokens.RefreshToken) bool {
		return matchClientIDRefreshToken(rt, homeID, envAliases, clientID)
	}

	var matchers []func(rt accesstokens.RefreshToken) bool
	if familyID == "" {
		matchers = []func(rt accesstokens.RefreshToken) bool{
			byClient, byFamily,
		}
	} else {
		matchers = []func(rt accesstokens.RefreshToken) bool{
			byFamily, byClient,
		}
	}

	// TODO(keegan): All the tests here pass, but Bogdan says this is
	// more complicated.  I'm opening an issue for this to have him
	// review the tests and suggest tests that would break this so
	// we can re-write against good tests. His comments as follow:
	// The algorithm is a bit more complex than this, I assume there are some tests covering everything. I would keep the order as is.
	// The algorithm is:
	// If application is NOT part of the family, search by client_ID
	// If app is part of the family or if we DO NOT KNOW if it's part of the family, search by family ID, then by client_id (we will know if an app is part of the family after the first token response).
	// https://github.com/AzureAD/microsoft-authentication-library-for-dotnet/blob/311fe8b16e7c293462806f397e189a6aa1159769/src/client/Microsoft.Identity.Client/Internal/Requests/Silent/CacheSilentStrategy.cs#L95
	m.contractMu.RLock()
	defer m.contractMu.RUnlock()
	for _, matcher := range matchers {
		for _, rt := range m.contract.RefreshTokens {
			if matcher(rt) {
				return rt, nil
			}
		}
	}

	return accesstokens.RefreshToken{}, fmt.Errorf("refresh token not found")
}

func matchFamilyRefreshToken(rt accesstokens.RefreshToken, homeID string, envAliases []string) bool {
	return rt.HomeAccountID == homeID && checkAlias(rt.Environment, envAliases) && rt.FamilyID != ""
}

func matchClientIDRefreshToken(rt accesstokens.RefreshToken, homeID string, envAliases []string, clientID string) bool {
	return rt.HomeAccountID == homeID && checkAlias(rt.Environment, envAliases) && rt.ClientID == clientID
}

func (m *Manager) writeRefreshToken(refreshToken accesstokens.RefreshToken) error {
	key := refreshToken.Key()
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	m.contract.RefreshTokens[key] = refreshToken
	return nil
}

func (m *Manager) readIDToken(homeID string, envAliases []string, realm, clientID string) (IDToken, error) {
	m.contractMu.RLock()
	defer m.contractMu.RUnlock()
	for _, idt := range m.contract.IDTokens {
		if idt.HomeAccountID == homeID && idt.Realm == realm && idt.ClientID == clientID {
			if checkAlias(idt.Environment, envAliases) {
				return idt, nil
			}
		}
	}
	return IDToken{}, fmt.Errorf("token not found")
}

func (m *Manager) writeIDToken(idToken IDToken) error {
	key := idToken.Key()
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	m.contract.IDTokens[key] = idToken
	return nil
}

func (m *Manager) AllAccounts() []shared.Account {
	m.contractMu.RLock()
	defer m.contractMu.RUnlock()

	var accounts []shared.Account
	for _, v := range m.contract.Accounts {
		accounts = append(accounts, v)
	}

	return accounts
}

func (m *Manager) Account(homeAccountID string) shared.Account {
	m.contractMu.RLock()
	defer m.contractMu.RUnlock()

	for _, v := range m.contract.Accounts {
		if v.HomeAccountID == homeAccountID {
			return v
		}
	}

	return shared.Account{}
}

func (m *Manager) readAccount(homeAccountID string, envAliases []string, realm string) (shared.Account, error) {
	m.contractMu.RLock()
	defer m.contractMu.RUnlock()

	// You might ask why, if cache.Accounts is a map, we would loop through all of these instead of using a key.
	// We only use a map because the storage contract shared between all language implementations says use a map.
	// We can't change that. The other is because the keys are made using a specific "env", but here we are allowing
	// a match in multiple envs (envAlias). That means we either need to hash each possible keyand do the lookup
	// or just statically check.  Since the design is to have a storage.Manager per user, the amount of keys stored
	// is really low (say 2).  Each hash is more expensive than the entire iteration.
	for _, acc := range m.contract.Accounts {
		if acc.HomeAccountID == homeAccountID && checkAlias(acc.Environment, envAliases) && acc.Realm == realm {
			return acc, nil
		}
	}
	return shared.Account{}, fmt.Errorf("account not found")
}

func (m *Manager) writeAccount(account shared.Account) error {
	key := account.Key()
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	m.contract.Accounts[key] = account
	return nil
}

func (m *Manager) readAppMetaData(envAliases []string, clientID string) (AppMetaData, error) {
	m.contractMu.RLock()
	defer m.contractMu.RUnlock()

	for _, app := range m.contract.AppMetaData {
		if checkAlias(app.Environment, envAliases) && app.ClientID == clientID {
			return app, nil
		}
	}
	return AppMetaData{}, fmt.Errorf("not found")
}

func (m *Manager) writeAppMetaData(AppMetaData AppMetaData) error {
	key := AppMetaData.Key()
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	m.contract.AppMetaData[key] = AppMetaData
	return nil
}

// RemoveAccount removes all the associated ATs, RTs and IDTs from the cache associated with this account.
func (m *Manager) RemoveAccount(account shared.Account, clientID string) {
	m.removeRefreshTokens(account.HomeAccountID, account.Environment, clientID)
	m.removeAccessTokens(account.HomeAccountID, account.Environment)
	m.removeIDTokens(account.HomeAccountID, account.Environment)
	m.removeAccounts(account.HomeAccountID, account.Environment)
}

func (m *Manager) removeRefreshTokens(homeID string, env string, clientID string) {
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	for key, rt := range m.contract.RefreshTokens {
		// Check for RTs associated with the account.
		if rt.HomeAccountID == homeID && rt.Environment == env {
			// Do RT's app ownership check as a precaution, in case family apps
			// and 3rd-party apps share same token cache, although they should not.
			if rt.ClientID == clientID || rt.FamilyID != "" {
				delete(m.contract.RefreshTokens, key)
			}
		}
	}
}

func (m *Manager) removeAccessTokens(homeID string, env string) {
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	for key, at := range m.contract.AccessTokens {
		// Remove AT's associated with the account
		if at.HomeAccountID == homeID && at.Environment == env {
			// # To avoid the complexity of locating sibling family app's AT, we skip AT's app ownership check.
			// It means ATs for other apps will also be removed, it is OK because:
			// non-family apps are not supposed to share token cache to begin with;
			// Even if it happens, we keep other app's RT already, so SSO still works.
			delete(m.contract.AccessTokens, key)
		}
	}
}

func (m *Manager) removeIDTokens(homeID string, env string) {
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	for key, idt := range m.contract.IDTokens {
		// Remove ID tokens associated with the account.
		if idt.HomeAccountID == homeID && idt.Environment == env {
			delete(m.contract.IDTokens, key)
		}
	}
}

func (m *Manager) removeAccounts(homeID string, env string) {
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	for key, acc := range m.contract.Accounts {
		// Remove the specified account.
		if acc.HomeAccountID == homeID && acc.Environment == env {
			delete(m.contract.Accounts, key)
		}
	}
}

// update updates the internal cache object. This is for use in tests, other uses are not
// supported.
func (m *Manager) update(cache *Contract) {
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	m.contract = cache
}

// Marshal implements cache.Marshaler.
func (m *Manager) Marshal() ([]byte, error) {
	return json.Marshal(m.contract)
}

// Unmarshal implements cache.Unmarshaler.
func (m *Manager) Unmarshal(b []byte) error {
	m.contractMu.Lock()
	defer m.contractMu.Unlock()

	contract := NewContract()

	err := json.Unmarshal(b, contract)
	if err != nil {
		return err
	}

	m.contract = contract

	return nil
}
