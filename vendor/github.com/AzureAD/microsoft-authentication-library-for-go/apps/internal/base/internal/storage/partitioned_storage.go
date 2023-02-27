// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

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

// PartitionedManager is a partitioned in-memory cache of access tokens, accounts and meta data.
type PartitionedManager struct {
	contract   *InMemoryContract
	contractMu sync.RWMutex
	requests   aadInstanceDiscoveryer // *oauth.Token

	aadCacheMu sync.RWMutex
	aadCache   map[string]authority.InstanceDiscoveryMetadata
}

// NewPartitionedManager is the constructor for PartitionedManager.
func NewPartitionedManager(requests *oauth.Client) *PartitionedManager {
	m := &PartitionedManager{requests: requests, aadCache: make(map[string]authority.InstanceDiscoveryMetadata)}
	m.contract = NewInMemoryContract()
	return m
}

// Read reads a storage token from the cache if it exists.
func (m *PartitionedManager) Read(ctx context.Context, authParameters authority.AuthParams) (TokenResponse, error) {
	tr := TokenResponse{}
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

	userAssertionHash := authParameters.AssertionHash()
	partitionKeyFromRequest := userAssertionHash

	// errors returned by read* methods indicate a cache miss and are therefore non-fatal. We continue populating
	// TokenResponse fields so that e.g. lack of an ID token doesn't prevent the caller from receiving a refresh token.
	accessToken, err := m.readAccessToken(aliases, realm, clientID, userAssertionHash, scopes, partitionKeyFromRequest)
	if err == nil {
		tr.AccessToken = accessToken
	}
	idToken, err := m.readIDToken(aliases, realm, clientID, userAssertionHash, getPartitionKeyIDTokenRead(accessToken))
	if err == nil {
		tr.IDToken = idToken
	}

	if appMetadata, err := m.readAppMetaData(aliases, clientID); err == nil {
		// we need the family ID to identify the correct refresh token, if any
		familyID := appMetadata.FamilyID
		refreshToken, err := m.readRefreshToken(aliases, familyID, clientID, userAssertionHash, partitionKeyFromRequest)
		if err == nil {
			tr.RefreshToken = refreshToken
		}
	}

	account, err := m.readAccount(aliases, realm, userAssertionHash, idToken.HomeAccountID)
	if err == nil {
		tr.Account = account
	}
	return tr, nil
}

// Write writes a token response to the cache and returns the account information the token is stored with.
func (m *PartitionedManager) Write(authParameters authority.AuthParams, tokenResponse accesstokens.TokenResponse) (shared.Account, error) {
	authParameters.HomeAccountID = tokenResponse.ClientInfo.HomeAccountID()
	homeAccountID := authParameters.HomeAccountID
	environment := authParameters.AuthorityInfo.Host
	realm := authParameters.AuthorityInfo.Tenant
	clientID := authParameters.ClientID
	target := strings.Join(tokenResponse.GrantedScopes.Slice, scopeSeparator)
	userAssertionHash := authParameters.AssertionHash()
	cachedAt := time.Now()

	var account shared.Account

	if len(tokenResponse.RefreshToken) > 0 {
		refreshToken := accesstokens.NewRefreshToken(homeAccountID, environment, clientID, tokenResponse.RefreshToken, tokenResponse.FamilyID)
		if authParameters.AuthorizationType == authority.ATOnBehalfOf {
			refreshToken.UserAssertionHash = userAssertionHash
		}
		if err := m.writeRefreshToken(refreshToken, getPartitionKeyRefreshToken(refreshToken)); err != nil {
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
		if authParameters.AuthorizationType == authority.ATOnBehalfOf {
			accessToken.UserAssertionHash = userAssertionHash // get Hash method on this
		}

		// Since we have a valid access token, cache it before moving on.
		if err := accessToken.Validate(); err == nil {
			if err := m.writeAccessToken(accessToken, getPartitionKeyAccessToken(accessToken)); err != nil {
				return account, err
			}
		} else {
			return shared.Account{}, err
		}
	}

	idTokenJwt := tokenResponse.IDToken
	if !idTokenJwt.IsZero() {
		idToken := NewIDToken(homeAccountID, environment, realm, clientID, idTokenJwt.RawToken)
		if authParameters.AuthorizationType == authority.ATOnBehalfOf {
			idToken.UserAssertionHash = userAssertionHash
		}
		if err := m.writeIDToken(idToken, getPartitionKeyIDToken(idToken)); err != nil {
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
		if authParameters.AuthorizationType == authority.ATOnBehalfOf {
			account.UserAssertionHash = userAssertionHash
		}
		if err := m.writeAccount(account, getPartitionKeyAccount(account)); err != nil {
			return shared.Account{}, err
		}
	}

	AppMetaData := NewAppMetaData(tokenResponse.FamilyID, clientID, environment)

	if err := m.writeAppMetaData(AppMetaData); err != nil {
		return shared.Account{}, err
	}
	return account, nil
}

func (m *PartitionedManager) getMetadataEntry(ctx context.Context, authorityInfo authority.Info) (authority.InstanceDiscoveryMetadata, error) {
	md, err := m.aadMetadataFromCache(ctx, authorityInfo)
	if err != nil {
		// not in the cache, retrieve it
		md, err = m.aadMetadata(ctx, authorityInfo)
	}
	return md, err
}

func (m *PartitionedManager) aadMetadataFromCache(ctx context.Context, authorityInfo authority.Info) (authority.InstanceDiscoveryMetadata, error) {
	m.aadCacheMu.RLock()
	defer m.aadCacheMu.RUnlock()
	metadata, ok := m.aadCache[authorityInfo.Host]
	if ok {
		return metadata, nil
	}
	return metadata, errors.New("not found")
}

func (m *PartitionedManager) aadMetadata(ctx context.Context, authorityInfo authority.Info) (authority.InstanceDiscoveryMetadata, error) {
	discoveryResponse, err := m.requests.AADInstanceDiscovery(ctx, authorityInfo)
	if err != nil {
		return authority.InstanceDiscoveryMetadata{}, err
	}

	m.aadCacheMu.Lock()
	defer m.aadCacheMu.Unlock()

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

func (m *PartitionedManager) readAccessToken(envAliases []string, realm, clientID, userAssertionHash string, scopes []string, partitionKey string) (AccessToken, error) {
	m.contractMu.RLock()
	defer m.contractMu.RUnlock()
	if accessTokens, ok := m.contract.AccessTokensPartition[partitionKey]; ok {
		// TODO: linear search (over a map no less) is slow for a large number (thousands) of tokens.
		// this shows up as the dominating node in a profile. for real-world scenarios this likely isn't
		// an issue, however if it does become a problem then we know where to look.
		for _, at := range accessTokens {
			if at.Realm == realm && at.ClientID == clientID && at.UserAssertionHash == userAssertionHash {
				if checkAlias(at.Environment, envAliases) {
					if isMatchingScopes(scopes, at.Scopes) {
						return at, nil
					}
				}
			}
		}
	}
	return AccessToken{}, fmt.Errorf("access token not found")
}

func (m *PartitionedManager) writeAccessToken(accessToken AccessToken, partitionKey string) error {
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	key := accessToken.Key()
	if m.contract.AccessTokensPartition[partitionKey] == nil {
		m.contract.AccessTokensPartition[partitionKey] = make(map[string]AccessToken)
	}
	m.contract.AccessTokensPartition[partitionKey][key] = accessToken
	return nil
}

func matchFamilyRefreshTokenObo(rt accesstokens.RefreshToken, userAssertionHash string, envAliases []string) bool {
	return rt.UserAssertionHash == userAssertionHash && checkAlias(rt.Environment, envAliases) && rt.FamilyID != ""
}

func matchClientIDRefreshTokenObo(rt accesstokens.RefreshToken, userAssertionHash string, envAliases []string, clientID string) bool {
	return rt.UserAssertionHash == userAssertionHash && checkAlias(rt.Environment, envAliases) && rt.ClientID == clientID
}

func (m *PartitionedManager) readRefreshToken(envAliases []string, familyID, clientID, userAssertionHash, partitionKey string) (accesstokens.RefreshToken, error) {
	byFamily := func(rt accesstokens.RefreshToken) bool {
		return matchFamilyRefreshTokenObo(rt, userAssertionHash, envAliases)
	}
	byClient := func(rt accesstokens.RefreshToken) bool {
		return matchClientIDRefreshTokenObo(rt, userAssertionHash, envAliases, clientID)
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
		for _, rt := range m.contract.RefreshTokensPartition[partitionKey] {
			if matcher(rt) {
				return rt, nil
			}
		}
	}

	return accesstokens.RefreshToken{}, fmt.Errorf("refresh token not found")
}

func (m *PartitionedManager) writeRefreshToken(refreshToken accesstokens.RefreshToken, partitionKey string) error {
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	key := refreshToken.Key()
	if m.contract.AccessTokensPartition[partitionKey] == nil {
		m.contract.RefreshTokensPartition[partitionKey] = make(map[string]accesstokens.RefreshToken)
	}
	m.contract.RefreshTokensPartition[partitionKey][key] = refreshToken
	return nil
}

func (m *PartitionedManager) readIDToken(envAliases []string, realm, clientID, userAssertionHash, partitionKey string) (IDToken, error) {
	m.contractMu.RLock()
	defer m.contractMu.RUnlock()
	for _, idt := range m.contract.IDTokensPartition[partitionKey] {
		if idt.Realm == realm && idt.ClientID == clientID && idt.UserAssertionHash == userAssertionHash {
			if checkAlias(idt.Environment, envAliases) {
				return idt, nil
			}
		}
	}
	return IDToken{}, fmt.Errorf("token not found")
}

func (m *PartitionedManager) writeIDToken(idToken IDToken, partitionKey string) error {
	key := idToken.Key()
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	if m.contract.IDTokensPartition[partitionKey] == nil {
		m.contract.IDTokensPartition[partitionKey] = make(map[string]IDToken)
	}
	m.contract.IDTokensPartition[partitionKey][key] = idToken
	return nil
}

func (m *PartitionedManager) readAccount(envAliases []string, realm, UserAssertionHash, partitionKey string) (shared.Account, error) {
	m.contractMu.RLock()
	defer m.contractMu.RUnlock()

	// You might ask why, if cache.Accounts is a map, we would loop through all of these instead of using a key.
	// We only use a map because the storage contract shared between all language implementations says use a map.
	// We can't change that. The other is because the keys are made using a specific "env", but here we are allowing
	// a match in multiple envs (envAlias). That means we either need to hash each possible keyand do the lookup
	// or just statically check.  Since the design is to have a storage.Manager per user, the amount of keys stored
	// is really low (say 2).  Each hash is more expensive than the entire iteration.
	for _, acc := range m.contract.AccountsPartition[partitionKey] {
		if checkAlias(acc.Environment, envAliases) && acc.UserAssertionHash == UserAssertionHash && acc.Realm == realm {
			return acc, nil
		}
	}
	return shared.Account{}, fmt.Errorf("account not found")
}

func (m *PartitionedManager) writeAccount(account shared.Account, partitionKey string) error {
	key := account.Key()
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	if m.contract.AccountsPartition[partitionKey] == nil {
		m.contract.AccountsPartition[partitionKey] = make(map[string]shared.Account)
	}
	m.contract.AccountsPartition[partitionKey][key] = account
	return nil
}

func (m *PartitionedManager) readAppMetaData(envAliases []string, clientID string) (AppMetaData, error) {
	m.contractMu.RLock()
	defer m.contractMu.RUnlock()

	for _, app := range m.contract.AppMetaData {
		if checkAlias(app.Environment, envAliases) && app.ClientID == clientID {
			return app, nil
		}
	}
	return AppMetaData{}, fmt.Errorf("not found")
}

func (m *PartitionedManager) writeAppMetaData(AppMetaData AppMetaData) error {
	key := AppMetaData.Key()
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	m.contract.AppMetaData[key] = AppMetaData
	return nil
}

// update updates the internal cache object. This is for use in tests, other uses are not
// supported.
func (m *PartitionedManager) update(cache *InMemoryContract) {
	m.contractMu.Lock()
	defer m.contractMu.Unlock()
	m.contract = cache
}

// Marshal implements cache.Marshaler.
func (m *PartitionedManager) Marshal() ([]byte, error) {
	return json.Marshal(m.contract)
}

// Unmarshal implements cache.Unmarshaler.
func (m *PartitionedManager) Unmarshal(b []byte) error {
	m.contractMu.Lock()
	defer m.contractMu.Unlock()

	contract := NewInMemoryContract()

	err := json.Unmarshal(b, contract)
	if err != nil {
		return err
	}

	m.contract = contract

	return nil
}

func getPartitionKeyAccessToken(item AccessToken) string {
	if item.UserAssertionHash != "" {
		return item.UserAssertionHash
	}
	return item.HomeAccountID
}

func getPartitionKeyRefreshToken(item accesstokens.RefreshToken) string {
	if item.UserAssertionHash != "" {
		return item.UserAssertionHash
	}
	return item.HomeAccountID
}

func getPartitionKeyIDToken(item IDToken) string {
	return item.HomeAccountID
}

func getPartitionKeyAccount(item shared.Account) string {
	return item.HomeAccountID
}

func getPartitionKeyIDTokenRead(item AccessToken) string {
	return item.HomeAccountID
}
