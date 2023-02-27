// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

package storage

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"

	internalTime "github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/json/types/time"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth/ops/accesstokens"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/shared"
)

// Contract is the JSON structure that is written to any storage medium when serializing
// the internal cache. This design is shared between MSAL versions in many languages.
// This cannot be changed without design that includes other SDKs.
type Contract struct {
	AccessTokens  map[string]AccessToken               `json:"AccessToken,omitempty"`
	RefreshTokens map[string]accesstokens.RefreshToken `json:"RefreshToken,omitempty"`
	IDTokens      map[string]IDToken                   `json:"IdToken,omitempty"`
	Accounts      map[string]shared.Account            `json:"Account,omitempty"`
	AppMetaData   map[string]AppMetaData               `json:"AppMetadata,omitempty"`

	AdditionalFields map[string]interface{}
}

// Contract is the JSON structure that is written to any storage medium when serializing
// the internal cache. This design is shared between MSAL versions in many languages.
// This cannot be changed without design that includes other SDKs.
type InMemoryContract struct {
	AccessTokensPartition  map[string]map[string]AccessToken
	RefreshTokensPartition map[string]map[string]accesstokens.RefreshToken
	IDTokensPartition      map[string]map[string]IDToken
	AccountsPartition      map[string]map[string]shared.Account
	AppMetaData            map[string]AppMetaData
}

// NewContract is the constructor for Contract.
func NewInMemoryContract() *InMemoryContract {
	return &InMemoryContract{
		AccessTokensPartition:  map[string]map[string]AccessToken{},
		RefreshTokensPartition: map[string]map[string]accesstokens.RefreshToken{},
		IDTokensPartition:      map[string]map[string]IDToken{},
		AccountsPartition:      map[string]map[string]shared.Account{},
		AppMetaData:            map[string]AppMetaData{},
	}
}

// NewContract is the constructor for Contract.
func NewContract() *Contract {
	return &Contract{
		AccessTokens:     map[string]AccessToken{},
		RefreshTokens:    map[string]accesstokens.RefreshToken{},
		IDTokens:         map[string]IDToken{},
		Accounts:         map[string]shared.Account{},
		AppMetaData:      map[string]AppMetaData{},
		AdditionalFields: map[string]interface{}{},
	}
}

// AccessToken is the JSON representation of a MSAL access token for encoding to storage.
type AccessToken struct {
	HomeAccountID     string            `json:"home_account_id,omitempty"`
	Environment       string            `json:"environment,omitempty"`
	Realm             string            `json:"realm,omitempty"`
	CredentialType    string            `json:"credential_type,omitempty"`
	ClientID          string            `json:"client_id,omitempty"`
	Secret            string            `json:"secret,omitempty"`
	Scopes            string            `json:"target,omitempty"`
	ExpiresOn         internalTime.Unix `json:"expires_on,omitempty"`
	ExtendedExpiresOn internalTime.Unix `json:"extended_expires_on,omitempty"`
	CachedAt          internalTime.Unix `json:"cached_at,omitempty"`
	UserAssertionHash string            `json:"user_assertion_hash,omitempty"`

	AdditionalFields map[string]interface{}
}

// NewAccessToken is the constructor for AccessToken.
func NewAccessToken(homeID, env, realm, clientID string, cachedAt, expiresOn, extendedExpiresOn time.Time, scopes, token string) AccessToken {
	return AccessToken{
		HomeAccountID:     homeID,
		Environment:       env,
		Realm:             realm,
		CredentialType:    "AccessToken",
		ClientID:          clientID,
		Secret:            token,
		Scopes:            scopes,
		CachedAt:          internalTime.Unix{T: cachedAt.UTC()},
		ExpiresOn:         internalTime.Unix{T: expiresOn.UTC()},
		ExtendedExpiresOn: internalTime.Unix{T: extendedExpiresOn.UTC()},
	}
}

// Key outputs the key that can be used to uniquely look up this entry in a map.
func (a AccessToken) Key() string {
	return strings.Join(
		[]string{a.HomeAccountID, a.Environment, a.CredentialType, a.ClientID, a.Realm, a.Scopes},
		shared.CacheKeySeparator,
	)
}

// FakeValidate enables tests to fake access token validation
var FakeValidate func(AccessToken) error

// Validate validates that this AccessToken can be used.
func (a AccessToken) Validate() error {
	if FakeValidate != nil {
		return FakeValidate(a)
	}
	if a.CachedAt.T.After(time.Now()) {
		return errors.New("access token isn't valid, it was cached at a future time")
	}
	if a.ExpiresOn.T.Before(time.Now().Add(5 * time.Minute)) {
		return fmt.Errorf("access token is expired")
	}
	if a.CachedAt.T.IsZero() {
		return fmt.Errorf("access token does not have CachedAt set")
	}
	return nil
}

// IDToken is the JSON representation of an MSAL id token for encoding to storage.
type IDToken struct {
	HomeAccountID     string `json:"home_account_id,omitempty"`
	Environment       string `json:"environment,omitempty"`
	Realm             string `json:"realm,omitempty"`
	CredentialType    string `json:"credential_type,omitempty"`
	ClientID          string `json:"client_id,omitempty"`
	Secret            string `json:"secret,omitempty"`
	UserAssertionHash string `json:"user_assertion_hash,omitempty"`
	AdditionalFields  map[string]interface{}
}

// IsZero determines if IDToken is the zero value.
func (i IDToken) IsZero() bool {
	v := reflect.ValueOf(i)
	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		if !field.IsZero() {
			switch field.Kind() {
			case reflect.Map, reflect.Slice:
				if field.Len() == 0 {
					continue
				}
			}
			return false
		}
	}
	return true
}

// NewIDToken is the constructor for IDToken.
func NewIDToken(homeID, env, realm, clientID, idToken string) IDToken {
	return IDToken{
		HomeAccountID:  homeID,
		Environment:    env,
		Realm:          realm,
		CredentialType: "IDToken",
		ClientID:       clientID,
		Secret:         idToken,
	}
}

// Key outputs the key that can be used to uniquely look up this entry in a map.
func (id IDToken) Key() string {
	return strings.Join(
		[]string{id.HomeAccountID, id.Environment, id.CredentialType, id.ClientID, id.Realm},
		shared.CacheKeySeparator,
	)
}

// AppMetaData is the JSON representation of application metadata for encoding to storage.
type AppMetaData struct {
	FamilyID    string `json:"family_id,omitempty"`
	ClientID    string `json:"client_id,omitempty"`
	Environment string `json:"environment,omitempty"`

	AdditionalFields map[string]interface{}
}

// NewAppMetaData is the constructor for AppMetaData.
func NewAppMetaData(familyID, clientID, environment string) AppMetaData {
	return AppMetaData{
		FamilyID:    familyID,
		ClientID:    clientID,
		Environment: environment,
	}
}

// Key outputs the key that can be used to uniquely look up this entry in a map.
func (a AppMetaData) Key() string {
	return strings.Join(
		[]string{"AppMetaData", a.Environment, a.ClientID},
		shared.CacheKeySeparator,
	)
}
