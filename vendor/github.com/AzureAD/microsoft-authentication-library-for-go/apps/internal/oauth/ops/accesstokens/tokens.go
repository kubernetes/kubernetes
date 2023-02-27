// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

package accesstokens

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"

	internalTime "github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/json/types/time"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth/ops/authority"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/shared"
)

// IDToken consists of all the information used to validate a user.
// https://docs.microsoft.com/azure/active-directory/develop/id-tokens .
type IDToken struct {
	PreferredUsername string `json:"preferred_username,omitempty"`
	GivenName         string `json:"given_name,omitempty"`
	FamilyName        string `json:"family_name,omitempty"`
	MiddleName        string `json:"middle_name,omitempty"`
	Name              string `json:"name,omitempty"`
	Oid               string `json:"oid,omitempty"`
	TenantID          string `json:"tid,omitempty"`
	Subject           string `json:"sub,omitempty"`
	UPN               string `json:"upn,omitempty"`
	Email             string `json:"email,omitempty"`
	AlternativeID     string `json:"alternative_id,omitempty"`
	Issuer            string `json:"iss,omitempty"`
	Audience          string `json:"aud,omitempty"`
	ExpirationTime    int64  `json:"exp,omitempty"`
	IssuedAt          int64  `json:"iat,omitempty"`
	NotBefore         int64  `json:"nbf,omitempty"`
	RawToken          string

	AdditionalFields map[string]interface{}
}

var null = []byte("null")

// UnmarshalJSON implements json.Unmarshaler.
func (i *IDToken) UnmarshalJSON(b []byte) error {
	if bytes.Equal(null, b) {
		return nil
	}

	// Because we have a custom unmarshaler, you
	// cannot directly call json.Unmarshal here. If you do, it will call this function
	// recursively until reach our recursion limit. We have to create a new type
	// that doesn't have this method in order to use json.Unmarshal.
	type idToken2 IDToken

	jwt := strings.Trim(string(b), `"`)
	jwtArr := strings.Split(jwt, ".")
	if len(jwtArr) < 2 {
		return errors.New("IDToken returned from server is invalid")
	}

	jwtPart := jwtArr[1]
	jwtDecoded, err := decodeJWT(jwtPart)
	if err != nil {
		return fmt.Errorf("unable to unmarshal IDToken, problem decoding JWT: %w", err)
	}

	token := idToken2{}
	err = json.Unmarshal(jwtDecoded, &token)
	if err != nil {
		return fmt.Errorf("unable to unmarshal IDToken: %w", err)
	}
	token.RawToken = jwt

	*i = IDToken(token)
	return nil
}

// IsZero indicates if the IDToken is the zero value.
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

// LocalAccountID extracts an account's local account ID from an ID token.
func (i IDToken) LocalAccountID() string {
	if i.Oid != "" {
		return i.Oid
	}
	return i.Subject
}

// jwtDecoder is provided to allow tests to provide their own.
var jwtDecoder = decodeJWT

// ClientInfo is used to create a Home Account ID for an account.
type ClientInfo struct {
	UID  string `json:"uid"`
	UTID string `json:"utid"`

	AdditionalFields map[string]interface{}
}

// UnmarshalJSON implements json.Unmarshaler.s
func (c *ClientInfo) UnmarshalJSON(b []byte) error {
	s := strings.Trim(string(b), `"`)
	// Client info may be empty in some flows, e.g. certificate exchange.
	if len(s) == 0 {
		return nil
	}

	// Because we have a custom unmarshaler, you
	// cannot directly call json.Unmarshal here. If you do, it will call this function
	// recursively until reach our recursion limit. We have to create a new type
	// that doesn't have this method in order to use json.Unmarshal.
	type clientInfo2 ClientInfo

	raw, err := jwtDecoder(s)
	if err != nil {
		return fmt.Errorf("TokenResponse client_info field had JWT decode error: %w", err)
	}

	var c2 clientInfo2

	err = json.Unmarshal(raw, &c2)
	if err != nil {
		return fmt.Errorf("was unable to unmarshal decoded JWT in TokenRespone to ClientInfo: %w", err)
	}

	*c = ClientInfo(c2)
	return nil
}

// HomeAccountID creates the home account ID.
func (c ClientInfo) HomeAccountID() string {
	if c.UID == "" {
		return ""
	} else if c.UTID == "" {
		return fmt.Sprintf("%s.%s", c.UID, c.UID)
	} else {
		return fmt.Sprintf("%s.%s", c.UID, c.UTID)
	}
}

// Scopes represents scopes in a TokenResponse.
type Scopes struct {
	Slice []string
}

// UnmarshalJSON implements json.Unmarshal.
func (s *Scopes) UnmarshalJSON(b []byte) error {
	str := strings.Trim(string(b), `"`)
	if len(str) == 0 {
		return nil
	}
	sl := strings.Split(str, " ")
	s.Slice = sl
	return nil
}

// TokenResponse is the information that is returned from a token endpoint during a token acquisition flow.
type TokenResponse struct {
	authority.OAuthResponseBase

	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`

	FamilyID       string                    `json:"foci"`
	IDToken        IDToken                   `json:"id_token"`
	ClientInfo     ClientInfo                `json:"client_info"`
	ExpiresOn      internalTime.DurationTime `json:"expires_in"`
	ExtExpiresOn   internalTime.DurationTime `json:"ext_expires_in"`
	GrantedScopes  Scopes                    `json:"scope"`
	DeclinedScopes []string                  // This is derived

	AdditionalFields map[string]interface{}

	scopesComputed bool
}

// ComputeScope computes the final scopes based on what was granted by the server and
// what our AuthParams were from the authority server. Per OAuth spec, if no scopes are returned, the response should be treated as if all scopes were granted
// This behavior can be observed in client assertion flows, but can happen at any time, this check ensures we treat
// those special responses properly Link to spec: https://tools.ietf.org/html/rfc6749#section-3.3
func (tr *TokenResponse) ComputeScope(authParams authority.AuthParams) {
	if len(tr.GrantedScopes.Slice) == 0 {
		tr.GrantedScopes = Scopes{Slice: authParams.Scopes}
	} else {
		tr.DeclinedScopes = findDeclinedScopes(authParams.Scopes, tr.GrantedScopes.Slice)
	}
	tr.scopesComputed = true
}

// Validate validates the TokenResponse has basic valid values. It must be called
// after ComputeScopes() is called.
func (tr *TokenResponse) Validate() error {
	if tr.Error != "" {
		return fmt.Errorf("%s: %s", tr.Error, tr.ErrorDescription)
	}

	if tr.AccessToken == "" {
		return errors.New("response is missing access_token")
	}

	if !tr.scopesComputed {
		return fmt.Errorf("TokenResponse hasn't had ScopesComputed() called")
	}
	return nil
}

func (tr *TokenResponse) CacheKey(authParams authority.AuthParams) string {
	if authParams.AuthorizationType == authority.ATOnBehalfOf {
		return authParams.AssertionHash()
	}
	if authParams.AuthorizationType == authority.ATClientCredentials {
		return authParams.AppKey()
	}
	if authParams.IsConfidentialClient || authParams.AuthorizationType == authority.ATRefreshToken {
		return tr.ClientInfo.HomeAccountID()
	}
	return ""
}

func findDeclinedScopes(requestedScopes []string, grantedScopes []string) []string {
	declined := []string{}
	grantedMap := map[string]bool{}
	for _, s := range grantedScopes {
		grantedMap[strings.ToLower(s)] = true
	}
	// Comparing the requested scopes with the granted scopes to see if there are any scopes that have been declined.
	for _, r := range requestedScopes {
		if !grantedMap[strings.ToLower(r)] {
			declined = append(declined, r)
		}
	}
	return declined
}

// decodeJWT decodes a JWT and converts it to a byte array representing a JSON object
// JWT has headers and payload base64url encoded without padding
// https://tools.ietf.org/html/rfc7519#section-3 and
// https://tools.ietf.org/html/rfc7515#section-2
func decodeJWT(data string) ([]byte, error) {
	// https://tools.ietf.org/html/rfc7515#appendix-C
	return base64.RawURLEncoding.DecodeString(data)
}

// RefreshToken is the JSON representation of a MSAL refresh token for encoding to storage.
type RefreshToken struct {
	HomeAccountID     string `json:"home_account_id,omitempty"`
	Environment       string `json:"environment,omitempty"`
	CredentialType    string `json:"credential_type,omitempty"`
	ClientID          string `json:"client_id,omitempty"`
	FamilyID          string `json:"family_id,omitempty"`
	Secret            string `json:"secret,omitempty"`
	Realm             string `json:"realm,omitempty"`
	Target            string `json:"target,omitempty"`
	UserAssertionHash string `json:"user_assertion_hash,omitempty"`

	AdditionalFields map[string]interface{}
}

// NewRefreshToken is the constructor for RefreshToken.
func NewRefreshToken(homeID, env, clientID, refreshToken, familyID string) RefreshToken {
	return RefreshToken{
		HomeAccountID:  homeID,
		Environment:    env,
		CredentialType: "RefreshToken",
		ClientID:       clientID,
		FamilyID:       familyID,
		Secret:         refreshToken,
	}
}

// Key outputs the key that can be used to uniquely look up this entry in a map.
func (rt RefreshToken) Key() string {
	var fourth = rt.FamilyID
	if fourth == "" {
		fourth = rt.ClientID
	}

	return strings.Join(
		[]string{rt.HomeAccountID, rt.Environment, rt.CredentialType, fourth},
		shared.CacheKeySeparator,
	)
}

func (rt RefreshToken) GetSecret() string {
	return rt.Secret
}

// DeviceCodeResult stores the response from the STS device code endpoint.
type DeviceCodeResult struct {
	// UserCode is the code the user needs to provide when authentication at the verification URI.
	UserCode string
	// DeviceCode is the code used in the access token request.
	DeviceCode string
	// VerificationURL is the the URL where user can authenticate.
	VerificationURL string
	// ExpiresOn is the expiration time of device code in seconds.
	ExpiresOn time.Time
	// Interval is the interval at which the STS should be polled at.
	Interval int
	// Message is the message which should be displayed to the user.
	Message string
	// ClientID is the UUID issued by the authorization server for your application.
	ClientID string
	// Scopes is the OpenID scopes used to request access a protected API.
	Scopes []string
}

// NewDeviceCodeResult creates a DeviceCodeResult instance.
func NewDeviceCodeResult(userCode, deviceCode, verificationURL string, expiresOn time.Time, interval int, message, clientID string, scopes []string) DeviceCodeResult {
	return DeviceCodeResult{userCode, deviceCode, verificationURL, expiresOn, interval, message, clientID, scopes}
}

func (dcr DeviceCodeResult) String() string {
	return fmt.Sprintf("UserCode: (%v)\nDeviceCode: (%v)\nURL: (%v)\nMessage: (%v)\n", dcr.UserCode, dcr.DeviceCode, dcr.VerificationURL, dcr.Message)

}
