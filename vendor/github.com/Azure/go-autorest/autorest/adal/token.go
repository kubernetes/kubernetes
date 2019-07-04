package adal

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha1"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/Azure/go-autorest/autorest/date"
	"github.com/Azure/go-autorest/version"
	"github.com/dgrijalva/jwt-go"
)

const (
	defaultRefresh = 5 * time.Minute

	// OAuthGrantTypeDeviceCode is the "grant_type" identifier used in device flow
	OAuthGrantTypeDeviceCode = "device_code"

	// OAuthGrantTypeClientCredentials is the "grant_type" identifier used in credential flows
	OAuthGrantTypeClientCredentials = "client_credentials"

	// OAuthGrantTypeUserPass is the "grant_type" identifier used in username and password auth flows
	OAuthGrantTypeUserPass = "password"

	// OAuthGrantTypeRefreshToken is the "grant_type" identifier used in refresh token flows
	OAuthGrantTypeRefreshToken = "refresh_token"

	// OAuthGrantTypeAuthorizationCode is the "grant_type" identifier used in authorization code flows
	OAuthGrantTypeAuthorizationCode = "authorization_code"

	// metadataHeader is the header required by MSI extension
	metadataHeader = "Metadata"

	// msiEndpoint is the well known endpoint for getting MSI authentications tokens
	msiEndpoint = "http://169.254.169.254/metadata/identity/oauth2/token"

	// the default number of attempts to refresh an MSI authentication token
	defaultMaxMSIRefreshAttempts = 5
)

// OAuthTokenProvider is an interface which should be implemented by an access token retriever
type OAuthTokenProvider interface {
	OAuthToken() string
}

// TokenRefreshError is an interface used by errors returned during token refresh.
type TokenRefreshError interface {
	error
	Response() *http.Response
}

// Refresher is an interface for token refresh functionality
type Refresher interface {
	Refresh() error
	RefreshExchange(resource string) error
	EnsureFresh() error
}

// RefresherWithContext is an interface for token refresh functionality
type RefresherWithContext interface {
	RefreshWithContext(ctx context.Context) error
	RefreshExchangeWithContext(ctx context.Context, resource string) error
	EnsureFreshWithContext(ctx context.Context) error
}

// TokenRefreshCallback is the type representing callbacks that will be called after
// a successful token refresh
type TokenRefreshCallback func(Token) error

// Token encapsulates the access token used to authorize Azure requests.
// https://docs.microsoft.com/en-us/azure/active-directory/develop/v1-oauth2-client-creds-grant-flow#service-to-service-access-token-response
type Token struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`

	ExpiresIn json.Number `json:"expires_in"`
	ExpiresOn json.Number `json:"expires_on"`
	NotBefore json.Number `json:"not_before"`

	Resource string `json:"resource"`
	Type     string `json:"token_type"`
}

func newToken() Token {
	return Token{
		ExpiresIn: "0",
		ExpiresOn: "0",
		NotBefore: "0",
	}
}

// IsZero returns true if the token object is zero-initialized.
func (t Token) IsZero() bool {
	return t == Token{}
}

// Expires returns the time.Time when the Token expires.
func (t Token) Expires() time.Time {
	s, err := t.ExpiresOn.Float64()
	if err != nil {
		s = -3600
	}

	expiration := date.NewUnixTimeFromSeconds(s)

	return time.Time(expiration).UTC()
}

// IsExpired returns true if the Token is expired, false otherwise.
func (t Token) IsExpired() bool {
	return t.WillExpireIn(0)
}

// WillExpireIn returns true if the Token will expire after the passed time.Duration interval
// from now, false otherwise.
func (t Token) WillExpireIn(d time.Duration) bool {
	return !t.Expires().After(time.Now().Add(d))
}

//OAuthToken return the current access token
func (t *Token) OAuthToken() string {
	return t.AccessToken
}

// ServicePrincipalSecret is an interface that allows various secret mechanism to fill the form
// that is submitted when acquiring an oAuth token.
type ServicePrincipalSecret interface {
	SetAuthenticationValues(spt *ServicePrincipalToken, values *url.Values) error
}

// ServicePrincipalNoSecret represents a secret type that contains no secret
// meaning it is not valid for fetching a fresh token. This is used by Manual
type ServicePrincipalNoSecret struct {
}

// SetAuthenticationValues is a method of the interface ServicePrincipalSecret
// It only returns an error for the ServicePrincipalNoSecret type
func (noSecret *ServicePrincipalNoSecret) SetAuthenticationValues(spt *ServicePrincipalToken, v *url.Values) error {
	return fmt.Errorf("Manually created ServicePrincipalToken does not contain secret material to retrieve a new access token")
}

// MarshalJSON implements the json.Marshaler interface.
func (noSecret ServicePrincipalNoSecret) MarshalJSON() ([]byte, error) {
	type tokenType struct {
		Type string `json:"type"`
	}
	return json.Marshal(tokenType{
		Type: "ServicePrincipalNoSecret",
	})
}

// ServicePrincipalTokenSecret implements ServicePrincipalSecret for client_secret type authorization.
type ServicePrincipalTokenSecret struct {
	ClientSecret string `json:"value"`
}

// SetAuthenticationValues is a method of the interface ServicePrincipalSecret.
// It will populate the form submitted during oAuth Token Acquisition using the client_secret.
func (tokenSecret *ServicePrincipalTokenSecret) SetAuthenticationValues(spt *ServicePrincipalToken, v *url.Values) error {
	v.Set("client_secret", tokenSecret.ClientSecret)
	return nil
}

// MarshalJSON implements the json.Marshaler interface.
func (tokenSecret ServicePrincipalTokenSecret) MarshalJSON() ([]byte, error) {
	type tokenType struct {
		Type  string `json:"type"`
		Value string `json:"value"`
	}
	return json.Marshal(tokenType{
		Type:  "ServicePrincipalTokenSecret",
		Value: tokenSecret.ClientSecret,
	})
}

// ServicePrincipalCertificateSecret implements ServicePrincipalSecret for generic RSA cert auth with signed JWTs.
type ServicePrincipalCertificateSecret struct {
	Certificate *x509.Certificate
	PrivateKey  *rsa.PrivateKey
}

// SignJwt returns the JWT signed with the certificate's private key.
func (secret *ServicePrincipalCertificateSecret) SignJwt(spt *ServicePrincipalToken) (string, error) {
	hasher := sha1.New()
	_, err := hasher.Write(secret.Certificate.Raw)
	if err != nil {
		return "", err
	}

	thumbprint := base64.URLEncoding.EncodeToString(hasher.Sum(nil))

	// The jti (JWT ID) claim provides a unique identifier for the JWT.
	jti := make([]byte, 20)
	_, err = rand.Read(jti)
	if err != nil {
		return "", err
	}

	token := jwt.New(jwt.SigningMethodRS256)
	token.Header["x5t"] = thumbprint
	x5c := []string{base64.StdEncoding.EncodeToString(secret.Certificate.Raw)}
	token.Header["x5c"] = x5c
	token.Claims = jwt.MapClaims{
		"aud": spt.inner.OauthConfig.TokenEndpoint.String(),
		"iss": spt.inner.ClientID,
		"sub": spt.inner.ClientID,
		"jti": base64.URLEncoding.EncodeToString(jti),
		"nbf": time.Now().Unix(),
		"exp": time.Now().Add(time.Hour * 24).Unix(),
	}

	signedString, err := token.SignedString(secret.PrivateKey)
	return signedString, err
}

// SetAuthenticationValues is a method of the interface ServicePrincipalSecret.
// It will populate the form submitted during oAuth Token Acquisition using a JWT signed with a certificate.
func (secret *ServicePrincipalCertificateSecret) SetAuthenticationValues(spt *ServicePrincipalToken, v *url.Values) error {
	jwt, err := secret.SignJwt(spt)
	if err != nil {
		return err
	}

	v.Set("client_assertion", jwt)
	v.Set("client_assertion_type", "urn:ietf:params:oauth:client-assertion-type:jwt-bearer")
	return nil
}

// MarshalJSON implements the json.Marshaler interface.
func (secret ServicePrincipalCertificateSecret) MarshalJSON() ([]byte, error) {
	return nil, errors.New("marshalling ServicePrincipalCertificateSecret is not supported")
}

// ServicePrincipalMSISecret implements ServicePrincipalSecret for machines running the MSI Extension.
type ServicePrincipalMSISecret struct {
}

// SetAuthenticationValues is a method of the interface ServicePrincipalSecret.
func (msiSecret *ServicePrincipalMSISecret) SetAuthenticationValues(spt *ServicePrincipalToken, v *url.Values) error {
	return nil
}

// MarshalJSON implements the json.Marshaler interface.
func (msiSecret ServicePrincipalMSISecret) MarshalJSON() ([]byte, error) {
	return nil, errors.New("marshalling ServicePrincipalMSISecret is not supported")
}

// ServicePrincipalUsernamePasswordSecret implements ServicePrincipalSecret for username and password auth.
type ServicePrincipalUsernamePasswordSecret struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

// SetAuthenticationValues is a method of the interface ServicePrincipalSecret.
func (secret *ServicePrincipalUsernamePasswordSecret) SetAuthenticationValues(spt *ServicePrincipalToken, v *url.Values) error {
	v.Set("username", secret.Username)
	v.Set("password", secret.Password)
	return nil
}

// MarshalJSON implements the json.Marshaler interface.
func (secret ServicePrincipalUsernamePasswordSecret) MarshalJSON() ([]byte, error) {
	type tokenType struct {
		Type     string `json:"type"`
		Username string `json:"username"`
		Password string `json:"password"`
	}
	return json.Marshal(tokenType{
		Type:     "ServicePrincipalUsernamePasswordSecret",
		Username: secret.Username,
		Password: secret.Password,
	})
}

// ServicePrincipalAuthorizationCodeSecret implements ServicePrincipalSecret for authorization code auth.
type ServicePrincipalAuthorizationCodeSecret struct {
	ClientSecret      string `json:"value"`
	AuthorizationCode string `json:"authCode"`
	RedirectURI       string `json:"redirect"`
}

// SetAuthenticationValues is a method of the interface ServicePrincipalSecret.
func (secret *ServicePrincipalAuthorizationCodeSecret) SetAuthenticationValues(spt *ServicePrincipalToken, v *url.Values) error {
	v.Set("code", secret.AuthorizationCode)
	v.Set("client_secret", secret.ClientSecret)
	v.Set("redirect_uri", secret.RedirectURI)
	return nil
}

// MarshalJSON implements the json.Marshaler interface.
func (secret ServicePrincipalAuthorizationCodeSecret) MarshalJSON() ([]byte, error) {
	type tokenType struct {
		Type     string `json:"type"`
		Value    string `json:"value"`
		AuthCode string `json:"authCode"`
		Redirect string `json:"redirect"`
	}
	return json.Marshal(tokenType{
		Type:     "ServicePrincipalAuthorizationCodeSecret",
		Value:    secret.ClientSecret,
		AuthCode: secret.AuthorizationCode,
		Redirect: secret.RedirectURI,
	})
}

// ServicePrincipalToken encapsulates a Token created for a Service Principal.
type ServicePrincipalToken struct {
	inner            servicePrincipalToken
	refreshLock      *sync.RWMutex
	sender           Sender
	refreshCallbacks []TokenRefreshCallback
	// MaxMSIRefreshAttempts is the maximum number of attempts to refresh an MSI token.
	MaxMSIRefreshAttempts int
}

// MarshalTokenJSON returns the marshalled inner token.
func (spt ServicePrincipalToken) MarshalTokenJSON() ([]byte, error) {
	return json.Marshal(spt.inner.Token)
}

// SetRefreshCallbacks replaces any existing refresh callbacks with the specified callbacks.
func (spt *ServicePrincipalToken) SetRefreshCallbacks(callbacks []TokenRefreshCallback) {
	spt.refreshCallbacks = callbacks
}

// MarshalJSON implements the json.Marshaler interface.
func (spt ServicePrincipalToken) MarshalJSON() ([]byte, error) {
	return json.Marshal(spt.inner)
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (spt *ServicePrincipalToken) UnmarshalJSON(data []byte) error {
	// need to determine the token type
	raw := map[string]interface{}{}
	err := json.Unmarshal(data, &raw)
	if err != nil {
		return err
	}
	secret := raw["secret"].(map[string]interface{})
	switch secret["type"] {
	case "ServicePrincipalNoSecret":
		spt.inner.Secret = &ServicePrincipalNoSecret{}
	case "ServicePrincipalTokenSecret":
		spt.inner.Secret = &ServicePrincipalTokenSecret{}
	case "ServicePrincipalCertificateSecret":
		return errors.New("unmarshalling ServicePrincipalCertificateSecret is not supported")
	case "ServicePrincipalMSISecret":
		return errors.New("unmarshalling ServicePrincipalMSISecret is not supported")
	case "ServicePrincipalUsernamePasswordSecret":
		spt.inner.Secret = &ServicePrincipalUsernamePasswordSecret{}
	case "ServicePrincipalAuthorizationCodeSecret":
		spt.inner.Secret = &ServicePrincipalAuthorizationCodeSecret{}
	default:
		return fmt.Errorf("unrecognized token type '%s'", secret["type"])
	}
	err = json.Unmarshal(data, &spt.inner)
	if err != nil {
		return err
	}
	spt.refreshLock = &sync.RWMutex{}
	spt.sender = &http.Client{}
	return nil
}

// internal type used for marshalling/unmarshalling
type servicePrincipalToken struct {
	Token         Token                  `json:"token"`
	Secret        ServicePrincipalSecret `json:"secret"`
	OauthConfig   OAuthConfig            `json:"oauth"`
	ClientID      string                 `json:"clientID"`
	Resource      string                 `json:"resource"`
	AutoRefresh   bool                   `json:"autoRefresh"`
	RefreshWithin time.Duration          `json:"refreshWithin"`
}

func validateOAuthConfig(oac OAuthConfig) error {
	if oac.IsZero() {
		return fmt.Errorf("parameter 'oauthConfig' cannot be zero-initialized")
	}
	return nil
}

// NewServicePrincipalTokenWithSecret create a ServicePrincipalToken using the supplied ServicePrincipalSecret implementation.
func NewServicePrincipalTokenWithSecret(oauthConfig OAuthConfig, id string, resource string, secret ServicePrincipalSecret, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	if err := validateOAuthConfig(oauthConfig); err != nil {
		return nil, err
	}
	if err := validateStringParam(id, "id"); err != nil {
		return nil, err
	}
	if err := validateStringParam(resource, "resource"); err != nil {
		return nil, err
	}
	if secret == nil {
		return nil, fmt.Errorf("parameter 'secret' cannot be nil")
	}
	spt := &ServicePrincipalToken{
		inner: servicePrincipalToken{
			Token:         newToken(),
			OauthConfig:   oauthConfig,
			Secret:        secret,
			ClientID:      id,
			Resource:      resource,
			AutoRefresh:   true,
			RefreshWithin: defaultRefresh,
		},
		refreshLock:      &sync.RWMutex{},
		sender:           &http.Client{},
		refreshCallbacks: callbacks,
	}
	return spt, nil
}

// NewServicePrincipalTokenFromManualToken creates a ServicePrincipalToken using the supplied token
func NewServicePrincipalTokenFromManualToken(oauthConfig OAuthConfig, clientID string, resource string, token Token, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	if err := validateOAuthConfig(oauthConfig); err != nil {
		return nil, err
	}
	if err := validateStringParam(clientID, "clientID"); err != nil {
		return nil, err
	}
	if err := validateStringParam(resource, "resource"); err != nil {
		return nil, err
	}
	if token.IsZero() {
		return nil, fmt.Errorf("parameter 'token' cannot be zero-initialized")
	}
	spt, err := NewServicePrincipalTokenWithSecret(
		oauthConfig,
		clientID,
		resource,
		&ServicePrincipalNoSecret{},
		callbacks...)
	if err != nil {
		return nil, err
	}

	spt.inner.Token = token

	return spt, nil
}

// NewServicePrincipalTokenFromManualTokenSecret creates a ServicePrincipalToken using the supplied token and secret
func NewServicePrincipalTokenFromManualTokenSecret(oauthConfig OAuthConfig, clientID string, resource string, token Token, secret ServicePrincipalSecret, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	if err := validateOAuthConfig(oauthConfig); err != nil {
		return nil, err
	}
	if err := validateStringParam(clientID, "clientID"); err != nil {
		return nil, err
	}
	if err := validateStringParam(resource, "resource"); err != nil {
		return nil, err
	}
	if secret == nil {
		return nil, fmt.Errorf("parameter 'secret' cannot be nil")
	}
	if token.IsZero() {
		return nil, fmt.Errorf("parameter 'token' cannot be zero-initialized")
	}
	spt, err := NewServicePrincipalTokenWithSecret(
		oauthConfig,
		clientID,
		resource,
		secret,
		callbacks...)
	if err != nil {
		return nil, err
	}

	spt.inner.Token = token

	return spt, nil
}

// NewServicePrincipalToken creates a ServicePrincipalToken from the supplied Service Principal
// credentials scoped to the named resource.
func NewServicePrincipalToken(oauthConfig OAuthConfig, clientID string, secret string, resource string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	if err := validateOAuthConfig(oauthConfig); err != nil {
		return nil, err
	}
	if err := validateStringParam(clientID, "clientID"); err != nil {
		return nil, err
	}
	if err := validateStringParam(secret, "secret"); err != nil {
		return nil, err
	}
	if err := validateStringParam(resource, "resource"); err != nil {
		return nil, err
	}
	return NewServicePrincipalTokenWithSecret(
		oauthConfig,
		clientID,
		resource,
		&ServicePrincipalTokenSecret{
			ClientSecret: secret,
		},
		callbacks...,
	)
}

// NewServicePrincipalTokenFromCertificate creates a ServicePrincipalToken from the supplied pkcs12 bytes.
func NewServicePrincipalTokenFromCertificate(oauthConfig OAuthConfig, clientID string, certificate *x509.Certificate, privateKey *rsa.PrivateKey, resource string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	if err := validateOAuthConfig(oauthConfig); err != nil {
		return nil, err
	}
	if err := validateStringParam(clientID, "clientID"); err != nil {
		return nil, err
	}
	if err := validateStringParam(resource, "resource"); err != nil {
		return nil, err
	}
	if certificate == nil {
		return nil, fmt.Errorf("parameter 'certificate' cannot be nil")
	}
	if privateKey == nil {
		return nil, fmt.Errorf("parameter 'privateKey' cannot be nil")
	}
	return NewServicePrincipalTokenWithSecret(
		oauthConfig,
		clientID,
		resource,
		&ServicePrincipalCertificateSecret{
			PrivateKey:  privateKey,
			Certificate: certificate,
		},
		callbacks...,
	)
}

// NewServicePrincipalTokenFromUsernamePassword creates a ServicePrincipalToken from the username and password.
func NewServicePrincipalTokenFromUsernamePassword(oauthConfig OAuthConfig, clientID string, username string, password string, resource string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	if err := validateOAuthConfig(oauthConfig); err != nil {
		return nil, err
	}
	if err := validateStringParam(clientID, "clientID"); err != nil {
		return nil, err
	}
	if err := validateStringParam(username, "username"); err != nil {
		return nil, err
	}
	if err := validateStringParam(password, "password"); err != nil {
		return nil, err
	}
	if err := validateStringParam(resource, "resource"); err != nil {
		return nil, err
	}
	return NewServicePrincipalTokenWithSecret(
		oauthConfig,
		clientID,
		resource,
		&ServicePrincipalUsernamePasswordSecret{
			Username: username,
			Password: password,
		},
		callbacks...,
	)
}

// NewServicePrincipalTokenFromAuthorizationCode creates a ServicePrincipalToken from the
func NewServicePrincipalTokenFromAuthorizationCode(oauthConfig OAuthConfig, clientID string, clientSecret string, authorizationCode string, redirectURI string, resource string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {

	if err := validateOAuthConfig(oauthConfig); err != nil {
		return nil, err
	}
	if err := validateStringParam(clientID, "clientID"); err != nil {
		return nil, err
	}
	if err := validateStringParam(clientSecret, "clientSecret"); err != nil {
		return nil, err
	}
	if err := validateStringParam(authorizationCode, "authorizationCode"); err != nil {
		return nil, err
	}
	if err := validateStringParam(redirectURI, "redirectURI"); err != nil {
		return nil, err
	}
	if err := validateStringParam(resource, "resource"); err != nil {
		return nil, err
	}

	return NewServicePrincipalTokenWithSecret(
		oauthConfig,
		clientID,
		resource,
		&ServicePrincipalAuthorizationCodeSecret{
			ClientSecret:      clientSecret,
			AuthorizationCode: authorizationCode,
			RedirectURI:       redirectURI,
		},
		callbacks...,
	)
}

// GetMSIVMEndpoint gets the MSI endpoint on Virtual Machines.
func GetMSIVMEndpoint() (string, error) {
	return msiEndpoint, nil
}

// NewServicePrincipalTokenFromMSI creates a ServicePrincipalToken via the MSI VM Extension.
// It will use the system assigned identity when creating the token.
func NewServicePrincipalTokenFromMSI(msiEndpoint, resource string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	return newServicePrincipalTokenFromMSI(msiEndpoint, resource, nil, callbacks...)
}

// NewServicePrincipalTokenFromMSIWithUserAssignedID creates a ServicePrincipalToken via the MSI VM Extension.
// It will use the specified user assigned identity when creating the token.
func NewServicePrincipalTokenFromMSIWithUserAssignedID(msiEndpoint, resource string, userAssignedID string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	return newServicePrincipalTokenFromMSI(msiEndpoint, resource, &userAssignedID, callbacks...)
}

func newServicePrincipalTokenFromMSI(msiEndpoint, resource string, userAssignedID *string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	if err := validateStringParam(msiEndpoint, "msiEndpoint"); err != nil {
		return nil, err
	}
	if err := validateStringParam(resource, "resource"); err != nil {
		return nil, err
	}
	if userAssignedID != nil {
		if err := validateStringParam(*userAssignedID, "userAssignedID"); err != nil {
			return nil, err
		}
	}
	// We set the oauth config token endpoint to be MSI's endpoint
	msiEndpointURL, err := url.Parse(msiEndpoint)
	if err != nil {
		return nil, err
	}

	v := url.Values{}
	v.Set("resource", resource)
	v.Set("api-version", "2018-02-01")
	if userAssignedID != nil {
		v.Set("client_id", *userAssignedID)
	}
	msiEndpointURL.RawQuery = v.Encode()

	spt := &ServicePrincipalToken{
		inner: servicePrincipalToken{
			Token: newToken(),
			OauthConfig: OAuthConfig{
				TokenEndpoint: *msiEndpointURL,
			},
			Secret:        &ServicePrincipalMSISecret{},
			Resource:      resource,
			AutoRefresh:   true,
			RefreshWithin: defaultRefresh,
		},
		refreshLock:           &sync.RWMutex{},
		sender:                &http.Client{},
		refreshCallbacks:      callbacks,
		MaxMSIRefreshAttempts: defaultMaxMSIRefreshAttempts,
	}

	if userAssignedID != nil {
		spt.inner.ClientID = *userAssignedID
	}

	return spt, nil
}

// internal type that implements TokenRefreshError
type tokenRefreshError struct {
	message string
	resp    *http.Response
}

// Error implements the error interface which is part of the TokenRefreshError interface.
func (tre tokenRefreshError) Error() string {
	return tre.message
}

// Response implements the TokenRefreshError interface, it returns the raw HTTP response from the refresh operation.
func (tre tokenRefreshError) Response() *http.Response {
	return tre.resp
}

func newTokenRefreshError(message string, resp *http.Response) TokenRefreshError {
	return tokenRefreshError{message: message, resp: resp}
}

// EnsureFresh will refresh the token if it will expire within the refresh window (as set by
// RefreshWithin) and autoRefresh flag is on.  This method is safe for concurrent use.
func (spt *ServicePrincipalToken) EnsureFresh() error {
	return spt.EnsureFreshWithContext(context.Background())
}

// EnsureFreshWithContext will refresh the token if it will expire within the refresh window (as set by
// RefreshWithin) and autoRefresh flag is on.  This method is safe for concurrent use.
func (spt *ServicePrincipalToken) EnsureFreshWithContext(ctx context.Context) error {
	if spt.inner.AutoRefresh && spt.inner.Token.WillExpireIn(spt.inner.RefreshWithin) {
		// take the write lock then check to see if the token was already refreshed
		spt.refreshLock.Lock()
		defer spt.refreshLock.Unlock()
		if spt.inner.Token.WillExpireIn(spt.inner.RefreshWithin) {
			return spt.refreshInternal(ctx, spt.inner.Resource)
		}
	}
	return nil
}

// InvokeRefreshCallbacks calls any TokenRefreshCallbacks that were added to the SPT during initialization
func (spt *ServicePrincipalToken) InvokeRefreshCallbacks(token Token) error {
	if spt.refreshCallbacks != nil {
		for _, callback := range spt.refreshCallbacks {
			err := callback(spt.inner.Token)
			if err != nil {
				return fmt.Errorf("adal: TokenRefreshCallback handler failed. Error = '%v'", err)
			}
		}
	}
	return nil
}

// Refresh obtains a fresh token for the Service Principal.
// This method is not safe for concurrent use and should be syncrhonized.
func (spt *ServicePrincipalToken) Refresh() error {
	return spt.RefreshWithContext(context.Background())
}

// RefreshWithContext obtains a fresh token for the Service Principal.
// This method is not safe for concurrent use and should be syncrhonized.
func (spt *ServicePrincipalToken) RefreshWithContext(ctx context.Context) error {
	spt.refreshLock.Lock()
	defer spt.refreshLock.Unlock()
	return spt.refreshInternal(ctx, spt.inner.Resource)
}

// RefreshExchange refreshes the token, but for a different resource.
// This method is not safe for concurrent use and should be syncrhonized.
func (spt *ServicePrincipalToken) RefreshExchange(resource string) error {
	return spt.RefreshExchangeWithContext(context.Background(), resource)
}

// RefreshExchangeWithContext refreshes the token, but for a different resource.
// This method is not safe for concurrent use and should be syncrhonized.
func (spt *ServicePrincipalToken) RefreshExchangeWithContext(ctx context.Context, resource string) error {
	spt.refreshLock.Lock()
	defer spt.refreshLock.Unlock()
	return spt.refreshInternal(ctx, resource)
}

func (spt *ServicePrincipalToken) getGrantType() string {
	switch spt.inner.Secret.(type) {
	case *ServicePrincipalUsernamePasswordSecret:
		return OAuthGrantTypeUserPass
	case *ServicePrincipalAuthorizationCodeSecret:
		return OAuthGrantTypeAuthorizationCode
	default:
		return OAuthGrantTypeClientCredentials
	}
}

func isIMDS(u url.URL) bool {
	imds, err := url.Parse(msiEndpoint)
	if err != nil {
		return false
	}
	return u.Host == imds.Host && u.Path == imds.Path
}

func (spt *ServicePrincipalToken) refreshInternal(ctx context.Context, resource string) error {
	req, err := http.NewRequest(http.MethodPost, spt.inner.OauthConfig.TokenEndpoint.String(), nil)
	if err != nil {
		return fmt.Errorf("adal: Failed to build the refresh request. Error = '%v'", err)
	}
	req.Header.Add("User-Agent", version.UserAgent())
	req = req.WithContext(ctx)
	if !isIMDS(spt.inner.OauthConfig.TokenEndpoint) {
		v := url.Values{}
		v.Set("client_id", spt.inner.ClientID)
		v.Set("resource", resource)

		if spt.inner.Token.RefreshToken != "" {
			v.Set("grant_type", OAuthGrantTypeRefreshToken)
			v.Set("refresh_token", spt.inner.Token.RefreshToken)
			// web apps must specify client_secret when refreshing tokens
			// see https://docs.microsoft.com/en-us/azure/active-directory/develop/active-directory-protocols-oauth-code#refreshing-the-access-tokens
			if spt.getGrantType() == OAuthGrantTypeAuthorizationCode {
				err := spt.inner.Secret.SetAuthenticationValues(spt, &v)
				if err != nil {
					return err
				}
			}
		} else {
			v.Set("grant_type", spt.getGrantType())
			err := spt.inner.Secret.SetAuthenticationValues(spt, &v)
			if err != nil {
				return err
			}
		}

		s := v.Encode()
		body := ioutil.NopCloser(strings.NewReader(s))
		req.ContentLength = int64(len(s))
		req.Header.Set(contentType, mimeTypeFormPost)
		req.Body = body
	}

	if _, ok := spt.inner.Secret.(*ServicePrincipalMSISecret); ok {
		req.Method = http.MethodGet
		req.Header.Set(metadataHeader, "true")
	}

	var resp *http.Response
	if isIMDS(spt.inner.OauthConfig.TokenEndpoint) {
		resp, err = retryForIMDS(spt.sender, req, spt.MaxMSIRefreshAttempts)
	} else {
		resp, err = spt.sender.Do(req)
	}
	if err != nil {
		return newTokenRefreshError(fmt.Sprintf("adal: Failed to execute the refresh request. Error = '%v'", err), nil)
	}

	defer resp.Body.Close()
	rb, err := ioutil.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		if err != nil {
			return newTokenRefreshError(fmt.Sprintf("adal: Refresh request failed. Status Code = '%d'. Failed reading response body: %v", resp.StatusCode, err), resp)
		}
		return newTokenRefreshError(fmt.Sprintf("adal: Refresh request failed. Status Code = '%d'. Response body: %s", resp.StatusCode, string(rb)), resp)
	}

	// for the following error cases don't return a TokenRefreshError.  the operation succeeded
	// but some transient failure happened during deserialization.  by returning a generic error
	// the retry logic will kick in (we don't retry on TokenRefreshError).

	if err != nil {
		return fmt.Errorf("adal: Failed to read a new service principal token during refresh. Error = '%v'", err)
	}
	if len(strings.Trim(string(rb), " ")) == 0 {
		return fmt.Errorf("adal: Empty service principal token received during refresh")
	}
	var token Token
	err = json.Unmarshal(rb, &token)
	if err != nil {
		return fmt.Errorf("adal: Failed to unmarshal the service principal token during refresh. Error = '%v' JSON = '%s'", err, string(rb))
	}

	spt.inner.Token = token

	return spt.InvokeRefreshCallbacks(token)
}

// retry logic specific to retrieving a token from the IMDS endpoint
func retryForIMDS(sender Sender, req *http.Request, maxAttempts int) (resp *http.Response, err error) {
	// copied from client.go due to circular dependency
	retries := []int{
		http.StatusRequestTimeout,      // 408
		http.StatusTooManyRequests,     // 429
		http.StatusInternalServerError, // 500
		http.StatusBadGateway,          // 502
		http.StatusServiceUnavailable,  // 503
		http.StatusGatewayTimeout,      // 504
	}
	// extra retry status codes specific to IMDS
	retries = append(retries,
		http.StatusNotFound,
		http.StatusGone,
		// all remaining 5xx
		http.StatusNotImplemented,
		http.StatusHTTPVersionNotSupported,
		http.StatusVariantAlsoNegotiates,
		http.StatusInsufficientStorage,
		http.StatusLoopDetected,
		http.StatusNotExtended,
		http.StatusNetworkAuthenticationRequired)

	// see https://docs.microsoft.com/en-us/azure/active-directory/managed-service-identity/how-to-use-vm-token#retry-guidance

	const maxDelay time.Duration = 60 * time.Second

	attempt := 0
	delay := time.Duration(0)

	for attempt < maxAttempts {
		resp, err = sender.Do(req)
		// retry on temporary network errors, e.g. transient network failures.
		// if we don't receive a response then assume we can't connect to the
		// endpoint so we're likely not running on an Azure VM so don't retry.
		if (err != nil && !isTemporaryNetworkError(err)) || resp == nil || resp.StatusCode == http.StatusOK || !containsInt(retries, resp.StatusCode) {
			return
		}

		// perform exponential backoff with a cap.
		// must increment attempt before calculating delay.
		attempt++
		// the base value of 2 is the "delta backoff" as specified in the guidance doc
		delay += (time.Duration(math.Pow(2, float64(attempt))) * time.Second)
		if delay > maxDelay {
			delay = maxDelay
		}

		select {
		case <-time.After(delay):
			// intentionally left blank
		case <-req.Context().Done():
			err = req.Context().Err()
			return
		}
	}
	return
}

// returns true if the specified error is a temporary network error or false if it's not.
// if the error doesn't implement the net.Error interface the return value is true.
func isTemporaryNetworkError(err error) bool {
	if netErr, ok := err.(net.Error); !ok || (ok && netErr.Temporary()) {
		return true
	}
	return false
}

// returns true if slice ints contains the value n
func containsInt(ints []int, n int) bool {
	for _, i := range ints {
		if i == n {
			return true
		}
	}
	return false
}

// SetAutoRefresh enables or disables automatic refreshing of stale tokens.
func (spt *ServicePrincipalToken) SetAutoRefresh(autoRefresh bool) {
	spt.inner.AutoRefresh = autoRefresh
}

// SetRefreshWithin sets the interval within which if the token will expire, EnsureFresh will
// refresh the token.
func (spt *ServicePrincipalToken) SetRefreshWithin(d time.Duration) {
	spt.inner.RefreshWithin = d
	return
}

// SetSender sets the http.Client used when obtaining the Service Principal token. An
// undecorated http.Client is used by default.
func (spt *ServicePrincipalToken) SetSender(s Sender) { spt.sender = s }

// OAuthToken implements the OAuthTokenProvider interface.  It returns the current access token.
func (spt *ServicePrincipalToken) OAuthToken() string {
	spt.refreshLock.RLock()
	defer spt.refreshLock.RUnlock()
	return spt.inner.Token.OAuthToken()
}

// Token returns a copy of the current token.
func (spt *ServicePrincipalToken) Token() Token {
	spt.refreshLock.RLock()
	defer spt.refreshLock.RUnlock()
	return spt.inner.Token
}
