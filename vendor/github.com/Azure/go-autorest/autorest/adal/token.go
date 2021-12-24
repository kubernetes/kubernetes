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
	"io"
	"io/ioutil"
	"math"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Azure/go-autorest/autorest/date"
	"github.com/Azure/go-autorest/logger"
	"github.com/golang-jwt/jwt/v4"
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

	// the API version to use for the MSI endpoint
	msiAPIVersion = "2018-02-01"

	// the default number of attempts to refresh an MSI authentication token
	defaultMaxMSIRefreshAttempts = 5

	// asMSIEndpointEnv is the environment variable used to store the endpoint on App Service and Functions
	msiEndpointEnv = "MSI_ENDPOINT"

	// asMSISecretEnv is the environment variable used to store the request secret on App Service and Functions
	msiSecretEnv = "MSI_SECRET"

	// the API version to use for the legacy App Service MSI endpoint
	appServiceAPIVersion2017 = "2017-09-01"

	// secret header used when authenticating against app service MSI endpoint
	secretHeader = "Secret"

	// the format for expires_on in UTC with AM/PM
	expiresOnDateFormatPM = "1/2/2006 15:04:05 PM +00:00"

	// the format for expires_on in UTC without AM/PM
	expiresOnDateFormat = "1/2/2006 15:04:05 +00:00"
)

// OAuthTokenProvider is an interface which should be implemented by an access token retriever
type OAuthTokenProvider interface {
	OAuthToken() string
}

// MultitenantOAuthTokenProvider provides tokens used for multi-tenant authorization.
type MultitenantOAuthTokenProvider interface {
	PrimaryOAuthToken() string
	AuxiliaryOAuthTokens() []string
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

// TokenRefresh is a type representing a custom callback to refresh a token
type TokenRefresh func(ctx context.Context, resource string) (*Token, error)

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
		"exp": time.Now().Add(24 * time.Hour).Unix(),
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
	msiType          msiType
	clientResourceID string
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
	inner             servicePrincipalToken
	refreshLock       *sync.RWMutex
	sender            Sender
	customRefreshFunc TokenRefresh
	refreshCallbacks  []TokenRefreshCallback
	// MaxMSIRefreshAttempts is the maximum number of attempts to refresh an MSI token.
	// Settings this to a value less than 1 will use the default value.
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

// SetCustomRefreshFunc sets a custom refresh function used to refresh the token.
func (spt *ServicePrincipalToken) SetCustomRefreshFunc(customRefreshFunc TokenRefresh) {
	spt.customRefreshFunc = customRefreshFunc
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
	// Don't override the refreshLock or the sender if those have been already set.
	if spt.refreshLock == nil {
		spt.refreshLock = &sync.RWMutex{}
	}
	if spt.sender == nil {
		spt.sender = sender()
	}
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
		sender:           sender(),
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

type msiType int

const (
	msiTypeUnavailable msiType = iota
	msiTypeAppServiceV20170901
	msiTypeCloudShell
	msiTypeIMDS
)

func (m msiType) String() string {
	switch m {
	case msiTypeUnavailable:
		return "unavailable"
	case msiTypeAppServiceV20170901:
		return "AppServiceV20170901"
	case msiTypeCloudShell:
		return "CloudShell"
	case msiTypeIMDS:
		return "IMDS"
	default:
		return fmt.Sprintf("unhandled MSI type %d", m)
	}
}

// returns the MSI type and endpoint, or an error
func getMSIType() (msiType, string, error) {
	if endpointEnvVar := os.Getenv(msiEndpointEnv); endpointEnvVar != "" {
		// if the env var MSI_ENDPOINT is set
		if secretEnvVar := os.Getenv(msiSecretEnv); secretEnvVar != "" {
			// if BOTH the env vars MSI_ENDPOINT and MSI_SECRET are set the msiType is AppService
			return msiTypeAppServiceV20170901, endpointEnvVar, nil
		}
		// if ONLY the env var MSI_ENDPOINT is set the msiType is CloudShell
		return msiTypeCloudShell, endpointEnvVar, nil
	} else if msiAvailableHook(context.Background(), sender()) {
		// if MSI_ENDPOINT is NOT set AND the IMDS endpoint is available the msiType is IMDS. This will timeout after 500 milliseconds
		return msiTypeIMDS, msiEndpoint, nil
	} else {
		// if MSI_ENDPOINT is NOT set and IMDS endpoint is not available Managed Identity is not available
		return msiTypeUnavailable, "", errors.New("MSI not available")
	}
}

// GetMSIVMEndpoint gets the MSI endpoint on Virtual Machines.
// NOTE: this always returns the IMDS endpoint, it does not work for app services or cloud shell.
// Deprecated: NewServicePrincipalTokenFromMSI() and variants will automatically detect the endpoint.
func GetMSIVMEndpoint() (string, error) {
	return msiEndpoint, nil
}

// GetMSIAppServiceEndpoint get the MSI endpoint for App Service and Functions.
// It will return an error when not running in an app service/functions environment.
// Deprecated: NewServicePrincipalTokenFromMSI() and variants will automatically detect the endpoint.
func GetMSIAppServiceEndpoint() (string, error) {
	msiType, endpoint, err := getMSIType()
	if err != nil {
		return "", err
	}
	switch msiType {
	case msiTypeAppServiceV20170901:
		return endpoint, nil
	default:
		return "", fmt.Errorf("%s is not app service environment", msiType)
	}
}

// GetMSIEndpoint get the appropriate MSI endpoint depending on the runtime environment
// Deprecated: NewServicePrincipalTokenFromMSI() and variants will automatically detect the endpoint.
func GetMSIEndpoint() (string, error) {
	_, endpoint, err := getMSIType()
	return endpoint, err
}

// NewServicePrincipalTokenFromMSI creates a ServicePrincipalToken via the MSI VM Extension.
// It will use the system assigned identity when creating the token.
// msiEndpoint - empty string, or pass a non-empty string to override the default value.
// Deprecated: use NewServicePrincipalTokenFromManagedIdentity() instead.
func NewServicePrincipalTokenFromMSI(msiEndpoint, resource string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	return newServicePrincipalTokenFromMSI(msiEndpoint, resource, "", "", callbacks...)
}

// NewServicePrincipalTokenFromMSIWithUserAssignedID creates a ServicePrincipalToken via the MSI VM Extension.
// It will use the clientID of specified user assigned identity when creating the token.
// msiEndpoint - empty string, or pass a non-empty string to override the default value.
// Deprecated: use NewServicePrincipalTokenFromManagedIdentity() instead.
func NewServicePrincipalTokenFromMSIWithUserAssignedID(msiEndpoint, resource string, userAssignedID string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	if err := validateStringParam(userAssignedID, "userAssignedID"); err != nil {
		return nil, err
	}
	return newServicePrincipalTokenFromMSI(msiEndpoint, resource, userAssignedID, "", callbacks...)
}

// NewServicePrincipalTokenFromMSIWithIdentityResourceID creates a ServicePrincipalToken via the MSI VM Extension.
// It will use the azure resource id of user assigned identity when creating the token.
// msiEndpoint - empty string, or pass a non-empty string to override the default value.
// Deprecated: use NewServicePrincipalTokenFromManagedIdentity() instead.
func NewServicePrincipalTokenFromMSIWithIdentityResourceID(msiEndpoint, resource string, identityResourceID string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	if err := validateStringParam(identityResourceID, "identityResourceID"); err != nil {
		return nil, err
	}
	return newServicePrincipalTokenFromMSI(msiEndpoint, resource, "", identityResourceID, callbacks...)
}

// ManagedIdentityOptions contains optional values for configuring managed identity authentication.
type ManagedIdentityOptions struct {
	// ClientID is the user-assigned identity to use during authentication.
	// It is mutually exclusive with IdentityResourceID.
	ClientID string

	// IdentityResourceID is the resource ID of the user-assigned identity to use during authentication.
	// It is mutually exclusive with ClientID.
	IdentityResourceID string
}

// NewServicePrincipalTokenFromManagedIdentity creates a ServicePrincipalToken using a managed identity.
// It supports the following managed identity environments.
// - App Service Environment (API version 2017-09-01 only)
// - Cloud shell
// - IMDS with a system or user assigned identity
func NewServicePrincipalTokenFromManagedIdentity(resource string, options *ManagedIdentityOptions, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	if options == nil {
		options = &ManagedIdentityOptions{}
	}
	return newServicePrincipalTokenFromMSI("", resource, options.ClientID, options.IdentityResourceID, callbacks...)
}

func newServicePrincipalTokenFromMSI(msiEndpoint, resource, userAssignedID, identityResourceID string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	if err := validateStringParam(resource, "resource"); err != nil {
		return nil, err
	}
	if userAssignedID != "" && identityResourceID != "" {
		return nil, errors.New("cannot specify userAssignedID and identityResourceID")
	}
	msiType, endpoint, err := getMSIType()
	if err != nil {
		logger.Instance.Writef(logger.LogError, "Error determining managed identity environment: %v\n", err)
		return nil, err
	}
	logger.Instance.Writef(logger.LogInfo, "Managed identity environment is %s, endpoint is %s\n", msiType, endpoint)
	if msiEndpoint != "" {
		endpoint = msiEndpoint
		logger.Instance.Writef(logger.LogInfo, "Managed identity custom endpoint is %s\n", endpoint)
	}
	msiEndpointURL, err := url.Parse(endpoint)
	if err != nil {
		return nil, err
	}
	// cloud shell sends its data in the request body
	if msiType != msiTypeCloudShell {
		v := url.Values{}
		v.Set("resource", resource)
		clientIDParam := "client_id"
		switch msiType {
		case msiTypeAppServiceV20170901:
			clientIDParam = "clientid"
			v.Set("api-version", appServiceAPIVersion2017)
			break
		case msiTypeIMDS:
			v.Set("api-version", msiAPIVersion)
		}
		if userAssignedID != "" {
			v.Set(clientIDParam, userAssignedID)
		} else if identityResourceID != "" {
			v.Set("mi_res_id", identityResourceID)
		}
		msiEndpointURL.RawQuery = v.Encode()
	}

	spt := &ServicePrincipalToken{
		inner: servicePrincipalToken{
			Token: newToken(),
			OauthConfig: OAuthConfig{
				TokenEndpoint: *msiEndpointURL,
			},
			Secret: &ServicePrincipalMSISecret{
				msiType:          msiType,
				clientResourceID: identityResourceID,
			},
			Resource:      resource,
			AutoRefresh:   true,
			RefreshWithin: defaultRefresh,
			ClientID:      userAssignedID,
		},
		refreshLock:           &sync.RWMutex{},
		sender:                sender(),
		refreshCallbacks:      callbacks,
		MaxMSIRefreshAttempts: defaultMaxMSIRefreshAttempts,
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
	// must take the read lock when initially checking the token's expiration
	if spt.inner.AutoRefresh && spt.Token().WillExpireIn(spt.inner.RefreshWithin) {
		// take the write lock then check again to see if the token was already refreshed
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
// This method is safe for concurrent use.
func (spt *ServicePrincipalToken) Refresh() error {
	return spt.RefreshWithContext(context.Background())
}

// RefreshWithContext obtains a fresh token for the Service Principal.
// This method is safe for concurrent use.
func (spt *ServicePrincipalToken) RefreshWithContext(ctx context.Context) error {
	spt.refreshLock.Lock()
	defer spt.refreshLock.Unlock()
	return spt.refreshInternal(ctx, spt.inner.Resource)
}

// RefreshExchange refreshes the token, but for a different resource.
// This method is safe for concurrent use.
func (spt *ServicePrincipalToken) RefreshExchange(resource string) error {
	return spt.RefreshExchangeWithContext(context.Background(), resource)
}

// RefreshExchangeWithContext refreshes the token, but for a different resource.
// This method is safe for concurrent use.
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

func (spt *ServicePrincipalToken) refreshInternal(ctx context.Context, resource string) error {
	if spt.customRefreshFunc != nil {
		token, err := spt.customRefreshFunc(ctx, resource)
		if err != nil {
			return err
		}
		spt.inner.Token = *token
		return spt.InvokeRefreshCallbacks(spt.inner.Token)
	}
	req, err := http.NewRequest(http.MethodPost, spt.inner.OauthConfig.TokenEndpoint.String(), nil)
	if err != nil {
		return fmt.Errorf("adal: Failed to build the refresh request. Error = '%v'", err)
	}
	req.Header.Add("User-Agent", UserAgent())
	req = req.WithContext(ctx)
	var resp *http.Response
	authBodyFilter := func(b []byte) []byte {
		if logger.Level() != logger.LogAuth {
			return []byte("**REDACTED** authentication body")
		}
		return b
	}
	if msiSecret, ok := spt.inner.Secret.(*ServicePrincipalMSISecret); ok {
		switch msiSecret.msiType {
		case msiTypeAppServiceV20170901:
			req.Method = http.MethodGet
			req.Header.Set("secret", os.Getenv(msiSecretEnv))
			break
		case msiTypeCloudShell:
			req.Header.Set("Metadata", "true")
			data := url.Values{}
			data.Set("resource", spt.inner.Resource)
			if spt.inner.ClientID != "" {
				data.Set("client_id", spt.inner.ClientID)
			} else if msiSecret.clientResourceID != "" {
				data.Set("msi_res_id", msiSecret.clientResourceID)
			}
			req.Body = ioutil.NopCloser(strings.NewReader(data.Encode()))
			req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
			break
		case msiTypeIMDS:
			req.Method = http.MethodGet
			req.Header.Set("Metadata", "true")
			break
		}
		logger.Instance.WriteRequest(req, logger.Filter{Body: authBodyFilter})
		resp, err = retryForIMDS(spt.sender, req, spt.MaxMSIRefreshAttempts)
	} else {
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
		logger.Instance.WriteRequest(req, logger.Filter{Body: authBodyFilter})
		resp, err = spt.sender.Do(req)
	}

	// don't return a TokenRefreshError here; this will allow retry logic to apply
	if err != nil {
		return fmt.Errorf("adal: Failed to execute the refresh request. Error = '%v'", err)
	} else if resp == nil {
		return fmt.Errorf("adal: received nil response and error")
	}

	logger.Instance.WriteResponse(resp, logger.Filter{Body: authBodyFilter})
	defer resp.Body.Close()
	rb, err := ioutil.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		if err != nil {
			return newTokenRefreshError(fmt.Sprintf("adal: Refresh request failed. Status Code = '%d'. Failed reading response body: %v Endpoint %s", resp.StatusCode, err, req.URL.String()), resp)
		}
		return newTokenRefreshError(fmt.Sprintf("adal: Refresh request failed. Status Code = '%d'. Response body: %s Endpoint %s", resp.StatusCode, string(rb), req.URL.String()), resp)
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
	token := struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`

		// AAD returns expires_in as a string, ADFS returns it as an int
		ExpiresIn json.Number `json:"expires_in"`
		// expires_on can be in two formats, a UTC time stamp or the number of seconds.
		ExpiresOn string      `json:"expires_on"`
		NotBefore json.Number `json:"not_before"`

		Resource string `json:"resource"`
		Type     string `json:"token_type"`
	}{}
	// return a TokenRefreshError in the follow error cases as the token is in an unexpected format
	err = json.Unmarshal(rb, &token)
	if err != nil {
		return newTokenRefreshError(fmt.Sprintf("adal: Failed to unmarshal the service principal token during refresh. Error = '%v' JSON = '%s'", err, string(rb)), resp)
	}
	expiresOn := json.Number("")
	// ADFS doesn't include the expires_on field
	if token.ExpiresOn != "" {
		if expiresOn, err = parseExpiresOn(token.ExpiresOn); err != nil {
			return newTokenRefreshError(fmt.Sprintf("adal: failed to parse expires_on: %v value '%s'", err, token.ExpiresOn), resp)
		}
	}
	spt.inner.Token.AccessToken = token.AccessToken
	spt.inner.Token.RefreshToken = token.RefreshToken
	spt.inner.Token.ExpiresIn = token.ExpiresIn
	spt.inner.Token.ExpiresOn = expiresOn
	spt.inner.Token.NotBefore = token.NotBefore
	spt.inner.Token.Resource = token.Resource
	spt.inner.Token.Type = token.Type

	return spt.InvokeRefreshCallbacks(spt.inner.Token)
}

// converts expires_on to the number of seconds
func parseExpiresOn(s string) (json.Number, error) {
	// convert the expiration date to the number of seconds from now
	timeToDuration := func(t time.Time) json.Number {
		dur := t.Sub(time.Now().UTC())
		return json.Number(strconv.FormatInt(int64(dur.Round(time.Second).Seconds()), 10))
	}
	if _, err := strconv.ParseInt(s, 10, 64); err == nil {
		// this is the number of seconds case, no conversion required
		return json.Number(s), nil
	} else if eo, err := time.Parse(expiresOnDateFormatPM, s); err == nil {
		return timeToDuration(eo), nil
	} else if eo, err := time.Parse(expiresOnDateFormat, s); err == nil {
		return timeToDuration(eo), nil
	} else {
		// unknown format
		return json.Number(""), err
	}
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

	// maxAttempts is user-specified, ensure that its value is greater than zero else no request will be made
	if maxAttempts < 1 {
		maxAttempts = defaultMaxMSIRefreshAttempts
	}

	for attempt < maxAttempts {
		if resp != nil && resp.Body != nil {
			io.Copy(ioutil.Discard, resp.Body)
			resp.Body.Close()
		}
		resp, err = sender.Do(req)
		// we want to retry if err is not nil or the status code is in the list of retry codes
		if err == nil && !responseHasStatusCode(resp, retries...) {
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

func responseHasStatusCode(resp *http.Response, codes ...int) bool {
	if resp != nil {
		for _, i := range codes {
			if i == resp.StatusCode {
				return true
			}
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

// MultiTenantServicePrincipalToken contains tokens for multi-tenant authorization.
type MultiTenantServicePrincipalToken struct {
	PrimaryToken    *ServicePrincipalToken
	AuxiliaryTokens []*ServicePrincipalToken
}

// PrimaryOAuthToken returns the primary authorization token.
func (mt *MultiTenantServicePrincipalToken) PrimaryOAuthToken() string {
	return mt.PrimaryToken.OAuthToken()
}

// AuxiliaryOAuthTokens returns one to three auxiliary authorization tokens.
func (mt *MultiTenantServicePrincipalToken) AuxiliaryOAuthTokens() []string {
	tokens := make([]string, len(mt.AuxiliaryTokens))
	for i := range mt.AuxiliaryTokens {
		tokens[i] = mt.AuxiliaryTokens[i].OAuthToken()
	}
	return tokens
}

// NewMultiTenantServicePrincipalToken creates a new MultiTenantServicePrincipalToken with the specified credentials and resource.
func NewMultiTenantServicePrincipalToken(multiTenantCfg MultiTenantOAuthConfig, clientID string, secret string, resource string) (*MultiTenantServicePrincipalToken, error) {
	if err := validateStringParam(clientID, "clientID"); err != nil {
		return nil, err
	}
	if err := validateStringParam(secret, "secret"); err != nil {
		return nil, err
	}
	if err := validateStringParam(resource, "resource"); err != nil {
		return nil, err
	}
	auxTenants := multiTenantCfg.AuxiliaryTenants()
	m := MultiTenantServicePrincipalToken{
		AuxiliaryTokens: make([]*ServicePrincipalToken, len(auxTenants)),
	}
	primary, err := NewServicePrincipalToken(*multiTenantCfg.PrimaryTenant(), clientID, secret, resource)
	if err != nil {
		return nil, fmt.Errorf("failed to create SPT for primary tenant: %v", err)
	}
	m.PrimaryToken = primary
	for i := range auxTenants {
		aux, err := NewServicePrincipalToken(*auxTenants[i], clientID, secret, resource)
		if err != nil {
			return nil, fmt.Errorf("failed to create SPT for auxiliary tenant: %v", err)
		}
		m.AuxiliaryTokens[i] = aux
	}
	return &m, nil
}

// NewMultiTenantServicePrincipalTokenFromCertificate creates a new MultiTenantServicePrincipalToken with the specified certificate credentials and resource.
func NewMultiTenantServicePrincipalTokenFromCertificate(multiTenantCfg MultiTenantOAuthConfig, clientID string, certificate *x509.Certificate, privateKey *rsa.PrivateKey, resource string) (*MultiTenantServicePrincipalToken, error) {
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
	auxTenants := multiTenantCfg.AuxiliaryTenants()
	m := MultiTenantServicePrincipalToken{
		AuxiliaryTokens: make([]*ServicePrincipalToken, len(auxTenants)),
	}
	primary, err := NewServicePrincipalTokenWithSecret(
		*multiTenantCfg.PrimaryTenant(),
		clientID,
		resource,
		&ServicePrincipalCertificateSecret{
			PrivateKey:  privateKey,
			Certificate: certificate,
		},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create SPT for primary tenant: %v", err)
	}
	m.PrimaryToken = primary
	for i := range auxTenants {
		aux, err := NewServicePrincipalTokenWithSecret(
			*auxTenants[i],
			clientID,
			resource,
			&ServicePrincipalCertificateSecret{
				PrivateKey:  privateKey,
				Certificate: certificate,
			},
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create SPT for auxiliary tenant: %v", err)
		}
		m.AuxiliaryTokens[i] = aux
	}
	return &m, nil
}

// MSIAvailable returns true if the MSI endpoint is available for authentication.
func MSIAvailable(ctx context.Context, sender Sender) bool {
	resp, err := getMSIEndpoint(ctx, sender)
	if err == nil {
		resp.Body.Close()
	}
	return err == nil
}

// used for testing purposes
var msiAvailableHook = func(ctx context.Context, sender Sender) bool {
	return MSIAvailable(ctx, sender)
}
