package adal

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha1"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/dgrijalva/jwt-go"
)

const (
	defaultRefresh = 5 * time.Minute
	tokenBaseDate  = "1970-01-01T00:00:00Z"

	// OAuthGrantTypeDeviceCode is the "grant_type" identifier used in device flow
	OAuthGrantTypeDeviceCode = "device_code"

	// OAuthGrantTypeClientCredentials is the "grant_type" identifier used in credential flows
	OAuthGrantTypeClientCredentials = "client_credentials"

	// OAuthGrantTypeRefreshToken is the "grant_type" identifier used in refresh token flows
	OAuthGrantTypeRefreshToken = "refresh_token"

	// managedIdentitySettingsPath is the path to the MSI Extension settings file (to discover the endpoint)
	managedIdentitySettingsPath = "/var/lib/waagent/ManagedIdentity-Settings"
)

var expirationBase time.Time

func init() {
	expirationBase, _ = time.Parse(time.RFC3339, tokenBaseDate)
}

// OAuthTokenProvider is an interface which should be implemented by an access token retriever
type OAuthTokenProvider interface {
	OAuthToken() string
}

// Refresher is an interface for token refresh functionality
type Refresher interface {
	Refresh() error
	RefreshExchange(resource string) error
	EnsureFresh() error
}

// TokenRefreshCallback is the type representing callbacks that will be called after
// a successful token refresh
type TokenRefreshCallback func(Token) error

// Token encapsulates the access token used to authorize Azure requests.
type Token struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`

	ExpiresIn string `json:"expires_in"`
	ExpiresOn string `json:"expires_on"`
	NotBefore string `json:"not_before"`

	Resource string `json:"resource"`
	Type     string `json:"token_type"`
}

// Expires returns the time.Time when the Token expires.
func (t Token) Expires() time.Time {
	s, err := strconv.Atoi(t.ExpiresOn)
	if err != nil {
		s = -3600
	}
	return expirationBase.Add(time.Duration(s) * time.Second).UTC()
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

// ServicePrincipalNoSecret represents a secret type that contains no secret
// meaning it is not valid for fetching a fresh token. This is used by Manual
type ServicePrincipalNoSecret struct {
}

// SetAuthenticationValues is a method of the interface ServicePrincipalSecret
// It only returns an error for the ServicePrincipalNoSecret type
func (noSecret *ServicePrincipalNoSecret) SetAuthenticationValues(spt *ServicePrincipalToken, v *url.Values) error {
	return fmt.Errorf("Manually created ServicePrincipalToken does not contain secret material to retrieve a new access token")
}

// ServicePrincipalSecret is an interface that allows various secret mechanism to fill the form
// that is submitted when acquiring an oAuth token.
type ServicePrincipalSecret interface {
	SetAuthenticationValues(spt *ServicePrincipalToken, values *url.Values) error
}

// ServicePrincipalTokenSecret implements ServicePrincipalSecret for client_secret type authorization.
type ServicePrincipalTokenSecret struct {
	ClientSecret string
}

// SetAuthenticationValues is a method of the interface ServicePrincipalSecret.
// It will populate the form submitted during oAuth Token Acquisition using the client_secret.
func (tokenSecret *ServicePrincipalTokenSecret) SetAuthenticationValues(spt *ServicePrincipalToken, v *url.Values) error {
	v.Set("client_secret", tokenSecret.ClientSecret)
	return nil
}

// ServicePrincipalCertificateSecret implements ServicePrincipalSecret for generic RSA cert auth with signed JWTs.
type ServicePrincipalCertificateSecret struct {
	Certificate *x509.Certificate
	PrivateKey  *rsa.PrivateKey
}

// ServicePrincipalMSISecret implements ServicePrincipalSecret for machines running the MSI Extension.
type ServicePrincipalMSISecret struct {
}

// SetAuthenticationValues is a method of the interface ServicePrincipalSecret.
// MSI extension requires the authority field to be set to the real tenant authority endpoint
func (msiSecret *ServicePrincipalMSISecret) SetAuthenticationValues(spt *ServicePrincipalToken, v *url.Values) error {
	v.Set("authority", spt.oauthConfig.AuthorityEndpoint.String())
	return nil
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
	token.Claims = jwt.MapClaims{
		"aud": spt.oauthConfig.TokenEndpoint.String(),
		"iss": spt.clientID,
		"sub": spt.clientID,
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

// ServicePrincipalToken encapsulates a Token created for a Service Principal.
type ServicePrincipalToken struct {
	Token

	secret        ServicePrincipalSecret
	oauthConfig   OAuthConfig
	clientID      string
	resource      string
	autoRefresh   bool
	refreshWithin time.Duration
	sender        Sender

	refreshCallbacks []TokenRefreshCallback
}

// NewServicePrincipalTokenWithSecret create a ServicePrincipalToken using the supplied ServicePrincipalSecret implementation.
func NewServicePrincipalTokenWithSecret(oauthConfig OAuthConfig, id string, resource string, secret ServicePrincipalSecret, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	spt := &ServicePrincipalToken{
		oauthConfig:      oauthConfig,
		secret:           secret,
		clientID:         id,
		resource:         resource,
		autoRefresh:      true,
		refreshWithin:    defaultRefresh,
		sender:           &http.Client{},
		refreshCallbacks: callbacks,
	}
	return spt, nil
}

// NewServicePrincipalTokenFromManualToken creates a ServicePrincipalToken using the supplied token
func NewServicePrincipalTokenFromManualToken(oauthConfig OAuthConfig, clientID string, resource string, token Token, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	spt, err := NewServicePrincipalTokenWithSecret(
		oauthConfig,
		clientID,
		resource,
		&ServicePrincipalNoSecret{},
		callbacks...)
	if err != nil {
		return nil, err
	}

	spt.Token = token

	return spt, nil
}

// NewServicePrincipalToken creates a ServicePrincipalToken from the supplied Service Principal
// credentials scoped to the named resource.
func NewServicePrincipalToken(oauthConfig OAuthConfig, clientID string, secret string, resource string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
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

// NewServicePrincipalTokenFromCertificate create a ServicePrincipalToken from the supplied pkcs12 bytes.
func NewServicePrincipalTokenFromCertificate(oauthConfig OAuthConfig, clientID string, certificate *x509.Certificate, privateKey *rsa.PrivateKey, resource string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
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

// NewServicePrincipalTokenFromMSI creates a ServicePrincipalToken via the MSI VM Extension.
func NewServicePrincipalTokenFromMSI(oauthConfig OAuthConfig, resource string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	return newServicePrincipalTokenFromMSI(oauthConfig, resource, managedIdentitySettingsPath, callbacks...)
}

func newServicePrincipalTokenFromMSI(oauthConfig OAuthConfig, resource, settingsPath string, callbacks ...TokenRefreshCallback) (*ServicePrincipalToken, error) {
	// Read MSI settings
	bytes, err := ioutil.ReadFile(settingsPath)
	if err != nil {
		return nil, err
	}
	msiSettings := struct {
		URL string `json:"url"`
	}{}
	err = json.Unmarshal(bytes, &msiSettings)
	if err != nil {
		return nil, err
	}

	// We set the oauth config token endpoint to be MSI's endpoint
	// We leave the authority as-is so MSI can POST it with the token request
	msiEndpointURL, err := url.Parse(msiSettings.URL)
	if err != nil {
		return nil, err
	}

	msiTokenEndpointURL, err := msiEndpointURL.Parse("/oauth2/token")
	if err != nil {
		return nil, err
	}

	oauthConfig.TokenEndpoint = *msiTokenEndpointURL

	spt := &ServicePrincipalToken{
		oauthConfig:      oauthConfig,
		secret:           &ServicePrincipalMSISecret{},
		resource:         resource,
		autoRefresh:      true,
		refreshWithin:    defaultRefresh,
		sender:           &http.Client{},
		refreshCallbacks: callbacks,
	}

	return spt, nil
}

// EnsureFresh will refresh the token if it will expire within the refresh window (as set by
// RefreshWithin) and autoRefresh flag is on.
func (spt *ServicePrincipalToken) EnsureFresh() error {
	if spt.autoRefresh && spt.WillExpireIn(spt.refreshWithin) {
		return spt.Refresh()
	}
	return nil
}

// InvokeRefreshCallbacks calls any TokenRefreshCallbacks that were added to the SPT during initialization
func (spt *ServicePrincipalToken) InvokeRefreshCallbacks(token Token) error {
	if spt.refreshCallbacks != nil {
		for _, callback := range spt.refreshCallbacks {
			err := callback(spt.Token)
			if err != nil {
				return fmt.Errorf("adal: TokenRefreshCallback handler failed. Error = '%v'", err)
			}
		}
	}
	return nil
}

// Refresh obtains a fresh token for the Service Principal.
func (spt *ServicePrincipalToken) Refresh() error {
	return spt.refreshInternal(spt.resource)
}

// RefreshExchange refreshes the token, but for a different resource.
func (spt *ServicePrincipalToken) RefreshExchange(resource string) error {
	return spt.refreshInternal(resource)
}

func (spt *ServicePrincipalToken) refreshInternal(resource string) error {
	v := url.Values{}
	v.Set("client_id", spt.clientID)
	v.Set("resource", resource)

	if spt.RefreshToken != "" {
		v.Set("grant_type", OAuthGrantTypeRefreshToken)
		v.Set("refresh_token", spt.RefreshToken)
	} else {
		v.Set("grant_type", OAuthGrantTypeClientCredentials)
		err := spt.secret.SetAuthenticationValues(spt, &v)
		if err != nil {
			return err
		}
	}

	s := v.Encode()
	body := ioutil.NopCloser(strings.NewReader(s))
	req, err := http.NewRequest(http.MethodPost, spt.oauthConfig.TokenEndpoint.String(), body)
	if err != nil {
		return fmt.Errorf("adal: Failed to build the refresh request. Error = '%v'", err)
	}

	req.ContentLength = int64(len(s))
	req.Header.Set(contentType, mimeTypeFormPost)
	resp, err := spt.sender.Do(req)
	if err != nil {
		return fmt.Errorf("adal: Failed to execute the refresh request. Error = '%v'", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("adal: Refresh request failed. Status Code = '%d'", resp.StatusCode)
	}

	rb, err := ioutil.ReadAll(resp.Body)
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

	spt.Token = token

	return spt.InvokeRefreshCallbacks(token)
}

// SetAutoRefresh enables or disables automatic refreshing of stale tokens.
func (spt *ServicePrincipalToken) SetAutoRefresh(autoRefresh bool) {
	spt.autoRefresh = autoRefresh
}

// SetRefreshWithin sets the interval within which if the token will expire, EnsureFresh will
// refresh the token.
func (spt *ServicePrincipalToken) SetRefreshWithin(d time.Duration) {
	spt.refreshWithin = d
	return
}

// SetSender sets the http.Client used when obtaining the Service Principal token. An
// undecorated http.Client is used by default.
func (spt *ServicePrincipalToken) SetSender(s Sender) { spt.sender = s }
