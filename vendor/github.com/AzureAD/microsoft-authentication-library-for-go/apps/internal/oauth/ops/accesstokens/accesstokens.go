// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
Package accesstokens exposes a REST client for querying backend systems to get various types of
access tokens (oauth) for use in authentication.

These calls are of type "application/x-www-form-urlencoded".  This means we use url.Values to
represent arguments and then encode them into the POST body message.  We receive JSON in
return for the requests.  The request definition is defined in https://tools.ietf.org/html/rfc7521#section-4.2 .
*/
package accesstokens

import (
	"context"
	"crypto"

	/* #nosec */
	"crypto/sha1"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/exported"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth/ops/authority"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth/ops/internal/grant"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/oauth/ops/wstrust"
	"github.com/golang-jwt/jwt/v4"
	"github.com/google/uuid"
)

const (
	grantType     = "grant_type"
	deviceCode    = "device_code"
	clientID      = "client_id"
	clientInfo    = "client_info"
	clientInfoVal = "1"
	username      = "username"
	password      = "password"
)

//go:generate stringer -type=AppType

// AppType is whether the authorization code flow is for a public or confidential client.
type AppType int8

const (
	// ATUnknown is the zero value when the type hasn't been set.
	ATUnknown AppType = iota
	// ATPublic indicates this if for the Public.Client.
	ATPublic
	// ATConfidential indicates this if for the Confidential.Client.
	ATConfidential
)

type urlFormCaller interface {
	URLFormCall(ctx context.Context, endpoint string, qv url.Values, resp interface{}) error
}

// DeviceCodeResponse represents the HTTP response received from the device code endpoint
type DeviceCodeResponse struct {
	authority.OAuthResponseBase

	UserCode        string `json:"user_code"`
	DeviceCode      string `json:"device_code"`
	VerificationURL string `json:"verification_url"`
	ExpiresIn       int    `json:"expires_in"`
	Interval        int    `json:"interval"`
	Message         string `json:"message"`

	AdditionalFields map[string]interface{}
}

// Convert converts the DeviceCodeResponse to a DeviceCodeResult
func (dcr DeviceCodeResponse) Convert(clientID string, scopes []string) DeviceCodeResult {
	expiresOn := time.Now().UTC().Add(time.Duration(dcr.ExpiresIn) * time.Second)
	return NewDeviceCodeResult(dcr.UserCode, dcr.DeviceCode, dcr.VerificationURL, expiresOn, dcr.Interval, dcr.Message, clientID, scopes)
}

// Credential represents the credential used in confidential client flows. This can be either
// a Secret or Cert/Key.
type Credential struct {
	// Secret contains the credential secret if we are doing auth by secret.
	Secret string

	// Cert is the public certificate, if we're authenticating by certificate.
	Cert *x509.Certificate
	// Key is the private key for signing, if we're authenticating by certificate.
	Key crypto.PrivateKey
	// X5c is the JWT assertion's x5c header value, required for SN/I authentication.
	X5c []string

	// AssertionCallback is a function provided by the application, if we're authenticating by assertion.
	AssertionCallback func(context.Context, exported.AssertionRequestOptions) (string, error)

	// TokenProvider is a function provided by the application that implements custom authentication
	// logic for a confidential client
	TokenProvider func(context.Context, exported.TokenProviderParameters) (exported.TokenProviderResult, error)
}

// JWT gets the jwt assertion when the credential is not using a secret.
func (c *Credential) JWT(ctx context.Context, authParams authority.AuthParams) (string, error) {
	if c.AssertionCallback != nil {
		options := exported.AssertionRequestOptions{
			ClientID:      authParams.ClientID,
			TokenEndpoint: authParams.Endpoints.TokenEndpoint,
		}
		return c.AssertionCallback(ctx, options)
	}

	token := jwt.NewWithClaims(jwt.SigningMethodRS256, jwt.MapClaims{
		"aud": authParams.Endpoints.TokenEndpoint,
		"exp": json.Number(strconv.FormatInt(time.Now().Add(10*time.Minute).Unix(), 10)),
		"iss": authParams.ClientID,
		"jti": uuid.New().String(),
		"nbf": json.Number(strconv.FormatInt(time.Now().Unix(), 10)),
		"sub": authParams.ClientID,
	})
	token.Header = map[string]interface{}{
		"alg": "RS256",
		"typ": "JWT",
		"x5t": base64.StdEncoding.EncodeToString(thumbprint(c.Cert)),
	}

	if authParams.SendX5C {
		token.Header["x5c"] = c.X5c
	}

	assertion, err := token.SignedString(c.Key)
	if err != nil {
		return "", fmt.Errorf("unable to sign a JWT token using private key: %w", err)
	}
	return assertion, nil
}

// thumbprint runs the asn1.Der bytes through sha1 for use in the x5t parameter of JWT.
// https://tools.ietf.org/html/rfc7517#section-4.8
func thumbprint(cert *x509.Certificate) []byte {
	/* #nosec */
	a := sha1.Sum(cert.Raw)
	return a[:]
}

// Client represents the REST calls to get tokens from token generator backends.
type Client struct {
	// Comm provides the HTTP transport client.
	Comm urlFormCaller

	testing bool
}

// FromUsernamePassword uses a username and password to get an access token.
func (c Client) FromUsernamePassword(ctx context.Context, authParameters authority.AuthParams) (TokenResponse, error) {
	qv := url.Values{}
	if err := addClaims(qv, authParameters); err != nil {
		return TokenResponse{}, err
	}
	qv.Set(grantType, grant.Password)
	qv.Set(username, authParameters.Username)
	qv.Set(password, authParameters.Password)
	qv.Set(clientID, authParameters.ClientID)
	qv.Set(clientInfo, clientInfoVal)
	addScopeQueryParam(qv, authParameters)

	return c.doTokenResp(ctx, authParameters, qv)
}

// AuthCodeRequest stores the values required to request a token from the authority using an authorization code
type AuthCodeRequest struct {
	AuthParams    authority.AuthParams
	Code          string
	CodeChallenge string
	Credential    *Credential
	AppType       AppType
}

// NewCodeChallengeRequest returns an AuthCodeRequest that uses a code challenge..
func NewCodeChallengeRequest(params authority.AuthParams, appType AppType, cc *Credential, code, challenge string) (AuthCodeRequest, error) {
	if appType == ATUnknown {
		return AuthCodeRequest{}, fmt.Errorf("bug: NewCodeChallengeRequest() called with AppType == ATUnknown")
	}
	return AuthCodeRequest{
		AuthParams:    params,
		AppType:       appType,
		Code:          code,
		CodeChallenge: challenge,
		Credential:    cc,
	}, nil
}

// FromAuthCode uses an authorization code to retrieve an access token.
func (c Client) FromAuthCode(ctx context.Context, req AuthCodeRequest) (TokenResponse, error) {
	var qv url.Values

	switch req.AppType {
	case ATUnknown:
		return TokenResponse{}, fmt.Errorf("bug: Token.AuthCode() received request with AppType == ATUnknown")
	case ATConfidential:
		var err error
		if req.Credential == nil {
			return TokenResponse{}, fmt.Errorf("AuthCodeRequest had nil Credential for Confidential app")
		}
		qv, err = prepURLVals(ctx, req.Credential, req.AuthParams)
		if err != nil {
			return TokenResponse{}, err
		}
	case ATPublic:
		qv = url.Values{}
	default:
		return TokenResponse{}, fmt.Errorf("bug: Token.AuthCode() received request with AppType == %v, which we do not recongnize", req.AppType)
	}

	qv.Set(grantType, grant.AuthCode)
	qv.Set("code", req.Code)
	qv.Set("code_verifier", req.CodeChallenge)
	qv.Set("redirect_uri", req.AuthParams.Redirecturi)
	qv.Set(clientID, req.AuthParams.ClientID)
	qv.Set(clientInfo, clientInfoVal)
	addScopeQueryParam(qv, req.AuthParams)
	if err := addClaims(qv, req.AuthParams); err != nil {
		return TokenResponse{}, err
	}

	return c.doTokenResp(ctx, req.AuthParams, qv)
}

// FromRefreshToken uses a refresh token (for refreshing credentials) to get a new access token.
func (c Client) FromRefreshToken(ctx context.Context, appType AppType, authParams authority.AuthParams, cc *Credential, refreshToken string) (TokenResponse, error) {
	qv := url.Values{}
	if appType == ATConfidential {
		var err error
		qv, err = prepURLVals(ctx, cc, authParams)
		if err != nil {
			return TokenResponse{}, err
		}
	}
	if err := addClaims(qv, authParams); err != nil {
		return TokenResponse{}, err
	}
	qv.Set(grantType, grant.RefreshToken)
	qv.Set(clientID, authParams.ClientID)
	qv.Set(clientInfo, clientInfoVal)
	qv.Set("refresh_token", refreshToken)
	addScopeQueryParam(qv, authParams)

	return c.doTokenResp(ctx, authParams, qv)
}

// FromClientSecret uses a client's secret (aka password) to get a new token.
func (c Client) FromClientSecret(ctx context.Context, authParameters authority.AuthParams, clientSecret string) (TokenResponse, error) {
	qv := url.Values{}
	if err := addClaims(qv, authParameters); err != nil {
		return TokenResponse{}, err
	}
	qv.Set(grantType, grant.ClientCredential)
	qv.Set("client_secret", clientSecret)
	qv.Set(clientID, authParameters.ClientID)
	addScopeQueryParam(qv, authParameters)

	token, err := c.doTokenResp(ctx, authParameters, qv)
	if err != nil {
		return token, fmt.Errorf("FromClientSecret(): %w", err)
	}
	return token, nil
}

func (c Client) FromAssertion(ctx context.Context, authParameters authority.AuthParams, assertion string) (TokenResponse, error) {
	qv := url.Values{}
	if err := addClaims(qv, authParameters); err != nil {
		return TokenResponse{}, err
	}
	qv.Set(grantType, grant.ClientCredential)
	qv.Set("client_assertion_type", grant.ClientAssertion)
	qv.Set("client_assertion", assertion)
	qv.Set(clientID, authParameters.ClientID)
	qv.Set(clientInfo, clientInfoVal)
	addScopeQueryParam(qv, authParameters)

	token, err := c.doTokenResp(ctx, authParameters, qv)
	if err != nil {
		return token, fmt.Errorf("FromAssertion(): %w", err)
	}
	return token, nil
}

func (c Client) FromUserAssertionClientSecret(ctx context.Context, authParameters authority.AuthParams, userAssertion string, clientSecret string) (TokenResponse, error) {
	qv := url.Values{}
	if err := addClaims(qv, authParameters); err != nil {
		return TokenResponse{}, err
	}
	qv.Set(grantType, grant.JWT)
	qv.Set(clientID, authParameters.ClientID)
	qv.Set("client_secret", clientSecret)
	qv.Set("assertion", userAssertion)
	qv.Set(clientInfo, clientInfoVal)
	qv.Set("requested_token_use", "on_behalf_of")
	addScopeQueryParam(qv, authParameters)

	return c.doTokenResp(ctx, authParameters, qv)
}

func (c Client) FromUserAssertionClientCertificate(ctx context.Context, authParameters authority.AuthParams, userAssertion string, assertion string) (TokenResponse, error) {
	qv := url.Values{}
	if err := addClaims(qv, authParameters); err != nil {
		return TokenResponse{}, err
	}
	qv.Set(grantType, grant.JWT)
	qv.Set("client_assertion_type", grant.ClientAssertion)
	qv.Set("client_assertion", assertion)
	qv.Set(clientID, authParameters.ClientID)
	qv.Set("assertion", userAssertion)
	qv.Set(clientInfo, clientInfoVal)
	qv.Set("requested_token_use", "on_behalf_of")
	addScopeQueryParam(qv, authParameters)

	return c.doTokenResp(ctx, authParameters, qv)
}

func (c Client) DeviceCodeResult(ctx context.Context, authParameters authority.AuthParams) (DeviceCodeResult, error) {
	qv := url.Values{}
	if err := addClaims(qv, authParameters); err != nil {
		return DeviceCodeResult{}, err
	}
	qv.Set(clientID, authParameters.ClientID)
	addScopeQueryParam(qv, authParameters)

	endpoint := strings.Replace(authParameters.Endpoints.TokenEndpoint, "token", "devicecode", -1)

	resp := DeviceCodeResponse{}
	err := c.Comm.URLFormCall(ctx, endpoint, qv, &resp)
	if err != nil {
		return DeviceCodeResult{}, err
	}

	return resp.Convert(authParameters.ClientID, authParameters.Scopes), nil
}

func (c Client) FromDeviceCodeResult(ctx context.Context, authParameters authority.AuthParams, deviceCodeResult DeviceCodeResult) (TokenResponse, error) {
	qv := url.Values{}
	if err := addClaims(qv, authParameters); err != nil {
		return TokenResponse{}, err
	}
	qv.Set(grantType, grant.DeviceCode)
	qv.Set(deviceCode, deviceCodeResult.DeviceCode)
	qv.Set(clientID, authParameters.ClientID)
	qv.Set(clientInfo, clientInfoVal)
	addScopeQueryParam(qv, authParameters)

	return c.doTokenResp(ctx, authParameters, qv)
}

func (c Client) FromSamlGrant(ctx context.Context, authParameters authority.AuthParams, samlGrant wstrust.SamlTokenInfo) (TokenResponse, error) {
	qv := url.Values{}
	if err := addClaims(qv, authParameters); err != nil {
		return TokenResponse{}, err
	}
	qv.Set(username, authParameters.Username)
	qv.Set(password, authParameters.Password)
	qv.Set(clientID, authParameters.ClientID)
	qv.Set(clientInfo, clientInfoVal)
	qv.Set("assertion", base64.StdEncoding.WithPadding(base64.StdPadding).EncodeToString([]byte(samlGrant.Assertion)))
	addScopeQueryParam(qv, authParameters)

	switch samlGrant.AssertionType {
	case grant.SAMLV1:
		qv.Set(grantType, grant.SAMLV1)
	case grant.SAMLV2:
		qv.Set(grantType, grant.SAMLV2)
	default:
		return TokenResponse{}, fmt.Errorf("GetAccessTokenFromSamlGrant returned unknown SAML assertion type: %q", samlGrant.AssertionType)
	}

	return c.doTokenResp(ctx, authParameters, qv)
}

func (c Client) doTokenResp(ctx context.Context, authParams authority.AuthParams, qv url.Values) (TokenResponse, error) {
	resp := TokenResponse{}
	err := c.Comm.URLFormCall(ctx, authParams.Endpoints.TokenEndpoint, qv, &resp)
	if err != nil {
		return resp, err
	}
	resp.ComputeScope(authParams)
	if c.testing {
		return resp, nil
	}
	return resp, resp.Validate()
}

// prepURLVals returns an url.Values that sets various key/values if we are doing secrets
// or JWT assertions.
func prepURLVals(ctx context.Context, cc *Credential, authParams authority.AuthParams) (url.Values, error) {
	params := url.Values{}
	if cc.Secret != "" {
		params.Set("client_secret", cc.Secret)
		return params, nil
	}

	jwt, err := cc.JWT(ctx, authParams)
	if err != nil {
		return nil, err
	}
	params.Set("client_assertion", jwt)
	params.Set("client_assertion_type", grant.ClientAssertion)
	return params, nil
}

// openid required to get an id token
// offline_access required to get a refresh token
// profile required to get the client_info field back
var detectDefaultScopes = map[string]bool{
	"openid":         true,
	"offline_access": true,
	"profile":        true,
}

var defaultScopes = []string{"openid", "offline_access", "profile"}

func AppendDefaultScopes(authParameters authority.AuthParams) []string {
	scopes := make([]string, 0, len(authParameters.Scopes)+len(defaultScopes))
	for _, scope := range authParameters.Scopes {
		s := strings.TrimSpace(scope)
		if s == "" {
			continue
		}
		if detectDefaultScopes[scope] {
			continue
		}
		scopes = append(scopes, scope)
	}
	scopes = append(scopes, defaultScopes...)
	return scopes
}

// addClaims adds client capabilities and claims from AuthParams to the given url.Values
func addClaims(v url.Values, ap authority.AuthParams) error {
	claims, err := ap.MergeCapabilitiesAndClaims()
	if err == nil && claims != "" {
		v.Set("claims", claims)
	}
	return err
}

func addScopeQueryParam(queryParams url.Values, authParameters authority.AuthParams) {
	scopes := AppendDefaultScopes(authParameters)
	queryParams.Set("scope", strings.Join(scopes, " "))
}
