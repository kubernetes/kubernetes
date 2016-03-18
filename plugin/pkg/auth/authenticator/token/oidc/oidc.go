/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// oidc implements the authenticator.Token interface using the OpenID Connect protocol.
package oidc

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/oauth2"
	"github.com/coreos/go-oidc/oidc"
	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/authenticator/bearertoken"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/net"
)

const (
	PathExchangeRefreshToken = "/oidc-exchange-refresh-token"
	PathAuthenticate         = "/oidc-authenticate"
	PathAuthCallback         = "/oidc-auth-callback"
	PathExchangeCode         = "/oidc-exchange-auth-code"
)

var (
	maxRetries   = 5
	retryBackoff = time.Second * 3

	empty           = struct{}{}
	bypassAuthPaths = map[string]struct{}{
		PathExchangeRefreshToken: empty,
		PathAuthenticate:         empty,
		PathAuthCallback:         empty,
		PathExchangeCode:         empty,
	}

	UserUnauthenticated = &user.DefaultInfo{
		Name: "oidc:unauthenticated_user",
		UID:  "oidc:unauthenticated_user",
	}
)

type OIDCAuthenticator struct {
	clientConfig        oidc.ClientConfig
	client              *oidc.Client
	usernameClaim       string
	groupsClaim         string
	stopSyncProvider    chan struct{}
	bearerAuthenticator authenticator.Request
	handlersAttached    bool
}

type OIDCHTTPHandler struct {
	client       OIDCClient
	apiServerURL url.URL
}

// OIDCClient peforms the subset of OIDC related operations necessary for the HTTP Handler.
type OIDCClient interface {
	// RefreshToken exchanges a refresh token for an ID token.
	RefreshToken(rt string) (jose.JWT, error)

	// AuthCode URL generates the authorization URL for the authorization code flow.
	AuthCodeURL(state, accessType, prompt string) string

	// RequestToken requests a token from the Token Endpoint with the specified grantType.
	RequestToken(grantType, value string) (oauth2.TokenResponse, error)
}

type oidcClient struct {
	oidc *oidc.Client
	oac  *oauth2.Client
}

func (o *oidcClient) RefreshToken(rt string) (jose.JWT, error) {
	return o.oidc.RefreshToken(rt)
}

func (o *oidcClient) AuthCodeURL(state, accessType, prompt string) string {
	return o.oac.AuthCodeURL(state, accessType, prompt)
}

func (o *oidcClient) RequestToken(grantType, value string) (oauth2.TokenResponse, error) {
	return o.oac.RequestToken(grantType, value)
}

// New creates a new request Authenticator which uses OpenID Connect ID Tokens for authentication.
//
// Internally, an OIDC client with the given issuerURL, clientID and
// clientSecret is created.
//
// A path to a Certificate Authority file can be provided for communicating with
// the Issuer. If none is provided, the hosts's root CA set will be used.
//
// NOTE(yifan): For now we assume the server provides the "jwks_uri" so we don't
// need to manager the key sets by ourselves.
func New(issuer, clientID, clientSecret, caFile, usernameClaim, groupsClaim string) (*OIDCAuthenticator, error) {
	var cfg oidc.ProviderConfig
	var err error
	var roots *x509.CertPool

	issuerURL, err := url.Parse(issuer)
	if err != nil {
		return nil, err
	}

	if issuerURL.Scheme != "https" {
		return nil, fmt.Errorf("'oidc-issuer-url' (%q) has invalid scheme (%q), require 'https'",
			issuer, issuerURL.Scheme)
	}

	if caFile != "" {
		roots, err = util.CertPoolFromFile(caFile)
		if err != nil {
			glog.Errorf("Failed to read the CA file: %v", err)
		}
	}
	if roots == nil {
		glog.Info("No x509 certificates provided, will use host's root CA set")
	}

	// Copied from http.DefaultTransport.
	tr := net.SetTransportDefaults(&http.Transport{
		// According to golang's doc, if RootCAs is nil,
		// TLS uses the host's root CA set.
		TLSClientConfig: &tls.Config{RootCAs: roots},
	})

	hc := &http.Client{}
	hc.Transport = tr

	for i := 0; i <= maxRetries; i++ {
		if i == maxRetries {
			return nil, fmt.Errorf("failed to fetch provider config after %v retries", maxRetries)
		}

		cfg, err = oidc.FetchProviderConfig(hc, strings.TrimSuffix(issuer, "/"))
		if err == nil {
			break
		}
		glog.Errorf("Failed to fetch provider config, trying again in %v: %v", retryBackoff, err)
		time.Sleep(retryBackoff)
	}

	glog.Infof("Fetched provider config from %s: %#v", issuer, cfg)

	ccfg := oidc.ClientConfig{
		HTTPClient: hc,
		Credentials: oidc.ClientCredentials{
			ID:     clientID,
			Secret: clientSecret,
		},
		Scope:          append(oidc.DefaultScope, "offline_access"),
		ProviderConfig: cfg,
	}

	client, err := oidc.NewClient(ccfg)
	if err != nil {
		return nil, err
	}

	// SyncProviderConfig will start a goroutine to periodically synchronize the provider config.
	// The synchronization interval is set by the expiration length of the config, and has a mininum
	// and maximum threshold.
	stop := client.SyncProviderConfig(issuer)

	authn := &OIDCAuthenticator{
		clientConfig:     ccfg,
		client:           client,
		usernameClaim:    usernameClaim,
		groupsClaim:      groupsClaim,
		stopSyncProvider: stop,
	}
	authn.bearerAuthenticator = bearertoken.New(authn)
	return authn, nil
}

// SetHandlersAttached tells the authenticator that the OIDC related HTTP handlers are in use and have been attached.
//
// When this is set to true, requests to the following paths will not require
// authentication:
//
// PathExchangeRefreshToken, PathAuthenticate, PathAuthCallback,
// PathExchangeCode
func (a *OIDCAuthenticator) SetHandlersAttached(attached bool) {
	a.handlersAttached = attached
}

// NewOIDCHTTPHandler creates an http.Handler which provides additional endpoints useful for OIDC-related authentication.
//
// An OIDCAuthenticator created with a client-secret is necessary, or else an
// error is returned.
func (a *OIDCAuthenticator) NewOIDCHTTPHandler(apiHostPort string) (*OIDCHTTPHandler, error) {
	apiServerURL := url.URL{
		Host:   apiHostPort,
		Scheme: "https",
	}

	oac, err := a.client.OAuthClient()
	if err != nil {
		return nil, err
	}

	client := &oidcClient{
		oidc: a.client,
		oac:  oac,
	}

	if a.clientConfig.Credentials.Secret == "" {
		return nil, errors.New("OIDCHTTPHandler requires a client config with a client secret")
	}

	return &OIDCHTTPHandler{
		client:       client,
		apiServerURL: apiServerURL,
	}, nil
}

func NewOIDCHTTPHandler(o OIDCClient) *OIDCHTTPHandler {
	return &OIDCHTTPHandler{
		client: o,
	}
}

func (a *OIDCAuthenticator) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	if a.handlersAttached {
		if _, ok := bypassAuthPaths[req.URL.Path]; ok {
			return UserUnauthenticated, true, nil
		}
	}
	return a.bearerAuthenticator.AuthenticateRequest(req)
}

// AuthenticateToken decodes and verifies a JWT using the OIDC client, if the verification succeeds,
// then it will extract the user info from the JWT claims.
func (a *OIDCAuthenticator) AuthenticateToken(value string) (user.Info, bool, error) {
	jwt, err := jose.ParseJWT(value)
	if err != nil {
		return nil, false, err
	}

	if err := a.client.VerifyJWT(jwt); err != nil {
		return nil, false, err
	}

	claims, err := jwt.Claims()
	if err != nil {
		return nil, false, err
	}

	claim, ok, err := claims.StringClaim(a.usernameClaim)
	if err != nil {
		return nil, false, err
	}
	if !ok {
		return nil, false, fmt.Errorf("cannot find %q in JWT claims", a.usernameClaim)
	}

	var username string
	switch a.usernameClaim {
	case "email":
		// TODO(yifan): Check 'email_verified' to make sure the email is valid.
		username = claim
	default:
		// For all other cases, use issuerURL + claim as the user name.
		username = fmt.Sprintf("%s#%s", a.clientConfig.ProviderConfig.Issuer, claim)
	}

	// TODO(yifan): Add UID, also populate the issuer to upper layer.
	info := &user.DefaultInfo{Name: username}

	if a.groupsClaim != "" {
		groups, found, err := claims.StringsClaim(a.groupsClaim)
		if err != nil {
			// Custom claim is present, but isn't an array of strings.
			return nil, false, fmt.Errorf("custom group claim contains invalid object: %v", err)
		}
		if found {
			info.Groups = groups
		}
	}
	return info, true, nil
}

// Close closes the OIDC authenticator, this will close the provider sync goroutine.
func (a *OIDCAuthenticator) Close() {
	// This assumes the s.stopSyncProvider is an unbuffered channel.
	// So instead of closing the channel, we send am empty struct here.
	// This guarantees that when this function returns, there is no flying requests,
	// because a send to an unbuffered channel happens after the receive from the channel.
	a.stopSyncProvider <- struct{}{}
}

func (t *OIDCHTTPHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	glog.Errorf("SERVE IT:")

	switch r.URL.Path {
	case PathExchangeRefreshToken:
		t.handleExchangeRefreshToken(w, r)
		return
	case PathAuthenticate:
		t.handleAuth(w, r)
		return
	case PathAuthCallback:
		t.handleAuthCallback(w, r)
		return
	case PathExchangeCode:
		t.handleExchangeCode(w, r)
		return
	default:
		w.WriteHeader(http.StatusNotFound)
		return
	}
}

// handleExchangeRefreshToken exchanges a refresh token, passed through the "refresh_token" parameter, for an ID Token, which can be used for authentication.
func (t *OIDCHTTPHandler) handleExchangeRefreshToken(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if err := r.ParseForm(); err != nil {
		glog.Errorf("could not parse form: %v", err)
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	rt := r.PostForm.Get("refresh_token")
	if rt == "" {
		glog.Errorf("Missing 'refresh_token' parameter")
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	// Get the JWT
	jwt, err := t.client.RefreshToken(rt)
	if err != nil {
		glog.Errorf("Could not get refresh token: %v", err)
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	v := url.Values{}
	v.Set("id_token", jwt.Encode())
	_, err = w.Write([]byte(v.Encode()))
	if err != nil {
		glog.Errorf("Could not write response: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	return
}

// handleAuth redirects the user-agent to the identity-provider for authentication.
// A 'callback' parameter must be present in the query string, and it must be
// URL with a host of either "localhost" or 127.0.0.1.
// After authenticating, the user will be redirected back to the API Server to
// be handled by the "handleAuthCallback" handler, which redirects the
// user-agent back to the URL that was specified in the 'callback' parameter
// passed to this handler.
func (t *OIDCHTTPHandler) handleAuth(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	// Get the URL, ensure that it is localhost
	cb := r.URL.Query().Get("callback")
	if cb == "" {
		glog.Errorf("Missing 'callback' parameter in request: %v")
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	cbURL, err := url.Parse(cb)
	if err != nil {
		glog.Errorf("%s not parseable as URL: %v", cb, err)
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	if !isLocalAddr(cbURL.Host) {
		glog.Errorf("'callback' must be localhost, not %v", cbURL.Host)
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	redirect := t.apiServerURL
	redirect.Path = PathAuthCallback

	authURL, err := t.getAuthURL(cb, redirect.String())
	if err != nil {
		glog.Errorf("Could not get an auth URL: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	http.Redirect(w, r, authURL, http.StatusSeeOther)
	return
}

// handleAuthCallback is the callback handler that the user-agent is redirected to after authentication.
// This handler verifies that the 'state' paramter contains a valid URL to a
// loopback address and then redirects the user there along with their
// authorization code from the Idp, where it is expected that they will then
// make a request back to the API Server to exchange the code for an ID token
// via the handleExchangeToken handler.
func (t *OIDCHTTPHandler) handleAuthCallback(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	code := r.URL.Query().Get("code")
	if code == "" {
		glog.Errorf("Missing 'code' paramter in request")
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	state := r.URL.Query().Get("state")
	if state == "" {
		glog.Errorf("Missing 'state' paramter in request")
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	redirURL, err := url.Parse(state)
	if err != nil {
		glog.Errorf("'state' paramter not parseable as URL")
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	if !isLocalAddr(redirURL.Host) {
		glog.Errorf("URL in 'state' must be localhost, not %v", redirURL.Host)
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	q := redirURL.Query()
	q.Set("code", code)
	redirURL.RawQuery = q.Encode()

	http.Redirect(w, r, redirURL.String(), http.StatusSeeOther)
}

// handleExchangeCode exchanges an authorization code (passed through the 'code' parameter) for an ID Token and refresh token, which are then written to the body of the response in a url-encoded form.
func (t *OIDCHTTPHandler) handleExchangeCode(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	if err := r.ParseForm(); err != nil {
		glog.Errorf("could not parse form: %v", err)
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	code := r.PostForm.Get("code")
	if code == "" {
		glog.Errorf("Missing 'code' paramter in request")
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	tokRes, err := t.client.RequestToken(oauth2.GrantTypeAuthCode, code)
	if err != nil {
		glog.Errorf("Could not exchange token: %v  ", err)
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	idToken, err := jose.ParseJWT(tokRes.IDToken)
	if err != nil {
		glog.Errorf("Unable to parse JWT %v:  ", err)
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	refreshToken := tokRes.RefreshToken

	params := url.Values{}
	params.Set("refresh_token", refreshToken)
	params.Set("id_token", idToken.Encode())
	_, err = w.Write([]byte(params.Encode()))
	if err != nil {
		glog.Errorf("Could not write response: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
}

func (t *OIDCHTTPHandler) getAuthURL(cb string, redirect string) (string, error) {
	acURLStr := t.client.AuthCodeURL(cb, "offline", "")
	acURL, err := url.Parse(acURLStr)
	if err != nil {
		return "", err
	}
	q := acURL.Query()
	q.Set("redirect_uri", redirect)
	acURL.RawQuery = q.Encode()
	return acURL.String(), nil
}

func isLocalAddr(host string) bool {
	host = strings.Split(host, ":")[0]
	return host == "localhost" || host == "127.0.0.1"
}
