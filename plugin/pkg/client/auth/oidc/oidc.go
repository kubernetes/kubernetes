/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package oidc

import (
	"encoding/base64"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/oauth2"
	"github.com/coreos/go-oidc/oidc"
	"github.com/golang/glog"
	"github.com/pkg/browser"

	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	cfgIssuerUrl                = "idp-issuer-url"
	cfgClientID                 = "client-id"
	cfgClientSecret             = "client-secret"
	cfgCertificateAuthority     = "idp-certificate-authority"
	cfgCertificateAuthorityData = "idp-certificate-authority-data"
	cfgExtraScopes              = "extra-scopes"
	cfgIDToken                  = "id-token"
	cfgRefreshToken             = "refresh-token"
	cfgRedirectUri              = "redirect-uri"
	cfgOffline                  = "offline-access"
	cbOOB                       = "urn:ietf:wg:oauth:2.0:oob"
)

var (
	backoff = wait.Backoff{
		Duration: 1 * time.Second,
		Factor:   2,
		Jitter:   .1,
		Steps:    5,
	}
)

func init() {
	if err := restclient.RegisterAuthProviderPlugin("oidc", newOIDCAuthProvider); err != nil {
		glog.Fatalf("Failed to register oidc auth plugin: %v", err)
	}
}

func newOIDCAuthProvider(_ string, cfg map[string]string, persister restclient.AuthProviderConfigPersister) (restclient.AuthProvider, error) {
	issuer := cfg[cfgIssuerUrl]
	if issuer == "" {
		return nil, fmt.Errorf("Must provide %s", cfgIssuerUrl)
	}

	clientID := cfg[cfgClientID]
	if clientID == "" {
		return nil, fmt.Errorf("Must provide %s", cfgClientID)
	}

	clientSecret := cfg[cfgClientSecret]
	if clientSecret == "" {
		return nil, fmt.Errorf("Must provide %s", cfgClientSecret)
	}

	var certAuthData []byte
	var err error
	if cfg[cfgCertificateAuthorityData] != "" {
		certAuthData, err = base64.StdEncoding.DecodeString(cfg[cfgCertificateAuthorityData])
		if err != nil {
			return nil, err
		}
	}

	clientConfig := restclient.Config{
		TLSClientConfig: restclient.TLSClientConfig{
			CAFile: cfg[cfgCertificateAuthority],
			CAData: certAuthData,
		},
	}

	trans, err := restclient.TransportFor(&clientConfig)
	if err != nil {
		return nil, err
	}
	hc := &http.Client{Transport: trans}

	providerCfg, err := oidc.FetchProviderConfig(hc, strings.TrimSuffix(issuer, "/"))
	if err != nil {
		return nil, fmt.Errorf("error fetching provider config: %v", err)
	}

	redirect := cfg[cfgRedirectUri]
	var oob bool
	var port int
	if redirect != "" {
		oob, port, err = parseRedirect(redirect)
		if err != nil {
			return nil, err
		}
	}

	scopes := strings.Split(cfg[cfgExtraScopes], ",")

	offline := cfg[cfgOffline] == "true"
	if offline {
		// NOTE: The Google OIDC Issuer will complain with the added
		// "offline_access" scope
		// (http://openid.net/specs/openid-connect-core-1_0.html#OfflineAccess),
		// so we handle the Google case slightly differently, adding
		// "access_type=offline" to the query string of the auth request
		if !isGoogle(issuer) {
			scopes = append(scopes, "offline_access")
		}
	}

	oidcCfg := oidc.ClientConfig{
		HTTPClient: hc,
		Credentials: oidc.ClientCredentials{
			ID:     clientID,
			Secret: clientSecret,
		},
		ProviderConfig: providerCfg,
		Scope:          append(scopes, oidc.DefaultScope...),
		RedirectURL:    redirect,
	}

	client, err := oidc.NewClient(oidcCfg)
	if err != nil {
		return nil, fmt.Errorf("error creating OIDC Client: %v", err)
	}

	oauthClient, err := client.OAuthClient()
	if err != nil {
		return nil, fmt.Errorf("error creating OAuth2 Client: %v", err)
	}

	oClient := &oidcClient{
		client:      client,
		oauthClient: oauthClient,
	}

	var initialIDToken jose.JWT
	if cfg[cfgIDToken] != "" {
		initialIDToken, err = jose.ParseJWT(cfg[cfgIDToken])
		if err != nil {
			return nil, err
		}
	}

	return &oidcAuthProvider{
		offline:        offline,
		redirectPort:   port,
		redirectOOB:    oob,
		initialIDToken: initialIDToken,
		refresher: &idTokenRefresher{
			client:    oClient,
			cfg:       cfg,
			persister: persister,
		},
	}, nil
}

type oidcAuthProvider struct {
	offline        bool
	redirectPort   int
	redirectOOB    bool
	refresher      *idTokenRefresher
	initialIDToken jose.JWT
}

func (g *oidcAuthProvider) WrapTransport(rt http.RoundTripper) http.RoundTripper {
	at := &oidc.AuthenticatedTransport{
		TokenRefresher: g.refresher,
		RoundTripper:   rt,
	}
	at.SetJWT(g.initialIDToken)
	return &roundTripper{
		wrapped:   at,
		refresher: g.refresher,
	}
}

func (g *oidcAuthProvider) Login() error {
	if !g.redirectOOB && g.redirectPort == 0 {
		return errors.New("Cannot login without a '" + cfgRedirectUri + "' set")
	}

	var code string
	var reqErr error
	var wg sync.WaitGroup
	if !g.redirectOOB {
		listener, err := net.Listen("tcp", fmt.Sprintf("localhost:%d", g.redirectPort))
		if err != nil {
			return err
		}

		// This server waits for the redirect coming back from API server, populates
		// code and reqErr from that request, and then stops itself.
		srv := &http.Server{
			Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// This is to handle unwanted but inevitable requests, like for
				// "favicon.ico"
				if r.URL.Path != "/" {
					return
				}

				// Stop listening once we've gotten a request.
				listener.Close()
				if r.Method != "GET" {
					reqErr = errors.New("The server made a bad request: Only GET is allowed")
				}

				code = r.URL.Query().Get("code")
				if code == "" {
					reqErr = errors.New("Missing 'code' parameter from server.")
				}

				var msg string
				if reqErr == nil {
					msg = "Login Successful!"
				} else {
					msg = reqErr.Error()
				}
				w.Write([]byte(fmt.Sprintf(authPostLoginTpl, msg)))
				wg.Done()
			}),
		}
		wg.Add(1)
		go srv.Serve(listener)
	}

	// NOTE: Google gets handlded slightly differently when requesting offline
	// access - see earlier NOTE.
	var accessType string
	if g.offline && isGoogle(g.refresher.cfg[cfgIssuerUrl]) {
		accessType = "offline"
	}
	authURL := g.refresher.client.authCodeURL(accessType)

	err := browser.OpenURL(authURL)
	if err != nil {
		return err
	}

	if g.redirectOOB {
		return nil
	}
	wg.Wait()

	if reqErr != nil {
		return reqErr
	}

	tokens, err := g.refresher.client.exchangeCode(code)
	if err != nil {
		return fmt.Errorf("error exchanging auth code: %v", err)
	}

	jwt, err := jose.ParseJWT(tokens.IDToken)
	if err != nil {
		return err
	}

	cfg := g.refresher.cfg
	if tokens.RefreshToken != "" {
		cfg[cfgRefreshToken] = tokens.RefreshToken
	}
	cfg[cfgIDToken] = jwt.Encode()
	err = g.refresher.persister.Persist(cfg)
	if err != nil {
		return fmt.Errorf("could not perist new tokens: %v", err)
	}

	return nil
}

type OIDCClient interface {
	refreshToken(rt string) (oauth2.TokenResponse, error)
	exchangeCode(code string) (oauth2.TokenResponse, error)
	verifyJWT(jwt jose.JWT) error
	authCodeURL(accessType string) string
}

type roundTripper struct {
	refresher *idTokenRefresher
	wrapped   *oidc.AuthenticatedTransport
}

func (r *roundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	var res *http.Response
	var err error
	firstTime := true
	wait.ExponentialBackoff(backoff, func() (bool, error) {
		if !firstTime {
			var jwt jose.JWT
			jwt, err = r.refresher.Refresh()
			if err != nil {
				return true, nil
			}
			r.wrapped.SetJWT(jwt)
		} else {
			firstTime = false
		}

		res, err = r.wrapped.RoundTrip(req)
		if err != nil {
			return true, nil
		}
		if res.StatusCode == http.StatusUnauthorized {
			return false, nil
		}
		return true, nil
	})
	return res, err
}

type idTokenRefresher struct {
	cfg           map[string]string
	client        OIDCClient
	persister     restclient.AuthProviderConfigPersister
	intialIDToken jose.JWT
}

func (r *idTokenRefresher) Verify(jwt jose.JWT) error {
	claims, err := jwt.Claims()
	if err != nil {
		return err
	}

	now := time.Now()
	exp, ok, err := claims.TimeClaim("exp")
	switch {
	case err != nil:
		return fmt.Errorf("failed to parse 'exp' claim: %v", err)
	case !ok:
		return errors.New("missing required 'exp' claim")
	case exp.Before(now):
		return fmt.Errorf("token already expired at: %v", exp)
	}

	return nil
}

func (r *idTokenRefresher) Refresh() (jose.JWT, error) {
	rt, ok := r.cfg[cfgRefreshToken]
	if !ok {
		return jose.JWT{}, errors.New("No valid id-token, and cannot refresh without refresh-token")
	}

	tokens, err := r.client.refreshToken(rt)
	if err != nil {
		return jose.JWT{}, fmt.Errorf("could not refresh token: %v", err)
	}
	jwt, err := jose.ParseJWT(tokens.IDToken)
	if err != nil {
		return jose.JWT{}, err
	}

	if tokens.RefreshToken != "" && tokens.RefreshToken != rt {
		r.cfg[cfgRefreshToken] = tokens.RefreshToken
	}
	r.cfg[cfgIDToken] = jwt.Encode()

	err = r.persister.Persist(r.cfg)
	if err != nil {
		return jose.JWT{}, fmt.Errorf("could not perist new tokens: %v", err)
	}

	return jwt, r.client.verifyJWT(jwt)
}

type oidcClient struct {
	client      *oidc.Client
	oauthClient *oauth2.Client
}

func (o *oidcClient) refreshToken(rt string) (oauth2.TokenResponse, error) {
	oac, err := o.client.OAuthClient()
	if err != nil {
		return oauth2.TokenResponse{}, err
	}

	return oac.RequestToken(oauth2.GrantTypeRefreshToken, rt)
}

func (o *oidcClient) exchangeCode(code string) (oauth2.TokenResponse, error) {
	oac, err := o.client.OAuthClient()
	if err != nil {
		return oauth2.TokenResponse{}, err
	}

	return oac.RequestToken(oauth2.GrantTypeAuthCode, code)
}

func (o *oidcClient) verifyJWT(jwt jose.JWT) error {
	return o.client.VerifyJWT(jwt)
}

func (o *oidcClient) authCodeURL(accessType string) string {
	return o.oauthClient.AuthCodeURL("", accessType, "")
}

// parseRedirect returns whether or not the redirect url is for an OOB flow, the port number for localhost redirects, and an error if the string is an invalid redirect.
// Valid redirects are either "urn:ietf:wg:oauth:2.0:oob" or a localhost:$port
// url. The "oob:auto" case is disallowed because this implementation doesn't
// cannot read the title bar of the browser window to determine the code.
func parseRedirect(r string) (bool, int, error) {
	if r == cbOOB {
		return true, 0, nil
	}

	u, err := url.Parse(r)
	if err != nil {
		return false, 0, fmt.Errorf("invalid %v: %v", cfgRedirectUri, err)
	}

	hostPort := strings.Split(u.Host, ":")
	if hostPort[0] != "localhost" {
		return false, 0, errors.New("Host must be 'localhost' in " + cfgRedirectUri)
	}

	if u.Scheme != "http" {
		return false, 0, errors.New("Scheme must be 'http' in " + cfgRedirectUri)
	}

	port, err := strconv.ParseInt(hostPort[1], 10, 0)
	if err != nil {
		return false, 0, fmt.Errorf("Could not parse port in %v", r)
	}

	return false, int(port), nil
}

func isGoogle(s string) bool {
	return strings.TrimRight(s, "/") == "https://accounts.google.com"
}

const authPostLoginTpl = `
  <body>
    %v
    <br>
    You can now close this window.
  </body>
</html>`
