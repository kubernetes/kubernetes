/*
Copyright 2016 The Kubernetes Authors.

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
	"net/http"
	"strings"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/oauth2"
	"github.com/coreos/go-oidc/oidc"
	"github.com/golang/glog"

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

	scopes := strings.Split(cfg[cfgExtraScopes], ",")
	oidcCfg := oidc.ClientConfig{
		HTTPClient: hc,
		Credentials: oidc.ClientCredentials{
			ID:     clientID,
			Secret: clientSecret,
		},
		ProviderConfig: providerCfg,
		Scope:          append(scopes, oidc.DefaultScope...),
	}

	client, err := oidc.NewClient(oidcCfg)
	if err != nil {
		return nil, fmt.Errorf("error creating OIDC Client: %v", err)
	}

	oClient := &oidcClient{client}

	var initialIDToken jose.JWT
	if cfg[cfgIDToken] != "" {
		initialIDToken, err = jose.ParseJWT(cfg[cfgIDToken])
		if err != nil {
			return nil, err
		}
	}

	return &oidcAuthProvider{
		initialIDToken: initialIDToken,
		refresher: &idTokenRefresher{
			client:    oClient,
			cfg:       cfg,
			persister: persister,
		},
	}, nil
}

type oidcAuthProvider struct {
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
	return errors.New("not yet implemented")
}

type OIDCClient interface {
	refreshToken(rt string) (oauth2.TokenResponse, error)
	verifyJWT(jwt jose.JWT) error
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
	client *oidc.Client
}

func (o *oidcClient) refreshToken(rt string) (oauth2.TokenResponse, error) {
	oac, err := o.client.OAuthClient()
	if err != nil {
		return oauth2.TokenResponse{}, err
	}

	return oac.RequestToken(oauth2.GrantTypeRefreshToken, rt)
}

func (o *oidcClient) verifyJWT(jwt jose.JWT) error {
	return o.client.VerifyJWT(jwt)
}
