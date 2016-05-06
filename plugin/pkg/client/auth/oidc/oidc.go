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
	"net/http"
	"strings"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/oauth2"
	"github.com/coreos/go-oidc/oidc"
	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/client/restclient"
)

const (
	cfgIssuerUrl                = "idp-issuer-url"
	cfgClientId                 = "client-id"
	cfgClientSecret             = "client-secret"
	cfgCertificateAuthority     = "idp-certificate-authority"
	cfgCertificateAuthorityData = "idp-certificate-authority-data"
	cfgExtraScopes              = "extra-scopes"
	cfgIdToken                  = "id-token"
	cfgRefreshToken             = "refresh-token"
)

func init() {
	if err := restclient.RegisterAuthProviderPlugin("oidc", newOIDCAuthProvider); err != nil {
		glog.Fatalf("Failed to register oidc auth plugin: %v", err)
	}
}

func newOIDCAuthProvider(_ string, cfg map[string]string, persister restclient.AuthProviderConfigPersister) (restclient.AuthProvider, error) {
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

	issuer, ok := cfg[cfgIssuerUrl]
	if !ok || issuer == "" {
		return nil, errors.New("Must provide idp-issuer-url")
	}

	providerCfg, err := oidc.FetchProviderConfig(hc, strings.TrimSuffix(issuer, "/"))
	if err != nil {
		return nil, fmt.Errorf("error fetching provider config: %v", err)
	}

	scopes := strings.Split(cfg[cfgExtraScopes], ",")
	oidcCfg := oidc.ClientConfig{
		HTTPClient: hc,
		Credentials: oidc.ClientCredentials{
			ID:     cfg[cfgClientId],
			Secret: cfg[cfgClientSecret],
		},
		ProviderConfig: providerCfg,
		Scope:          scopes,
	}

	client, err := oidc.NewClient(oidcCfg)
	if err != nil {
		return nil, fmt.Errorf("error creating OIDC Client: %v", err)
	}

	var initialIDToken jose.JWT
	if cfg[cfgIdToken] != "" {
		initialIDToken, err = jose.ParseJWT(cfg[cfgIdToken])
		if err != nil {
			return nil, err
		}
	}

	return &oidcAuthProvider{
		intialIDToken: initialIDToken,
		refresher: &idTokenRefresher{
			client:    client,
			cfg:       cfg,
			persister: persister,
		},
	}, nil
}

type oidcAuthProvider struct {
	refresher     *idTokenRefresher
	intialIDToken jose.JWT
}

func (g *oidcAuthProvider) WrapTransport(rt http.RoundTripper) http.RoundTripper {
	at := &oidc.AuthenticatedTransport{
		TokenRefresher: g.refresher,
		RoundTripper:   rt,
	}
	at.SetJWT(g.intialIDToken)
	return at
}

func (g *oidcAuthProvider) Login() error {
	return errors.New("not yet implemented")
}

type idTokenRefresher struct {
	cfg           map[string]string
	client        *oidc.Client
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

	oac, err := r.client.OAuthClient()
	if err != nil {
		return jose.JWT{}, err
	}

	tokens, err := oac.RequestToken(oauth2.GrantTypeRefreshToken, rt)
	if err != nil {
		return jose.JWT{}, err
	}

	jwt, err := jose.ParseJWT(tokens.IDToken)
	if err != nil {
		return jose.JWT{}, err
	}

	if tokens.RefreshToken != "" && tokens.RefreshToken != rt {
		r.cfg[cfgRefreshToken] = tokens.RefreshToken
	}
	r.cfg[cfgIdToken] = jwt.Encode()

	err = r.persister.Persist(r.cfg)
	if err != nil {
		return jose.JWT{}, fmt.Errorf("could not perist new tokens: %v", err)
	}

	return jwt, r.client.VerifyJWT(jwt)
}
