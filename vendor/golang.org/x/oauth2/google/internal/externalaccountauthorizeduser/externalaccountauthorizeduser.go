// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package externalaccountauthorizeduser

import (
	"context"
	"errors"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google/internal/stsexchange"
)

// now aliases time.Now for testing.
var now = func() time.Time {
	return time.Now().UTC()
}

var tokenValid = func(token oauth2.Token) bool {
	return token.Valid()
}

type Config struct {
	// Audience is the Secure Token Service (STS) audience which contains the resource name for the workforce pool and
	// the provider identifier in that pool.
	Audience string
	// RefreshToken is the optional OAuth 2.0 refresh token. If specified, credentials can be refreshed.
	RefreshToken string
	// TokenURL is the optional STS token exchange endpoint for refresh. Must be specified for refresh, can be left as
	// None if the token can not be refreshed.
	TokenURL string
	// TokenInfoURL is the optional STS endpoint URL for token introspection.
	TokenInfoURL string
	// ClientID is only required in conjunction with ClientSecret, as described above.
	ClientID string
	// ClientSecret is currently only required if token_info endpoint also needs to be called with the generated GCP
	// access token. When provided, STS will be called with additional basic authentication using client_id as username
	// and client_secret as password.
	ClientSecret string
	// Token is the OAuth2.0 access token. Can be nil if refresh information is provided.
	Token string
	// Expiry is the optional expiration datetime of the OAuth 2.0 access token.
	Expiry time.Time
	// RevokeURL is the optional STS endpoint URL for revoking tokens.
	RevokeURL string
	// QuotaProjectID is the optional project ID used for quota and billing. This project may be different from the
	// project used to create the credentials.
	QuotaProjectID string
	Scopes         []string
}

func (c *Config) canRefresh() bool {
	return c.ClientID != "" && c.ClientSecret != "" && c.RefreshToken != "" && c.TokenURL != ""
}

func (c *Config) TokenSource(ctx context.Context) (oauth2.TokenSource, error) {
	var token oauth2.Token
	if c.Token != "" && !c.Expiry.IsZero() {
		token = oauth2.Token{
			AccessToken: c.Token,
			Expiry:      c.Expiry,
			TokenType:   "Bearer",
		}
	}
	if !tokenValid(token) && !c.canRefresh() {
		return nil, errors.New("oauth2/google: Token should be created with fields to make it valid (`token` and `expiry`), or fields to allow it to refresh (`refresh_token`, `token_url`, `client_id`, `client_secret`).")
	}

	ts := tokenSource{
		ctx:  ctx,
		conf: c,
	}

	return oauth2.ReuseTokenSource(&token, ts), nil
}

type tokenSource struct {
	ctx  context.Context
	conf *Config
}

func (ts tokenSource) Token() (*oauth2.Token, error) {
	conf := ts.conf
	if !conf.canRefresh() {
		return nil, errors.New("oauth2/google: The credentials do not contain the necessary fields need to refresh the access token. You must specify refresh_token, token_url, client_id, and client_secret.")
	}

	clientAuth := stsexchange.ClientAuthentication{
		AuthStyle:    oauth2.AuthStyleInHeader,
		ClientID:     conf.ClientID,
		ClientSecret: conf.ClientSecret,
	}

	stsResponse, err := stsexchange.RefreshAccessToken(ts.ctx, conf.TokenURL, conf.RefreshToken, clientAuth, nil)
	if err != nil {
		return nil, err
	}
	if stsResponse.ExpiresIn < 0 {
		return nil, errors.New("oauth2/google: got invalid expiry from security token service")
	}

	if stsResponse.RefreshToken != "" {
		conf.RefreshToken = stsResponse.RefreshToken
	}

	token := &oauth2.Token{
		AccessToken: stsResponse.AccessToken,
		Expiry:      now().Add(time.Duration(stsResponse.ExpiresIn) * time.Second),
		TokenType:   "Bearer",
	}
	return token, nil
}
