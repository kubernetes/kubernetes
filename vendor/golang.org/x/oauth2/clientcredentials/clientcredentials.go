// Copyright 2014 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package clientcredentials implements the OAuth2.0 "client credentials" token flow,
// also known as the "two-legged OAuth 2.0".
//
// This should be used when the client is acting on its own behalf or when the client
// is the resource owner. It may also be used when requesting access to protected
// resources based on an authorization previously arranged with the authorization
// server.
//
// See http://tools.ietf.org/html/draft-ietf-oauth-v2-31#section-4.4
package clientcredentials // import "golang.org/x/oauth2/clientcredentials"

import (
	"net/http"
	"net/url"
	"strings"

	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/internal"
)

// tokenFromInternal maps an *internal.Token struct into
// an *oauth2.Token struct.
func tokenFromInternal(t *internal.Token) *oauth2.Token {
	if t == nil {
		return nil
	}
	tk := &oauth2.Token{
		AccessToken:  t.AccessToken,
		TokenType:    t.TokenType,
		RefreshToken: t.RefreshToken,
		Expiry:       t.Expiry,
	}
	return tk.WithExtra(t.Raw)
}

// retrieveToken takes a *Config and uses that to retrieve an *internal.Token.
// This token is then mapped from *internal.Token into an *oauth2.Token which is
// returned along with an error.
func retrieveToken(ctx context.Context, c *Config, v url.Values) (*oauth2.Token, error) {
	tk, err := internal.RetrieveToken(ctx, c.ClientID, c.ClientSecret, c.TokenURL, v)
	if err != nil {
		return nil, err
	}
	return tokenFromInternal(tk), nil
}

// Client Credentials Config describes a 2-legged OAuth2 flow, with both the
// client application information and the server's endpoint URLs.
type Config struct {
	// ClientID is the application's ID.
	ClientID string

	// ClientSecret is the application's secret.
	ClientSecret string

	// TokenURL is the resource server's token endpoint
	// URL. This is a constant specific to each server.
	TokenURL string

	// Scope specifies optional requested permissions.
	Scopes []string
}

// Token uses client credentials to retreive a token.
// The HTTP client to use is derived from the context.
// If nil, http.DefaultClient is used.
func (c *Config) Token(ctx context.Context) (*oauth2.Token, error) {
	return retrieveToken(ctx, c, url.Values{
		"grant_type": {"client_credentials"},
		"scope":      internal.CondVal(strings.Join(c.Scopes, " ")),
	})
}

// Client returns an HTTP client using the provided token.
// The token will auto-refresh as necessary. The underlying
// HTTP transport will be obtained using the provided context.
// The returned client and its Transport should not be modified.
func (c *Config) Client(ctx context.Context) *http.Client {
	return oauth2.NewClient(ctx, c.TokenSource(ctx))
}

// TokenSource returns a TokenSource that returns t until t expires,
// automatically refreshing it as necessary using the provided context and the
// client ID and client secret.
//
// Most users will use Config.Client instead.
func (c *Config) TokenSource(ctx context.Context) oauth2.TokenSource {
	source := &tokenSource{
		ctx:  ctx,
		conf: c,
	}
	return oauth2.ReuseTokenSource(nil, source)
}

type tokenSource struct {
	ctx  context.Context
	conf *Config
}

// Token refreshes the token by using a new client credentials request.
// tokens received this way do not include a refresh token
func (c *tokenSource) Token() (*oauth2.Token, error) {
	return retrieveToken(c.ctx, c.conf, url.Values{
		"grant_type": {"client_credentials"},
		"scope":      internal.CondVal(strings.Join(c.conf.Scopes, " ")),
	})
}
