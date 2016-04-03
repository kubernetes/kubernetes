// Copyright 2014 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oauth2

import (
	"net/http"
	"net/url"
	"strings"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/oauth2/internal"
)

// expiryDelta determines how earlier a token should be considered
// expired than its actual expiration time. It is used to avoid late
// expirations due to client-server time mismatches.
const expiryDelta = 10 * time.Second

// Token represents the crendentials used to authorize
// the requests to access protected resources on the OAuth 2.0
// provider's backend.
//
// Most users of this package should not access fields of Token
// directly. They're exported mostly for use by related packages
// implementing derivative OAuth2 flows.
type Token struct {
	// AccessToken is the token that authorizes and authenticates
	// the requests.
	AccessToken string `json:"access_token"`

	// TokenType is the type of token.
	// The Type method returns either this or "Bearer", the default.
	TokenType string `json:"token_type,omitempty"`

	// RefreshToken is a token that's used by the application
	// (as opposed to the user) to refresh the access token
	// if it expires.
	RefreshToken string `json:"refresh_token,omitempty"`

	// Expiry is the optional expiration time of the access token.
	//
	// If zero, TokenSource implementations will reuse the same
	// token forever and RefreshToken or equivalent
	// mechanisms for that TokenSource will not be used.
	Expiry time.Time `json:"expiry,omitempty"`

	// raw optionally contains extra metadata from the server
	// when updating a token.
	raw interface{}
}

// Type returns t.TokenType if non-empty, else "Bearer".
func (t *Token) Type() string {
	if strings.EqualFold(t.TokenType, "bearer") {
		return "Bearer"
	}
	if strings.EqualFold(t.TokenType, "mac") {
		return "MAC"
	}
	if strings.EqualFold(t.TokenType, "basic") {
		return "Basic"
	}
	if t.TokenType != "" {
		return t.TokenType
	}
	return "Bearer"
}

// SetAuthHeader sets the Authorization header to r using the access
// token in t.
//
// This method is unnecessary when using Transport or an HTTP Client
// returned by this package.
func (t *Token) SetAuthHeader(r *http.Request) {
	r.Header.Set("Authorization", t.Type()+" "+t.AccessToken)
}

// WithExtra returns a new Token that's a clone of t, but using the
// provided raw extra map. This is only intended for use by packages
// implementing derivative OAuth2 flows.
func (t *Token) WithExtra(extra interface{}) *Token {
	t2 := new(Token)
	*t2 = *t
	t2.raw = extra
	return t2
}

// Extra returns an extra field.
// Extra fields are key-value pairs returned by the server as a
// part of the token retrieval response.
func (t *Token) Extra(key string) interface{} {
	if vals, ok := t.raw.(url.Values); ok {
		// TODO(jbd): Cast numeric values to int64 or float64.
		return vals.Get(key)
	}
	if raw, ok := t.raw.(map[string]interface{}); ok {
		return raw[key]
	}
	return nil
}

// expired reports whether the token is expired.
// t must be non-nil.
func (t *Token) expired() bool {
	if t.Expiry.IsZero() {
		return false
	}
	return t.Expiry.Add(-expiryDelta).Before(time.Now())
}

// Valid reports whether t is non-nil, has an AccessToken, and is not expired.
func (t *Token) Valid() bool {
	return t != nil && t.AccessToken != "" && !t.expired()
}

// tokenFromInternal maps an *internal.Token struct into
// a *Token struct.
func tokenFromInternal(t *internal.Token) *Token {
	if t == nil {
		return nil
	}
	return &Token{
		AccessToken:  t.AccessToken,
		TokenType:    t.TokenType,
		RefreshToken: t.RefreshToken,
		Expiry:       t.Expiry,
		raw:          t.Raw,
	}
}

// retrieveToken takes a *Config and uses that to retrieve an *internal.Token.
// This token is then mapped from *internal.Token into an *oauth2.Token which is returned along
// with an error..
func retrieveToken(ctx context.Context, c *Config, v url.Values) (*Token, error) {
	tk, err := internal.RetrieveToken(ctx, c.ClientID, c.ClientSecret, c.Endpoint.TokenURL, v)
	if err != nil {
		return nil, err
	}
	return tokenFromInternal(tk), nil
}
