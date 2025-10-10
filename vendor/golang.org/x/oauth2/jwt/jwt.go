// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package jwt implements the OAuth 2.0 JSON Web Token flow, commonly
// known as "two-legged OAuth 2.0".
//
// See: https://tools.ietf.org/html/draft-ietf-oauth-jwt-bearer-12
package jwt

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/internal"
	"golang.org/x/oauth2/jws"
)

var (
	defaultGrantType = "urn:ietf:params:oauth:grant-type:jwt-bearer"
	defaultHeader    = &jws.Header{Algorithm: "RS256", Typ: "JWT"}
)

// Config is the configuration for using JWT to fetch tokens,
// commonly known as "two-legged OAuth 2.0".
type Config struct {
	// Email is the OAuth client identifier used when communicating with
	// the configured OAuth provider.
	Email string

	// PrivateKey contains the contents of an RSA private key or the
	// contents of a PEM file that contains a private key. The provided
	// private key is used to sign JWT payloads.
	// PEM containers with a passphrase are not supported.
	// Use the following command to convert a PKCS 12 file into a PEM.
	//
	//    $ openssl pkcs12 -in key.p12 -out key.pem -nodes
	//
	PrivateKey []byte

	// PrivateKeyID contains an optional hint indicating which key is being
	// used.
	PrivateKeyID string

	// Subject is the optional user to impersonate.
	Subject string

	// Scopes optionally specifies a list of requested permission scopes.
	Scopes []string

	// TokenURL is the endpoint required to complete the 2-legged JWT flow.
	TokenURL string

	// Expires optionally specifies how long the token is valid for.
	Expires time.Duration

	// Audience optionally specifies the intended audience of the
	// request.  If empty, the value of TokenURL is used as the
	// intended audience.
	Audience string

	// PrivateClaims optionally specifies custom private claims in the JWT.
	// See http://tools.ietf.org/html/draft-jones-json-web-token-10#section-4.3
	PrivateClaims map[string]any

	// UseIDToken optionally specifies whether ID token should be used instead
	// of access token when the server returns both.
	UseIDToken bool
}

// TokenSource returns a JWT TokenSource using the configuration
// in c and the HTTP client from the provided context.
func (c *Config) TokenSource(ctx context.Context) oauth2.TokenSource {
	return oauth2.ReuseTokenSource(nil, jwtSource{ctx, c})
}

// Client returns an HTTP client wrapping the context's
// HTTP transport and adding Authorization headers with tokens
// obtained from c.
//
// The returned client and its Transport should not be modified.
func (c *Config) Client(ctx context.Context) *http.Client {
	return oauth2.NewClient(ctx, c.TokenSource(ctx))
}

// jwtSource is a source that always does a signed JWT request for a token.
// It should typically be wrapped with a reuseTokenSource.
type jwtSource struct {
	ctx  context.Context
	conf *Config
}

func (js jwtSource) Token() (*oauth2.Token, error) {
	pk, err := internal.ParseKey(js.conf.PrivateKey)
	if err != nil {
		return nil, err
	}
	hc := oauth2.NewClient(js.ctx, nil)
	claimSet := &jws.ClaimSet{
		Iss:           js.conf.Email,
		Scope:         strings.Join(js.conf.Scopes, " "),
		Aud:           js.conf.TokenURL,
		PrivateClaims: js.conf.PrivateClaims,
	}
	if subject := js.conf.Subject; subject != "" {
		claimSet.Sub = subject
		// prn is the old name of sub. Keep setting it
		// to be compatible with legacy OAuth 2.0 providers.
		claimSet.Prn = subject
	}
	if t := js.conf.Expires; t > 0 {
		claimSet.Exp = time.Now().Add(t).Unix()
	}
	if aud := js.conf.Audience; aud != "" {
		claimSet.Aud = aud
	}
	h := *defaultHeader
	h.KeyID = js.conf.PrivateKeyID
	payload, err := jws.Encode(&h, claimSet, pk)
	if err != nil {
		return nil, err
	}
	v := url.Values{}
	v.Set("grant_type", defaultGrantType)
	v.Set("assertion", payload)
	resp, err := hc.PostForm(js.conf.TokenURL, v)
	if err != nil {
		return nil, fmt.Errorf("oauth2: cannot fetch token: %v", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("oauth2: cannot fetch token: %v", err)
	}
	if c := resp.StatusCode; c < 200 || c > 299 {
		return nil, &oauth2.RetrieveError{
			Response: resp,
			Body:     body,
		}
	}
	// tokenRes is the JSON response body.
	var tokenRes struct {
		oauth2.Token
		IDToken string `json:"id_token"`
	}
	if err := json.Unmarshal(body, &tokenRes); err != nil {
		return nil, fmt.Errorf("oauth2: cannot fetch token: %v", err)
	}
	token := &oauth2.Token{
		AccessToken: tokenRes.AccessToken,
		TokenType:   tokenRes.TokenType,
	}
	raw := make(map[string]any)
	json.Unmarshal(body, &raw) // no error checks for optional fields
	token = token.WithExtra(raw)

	if secs := tokenRes.ExpiresIn; secs > 0 {
		token.Expiry = time.Now().Add(time.Duration(secs) * time.Second)
	}
	if v := tokenRes.IDToken; v != "" {
		// decode returned id token to get expiry
		claimSet, err := jws.Decode(v)
		if err != nil {
			return nil, fmt.Errorf("oauth2: error decoding JWT token: %v", err)
		}
		token.Expiry = time.Unix(claimSet.Exp, 0)
	}
	if js.conf.UseIDToken {
		if tokenRes.IDToken == "" {
			return nil, fmt.Errorf("oauth2: response doesn't have JWT token")
		}
		token.AccessToken = tokenRes.IDToken
	}
	return token, nil
}
