// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package jira provides claims and JWT signing for OAuth2 to access JIRA/Confluence.
package jira

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"time"

	"golang.org/x/oauth2"
)

// ClaimSet contains information about the JWT signature according
// to Atlassian's documentation
// https://developer.atlassian.com/cloud/jira/software/oauth-2-jwt-bearer-token-authorization-grant-type/
type ClaimSet struct {
	Issuer       string `json:"iss"`
	Subject      string `json:"sub"`
	InstalledURL string `json:"tnt"` // URL of installed app
	AuthURL      string `json:"aud"` // URL of auth server
	ExpiresIn    int64  `json:"exp"` // Must be no later that 60 seconds in the future
	IssuedAt     int64  `json:"iat"`
}

var (
	defaultGrantType = "urn:ietf:params:oauth:grant-type:jwt-bearer"
	defaultHeader    = map[string]string{
		"typ": "JWT",
		"alg": "HS256",
	}
)

// Config is the configuration for using JWT to fetch tokens,
// commonly known as "two-legged OAuth 2.0".
type Config struct {
	// BaseURL for your app
	BaseURL string

	// Subject is the userkey as defined by Atlassian
	// Different than username (ex: /rest/api/2/user?username=alex)
	Subject string

	oauth2.Config
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
	exp := time.Duration(59) * time.Second
	claimSet := &ClaimSet{
		Issuer:       fmt.Sprintf("urn:atlassian:connect:clientid:%s", js.conf.ClientID),
		Subject:      fmt.Sprintf("urn:atlassian:connect:useraccountid:%s", js.conf.Subject),
		InstalledURL: js.conf.BaseURL,
		AuthURL:      js.conf.Endpoint.AuthURL,
		IssuedAt:     time.Now().Unix(),
		ExpiresIn:    time.Now().Add(exp).Unix(),
	}

	v := url.Values{}
	v.Set("grant_type", defaultGrantType)

	// Add scopes if they exist;  If not, it defaults to app scopes
	if scopes := js.conf.Scopes; scopes != nil {
		upperScopes := make([]string, len(scopes))
		for i, k := range scopes {
			upperScopes[i] = strings.ToUpper(k)
		}
		v.Set("scope", strings.Join(upperScopes, "+"))
	}

	// Sign claims for assertion
	assertion, err := sign(js.conf.ClientSecret, claimSet)
	if err != nil {
		return nil, err
	}
	v.Set("assertion", string(assertion))

	// Fetch access token from auth server
	hc := oauth2.NewClient(js.ctx, nil)
	resp, err := hc.PostForm(js.conf.Endpoint.TokenURL, v)
	if err != nil {
		return nil, fmt.Errorf("oauth2: cannot fetch token: %v", err)
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("oauth2: cannot fetch token: %v", err)
	}
	if c := resp.StatusCode; c < 200 || c > 299 {
		return nil, fmt.Errorf("oauth2: cannot fetch token: %v\nResponse: %s", resp.Status, body)
	}

	// tokenRes is the JSON response body.
	var tokenRes struct {
		AccessToken string `json:"access_token"`
		TokenType   string `json:"token_type"`
		ExpiresIn   int64  `json:"expires_in"` // relative seconds from now
	}
	if err := json.Unmarshal(body, &tokenRes); err != nil {
		return nil, fmt.Errorf("oauth2: cannot fetch token: %v", err)
	}
	token := &oauth2.Token{
		AccessToken: tokenRes.AccessToken,
		TokenType:   tokenRes.TokenType,
	}

	if secs := tokenRes.ExpiresIn; secs > 0 {
		token.Expiry = time.Now().Add(time.Duration(secs) * time.Second)
	}
	return token, nil
}

// Sign the claim set with the shared secret
// Result to be sent as assertion
func sign(key string, claims *ClaimSet) (string, error) {
	b, err := json.Marshal(defaultHeader)
	if err != nil {
		return "", err
	}
	header := base64.RawURLEncoding.EncodeToString(b)

	jsonClaims, err := json.Marshal(claims)
	if err != nil {
		return "", err
	}
	encodedClaims := strings.TrimRight(base64.URLEncoding.EncodeToString(jsonClaims), "=")

	ss := fmt.Sprintf("%s.%s", header, encodedClaims)

	mac := hmac.New(sha256.New, []byte(key))
	mac.Write([]byte(ss))
	signature := mac.Sum(nil)

	return fmt.Sprintf("%s.%s", ss, base64.RawURLEncoding.EncodeToString(signature)), nil
}
