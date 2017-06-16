// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package internal contains support packages for oauth2 package.
package internal

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"mime"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"golang.org/x/net/context"
)

// Token represents the crendentials used to authorize
// the requests to access protected resources on the OAuth 2.0
// provider's backend.
//
// This type is a mirror of oauth2.Token and exists to break
// an otherwise-circular dependency. Other internal packages
// should convert this Token into an oauth2.Token before use.
type Token struct {
	// AccessToken is the token that authorizes and authenticates
	// the requests.
	AccessToken string

	// TokenType is the type of token.
	// The Type method returns either this or "Bearer", the default.
	TokenType string

	// RefreshToken is a token that's used by the application
	// (as opposed to the user) to refresh the access token
	// if it expires.
	RefreshToken string

	// Expiry is the optional expiration time of the access token.
	//
	// If zero, TokenSource implementations will reuse the same
	// token forever and RefreshToken or equivalent
	// mechanisms for that TokenSource will not be used.
	Expiry time.Time

	// Raw optionally contains extra metadata from the server
	// when updating a token.
	Raw interface{}
}

// tokenJSON is the struct representing the HTTP response from OAuth2
// providers returning a token in JSON form.
type tokenJSON struct {
	AccessToken  string         `json:"access_token"`
	TokenType    string         `json:"token_type"`
	RefreshToken string         `json:"refresh_token"`
	ExpiresIn    expirationTime `json:"expires_in"` // at least PayPal returns string, while most return number
	Expires      expirationTime `json:"expires"`    // broken Facebook spelling of expires_in
}

func (e *tokenJSON) expiry() (t time.Time) {
	if v := e.ExpiresIn; v != 0 {
		return time.Now().Add(time.Duration(v) * time.Second)
	}
	if v := e.Expires; v != 0 {
		return time.Now().Add(time.Duration(v) * time.Second)
	}
	return
}

type expirationTime int32

func (e *expirationTime) UnmarshalJSON(b []byte) error {
	var n json.Number
	err := json.Unmarshal(b, &n)
	if err != nil {
		return err
	}
	i, err := n.Int64()
	if err != nil {
		return err
	}
	*e = expirationTime(i)
	return nil
}

var brokenAuthHeaderProviders = []string{
	"https://accounts.google.com/",
	"https://api.codeswholesale.com/oauth/token",
	"https://api.dropbox.com/",
	"https://api.dropboxapi.com/",
	"https://api.instagram.com/",
	"https://api.netatmo.net/",
	"https://api.odnoklassniki.ru/",
	"https://api.pushbullet.com/",
	"https://api.soundcloud.com/",
	"https://api.twitch.tv/",
	"https://app.box.com/",
	"https://connect.stripe.com/",
	"https://graph.facebook.com", // see https://github.com/golang/oauth2/issues/214
	"https://login.microsoftonline.com/",
	"https://login.salesforce.com/",
	"https://oauth.sandbox.trainingpeaks.com/",
	"https://oauth.trainingpeaks.com/",
	"https://oauth.vk.com/",
	"https://openapi.baidu.com/",
	"https://slack.com/",
	"https://test-sandbox.auth.corp.google.com",
	"https://test.salesforce.com/",
	"https://user.gini.net/",
	"https://www.douban.com/",
	"https://www.googleapis.com/",
	"https://www.linkedin.com/",
	"https://www.strava.com/oauth/",
	"https://www.wunderlist.com/oauth/",
	"https://api.patreon.com/",
	"https://sandbox.codeswholesale.com/oauth/token",
}

// brokenAuthHeaderDomains lists broken providers that issue dynamic endpoints.
var brokenAuthHeaderDomains = []string{
	".force.com",
	".okta.com",
	".oktapreview.com",
}

func RegisterBrokenAuthHeaderProvider(tokenURL string) {
	brokenAuthHeaderProviders = append(brokenAuthHeaderProviders, tokenURL)
}

// providerAuthHeaderWorks reports whether the OAuth2 server identified by the tokenURL
// implements the OAuth2 spec correctly
// See https://code.google.com/p/goauth2/issues/detail?id=31 for background.
// In summary:
// - Reddit only accepts client secret in the Authorization header
// - Dropbox accepts either it in URL param or Auth header, but not both.
// - Google only accepts URL param (not spec compliant?), not Auth header
// - Stripe only accepts client secret in Auth header with Bearer method, not Basic
func providerAuthHeaderWorks(tokenURL string) bool {
	for _, s := range brokenAuthHeaderProviders {
		if strings.HasPrefix(tokenURL, s) {
			// Some sites fail to implement the OAuth2 spec fully.
			return false
		}
	}

	if u, err := url.Parse(tokenURL); err == nil {
		for _, s := range brokenAuthHeaderDomains {
			if strings.HasSuffix(u.Host, s) {
				return false
			}
		}
	}

	// Assume the provider implements the spec properly
	// otherwise. We can add more exceptions as they're
	// discovered. We will _not_ be adding configurable hooks
	// to this package to let users select server bugs.
	return true
}

func RetrieveToken(ctx context.Context, clientID, clientSecret, tokenURL string, v url.Values) (*Token, error) {
	hc, err := ContextClient(ctx)
	if err != nil {
		return nil, err
	}
	bustedAuth := !providerAuthHeaderWorks(tokenURL)
	if bustedAuth {
		if clientID != "" {
			v.Set("client_id", clientID)
		}
		if clientSecret != "" {
			v.Set("client_secret", clientSecret)
		}
	}
	req, err := http.NewRequest("POST", tokenURL, strings.NewReader(v.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if !bustedAuth {
		req.SetBasicAuth(clientID, clientSecret)
	}
	r, err := hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer r.Body.Close()
	body, err := ioutil.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("oauth2: cannot fetch token: %v", err)
	}
	if code := r.StatusCode; code < 200 || code > 299 {
		return nil, fmt.Errorf("oauth2: cannot fetch token: %v\nResponse: %s", r.Status, body)
	}

	var token *Token
	content, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	switch content {
	case "application/x-www-form-urlencoded", "text/plain":
		vals, err := url.ParseQuery(string(body))
		if err != nil {
			return nil, err
		}
		token = &Token{
			AccessToken:  vals.Get("access_token"),
			TokenType:    vals.Get("token_type"),
			RefreshToken: vals.Get("refresh_token"),
			Raw:          vals,
		}
		e := vals.Get("expires_in")
		if e == "" {
			// TODO(jbd): Facebook's OAuth2 implementation is broken and
			// returns expires_in field in expires. Remove the fallback to expires,
			// when Facebook fixes their implementation.
			e = vals.Get("expires")
		}
		expires, _ := strconv.Atoi(e)
		if expires != 0 {
			token.Expiry = time.Now().Add(time.Duration(expires) * time.Second)
		}
	default:
		var tj tokenJSON
		if err = json.Unmarshal(body, &tj); err != nil {
			return nil, err
		}
		token = &Token{
			AccessToken:  tj.AccessToken,
			TokenType:    tj.TokenType,
			RefreshToken: tj.RefreshToken,
			Expiry:       tj.expiry(),
			Raw:          make(map[string]interface{}),
		}
		json.Unmarshal(body, &token.Raw) // no error checks for optional fields
	}
	// Don't overwrite `RefreshToken` with an empty value
	// if this was a token refreshing request.
	if token.RefreshToken == "" {
		token.RefreshToken = v.Get("refresh_token")
	}
	return token, nil
}
