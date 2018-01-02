/*
Copyright 2015 The Camlistore Authors

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

// Package oauthutil contains OAuth 2 related utilities.
package oauthutil // import "go4.org/oauthutil"

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"go4.org/wkfs"
	"golang.org/x/oauth2"
)

// TitleBarRedirectURL is the OAuth2 redirect URL to use when the authorization
// code should be returned in the title bar of the browser, with the page text
// prompting the user to copy the code and paste it in the application.
const TitleBarRedirectURL = "urn:ietf:wg:oauth:2.0:oob"

// ErrNoAuthCode is returned when Token() has not found any valid cached token
// and TokenSource does not have an AuthCode for getting a new token.
var ErrNoAuthCode = errors.New("oauthutil: unspecified TokenSource.AuthCode")

// TokenSource is an implementation of oauth2.TokenSource. It uses CacheFile to store and
// reuse the the acquired token, and AuthCode to provide the authorization code that will be
// exchanged for a token otherwise.
type TokenSource struct {
	Config *oauth2.Config

	// CacheFile is where the token will be stored JSON-encoded. Any call to Token
	// first tries to read a valid token from CacheFile.
	CacheFile string

	// AuthCode provides the authorization code that Token will exchange for a token.
	// It usually is a way to prompt the user for the code. If CacheFile does not provide
	// a token and AuthCode is nil, Token returns ErrNoAuthCode.
	AuthCode func() string
}

var errExpiredToken = errors.New("expired token")

// cachedToken returns the token saved in cacheFile. It specifically returns
// errTokenExpired if the token is expired.
func cachedToken(cacheFile string) (*oauth2.Token, error) {
	tok := new(oauth2.Token)
	tokenData, err := wkfs.ReadFile(cacheFile)
	if err != nil {
		return nil, err
	}
	if err = json.Unmarshal(tokenData, tok); err != nil {
		return nil, err
	}
	if !tok.Valid() {
		if tok != nil && time.Now().After(tok.Expiry) {
			return nil, errExpiredToken
		}
		return nil, errors.New("invalid token")
	}
	return tok, nil
}

// Token first tries to find a valid token in CacheFile, and otherwise uses
// Config and AuthCode to fetch a new token. This new token is saved in CacheFile
// (if not blank). If CacheFile did not provide a token and AuthCode is nil,
// ErrNoAuthCode is returned.
func (src TokenSource) Token() (*oauth2.Token, error) {
	var tok *oauth2.Token
	var err error
	if src.CacheFile != "" {
		tok, err = cachedToken(src.CacheFile)
		if err == nil {
			return tok, nil
		}
		if err != errExpiredToken {
			fmt.Printf("Error getting token from %s: %v\n", src.CacheFile, err)
		}
	}
	if src.AuthCode == nil {
		return nil, ErrNoAuthCode
	}
	tok, err = src.Config.Exchange(oauth2.NoContext, src.AuthCode())
	if err != nil {
		return nil, fmt.Errorf("could not exchange auth code for a token: %v", err)
	}
	if src.CacheFile == "" {
		return tok, nil
	}
	tokenData, err := json.Marshal(&tok)
	if err != nil {
		return nil, fmt.Errorf("could not encode token as json: %v", err)
	}
	if err := wkfs.WriteFile(src.CacheFile, tokenData, 0600); err != nil {
		return nil, fmt.Errorf("could not cache token in %v: %v", src.CacheFile, err)
	}
	return tok, nil
}

// NewRefreshTokenSource returns a token source that obtains its initial token
// based on the provided config and the refresh token.
func NewRefreshTokenSource(config *oauth2.Config, refreshToken string) oauth2.TokenSource {
	var noInitialToken *oauth2.Token = nil
	return oauth2.ReuseTokenSource(noInitialToken, config.TokenSource(
		oauth2.NoContext, // TODO: maybe accept a context later.
		&oauth2.Token{RefreshToken: refreshToken},
	))
}
