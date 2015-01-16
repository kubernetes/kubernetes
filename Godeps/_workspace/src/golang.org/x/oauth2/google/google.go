// Copyright 2014 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package google provides support for making
// OAuth2 authorized and authenticated HTTP requests
// to Google APIs. It supports Web server, client-side,
// service accounts, Google Compute Engine service accounts,
// and Google App Engine service accounts authorization
// and authentications flows:
//
// For more information, please read
// https://developers.google.com/accounts/docs/OAuth2.
package google

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/jwt"
	"google.golang.org/cloud/compute/metadata"
)

// TODO(bradfitz,jbd): import "google.golang.org/cloud/compute/metadata" instead of
// the metaClient and metadata.google.internal stuff below.

// Endpoint is Google's OAuth 2.0 endpoint.
var Endpoint = oauth2.Endpoint{
	AuthURL:  "https://accounts.google.com/o/oauth2/auth",
	TokenURL: "https://accounts.google.com/o/oauth2/token",
}

// JWTTokenURL is Google's OAuth 2.0 token URL to use with the JWT flow.
const JWTTokenURL = "https://accounts.google.com/o/oauth2/token"

// JWTConfigFromJSON uses a Google Developers service account JSON key file to read
// the credentials that authorize and authenticate the requests.
// Create a service account on "Credentials" page under "APIs & Auth" for your
// project at https://console.developers.google.com to download a JSON key file.
func JWTConfigFromJSON(jsonKey []byte, scope ...string) (*jwt.Config, error) {
	var key struct {
		Email      string `json:"client_email"`
		PrivateKey string `json:"private_key"`
	}
	if err := json.Unmarshal(jsonKey, &key); err != nil {
		return nil, err
	}
	return &jwt.Config{
		Email:      key.Email,
		PrivateKey: []byte(key.PrivateKey),
		Scopes:     scope,
		TokenURL:   JWTTokenURL,
	}, nil
}

// ComputeTokenSource returns a token source that fetches access tokens
// from Google Compute Engine (GCE)'s metadata server. It's only valid to use
// this token source if your program is running on a GCE instance.
// If no account is specified, "default" is used.
// Further information about retrieving access tokens from the GCE metadata
// server can be found at https://cloud.google.com/compute/docs/authentication.
func ComputeTokenSource(account string) oauth2.TokenSource {
	return oauth2.ReuseTokenSource(nil, computeSource{account: account})
}

type computeSource struct {
	account string
}

func (cs computeSource) Token() (*oauth2.Token, error) {
	if !metadata.OnGCE() {
		return nil, errors.New("oauth2/google: can't get a token from the metadata service; not running on GCE")
	}
	acct := cs.account
	if acct == "" {
		acct = "default"
	}
	tokenJSON, err := metadata.Get("instance/service-accounts/" + acct + "/token")
	if err != nil {
		return nil, err
	}
	var res struct {
		AccessToken  string `json:"access_token"`
		ExpiresInSec int    `json:"expires_in"`
		TokenType    string `json:"token_type"`
	}
	err = json.NewDecoder(strings.NewReader(tokenJSON)).Decode(&res)
	if err != nil {
		return nil, fmt.Errorf("oauth2/google: invalid token JSON from metadata: %v", err)
	}
	if res.ExpiresInSec == 0 || res.AccessToken == "" {
		return nil, fmt.Errorf("oauth2/google: incomplete token received from metadata")
	}
	return &oauth2.Token{
		AccessToken: res.AccessToken,
		TokenType:   res.TokenType,
		Expiry:      time.Now().Add(time.Duration(res.ExpiresInSec) * time.Second),
	}, nil
}
