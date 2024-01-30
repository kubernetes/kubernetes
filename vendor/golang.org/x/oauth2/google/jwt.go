// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google

import (
	"crypto/rsa"
	"fmt"
	"strings"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/internal"
	"golang.org/x/oauth2/jws"
)

// JWTAccessTokenSourceFromJSON uses a Google Developers service account JSON
// key file to read the credentials that authorize and authenticate the
// requests, and returns a TokenSource that does not use any OAuth2 flow but
// instead creates a JWT and sends that as the access token.
// The audience is typically a URL that specifies the scope of the credentials.
//
// Note that this is not a standard OAuth flow, but rather an
// optimization supported by a few Google services.
// Unless you know otherwise, you should use JWTConfigFromJSON instead.
func JWTAccessTokenSourceFromJSON(jsonKey []byte, audience string) (oauth2.TokenSource, error) {
	return newJWTSource(jsonKey, audience, nil)
}

// JWTAccessTokenSourceWithScope uses a Google Developers service account JSON
// key file to read the credentials that authorize and authenticate the
// requests, and returns a TokenSource that does not use any OAuth2 flow but
// instead creates a JWT and sends that as the access token.
// The scope is typically a list of URLs that specifies the scope of the
// credentials.
//
// Note that this is not a standard OAuth flow, but rather an
// optimization supported by a few Google services.
// Unless you know otherwise, you should use JWTConfigFromJSON instead.
func JWTAccessTokenSourceWithScope(jsonKey []byte, scope ...string) (oauth2.TokenSource, error) {
	return newJWTSource(jsonKey, "", scope)
}

func newJWTSource(jsonKey []byte, audience string, scopes []string) (oauth2.TokenSource, error) {
	if len(scopes) == 0 && audience == "" {
		return nil, fmt.Errorf("google: missing scope/audience for JWT access token")
	}

	cfg, err := JWTConfigFromJSON(jsonKey)
	if err != nil {
		return nil, fmt.Errorf("google: could not parse JSON key: %v", err)
	}
	pk, err := internal.ParseKey(cfg.PrivateKey)
	if err != nil {
		return nil, fmt.Errorf("google: could not parse key: %v", err)
	}
	ts := &jwtAccessTokenSource{
		email:    cfg.Email,
		audience: audience,
		scopes:   scopes,
		pk:       pk,
		pkID:     cfg.PrivateKeyID,
	}
	tok, err := ts.Token()
	if err != nil {
		return nil, err
	}
	rts := newErrWrappingTokenSource(oauth2.ReuseTokenSource(tok, ts))
	return rts, nil
}

type jwtAccessTokenSource struct {
	email, audience string
	scopes          []string
	pk              *rsa.PrivateKey
	pkID            string
}

func (ts *jwtAccessTokenSource) Token() (*oauth2.Token, error) {
	iat := time.Now()
	exp := iat.Add(time.Hour)
	scope := strings.Join(ts.scopes, " ")
	cs := &jws.ClaimSet{
		Iss:   ts.email,
		Sub:   ts.email,
		Aud:   ts.audience,
		Scope: scope,
		Iat:   iat.Unix(),
		Exp:   exp.Unix(),
	}
	hdr := &jws.Header{
		Algorithm: "RS256",
		Typ:       "JWT",
		KeyID:     string(ts.pkID),
	}
	msg, err := jws.Encode(hdr, cs, ts.pk)
	if err != nil {
		return nil, fmt.Errorf("google: could not encode JWT: %v", err)
	}
	return &oauth2.Token{AccessToken: msg, TokenType: "Bearer", Expiry: exp}, nil
}
