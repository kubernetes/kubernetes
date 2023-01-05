// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package authhandler implements a TokenSource to support
// "three-legged OAuth 2.0" via a custom AuthorizationHandler.
package authhandler

import (
	"context"
	"errors"

	"golang.org/x/oauth2"
)

// AuthorizationHandler is a 3-legged-OAuth helper that prompts
// the user for OAuth consent at the specified auth code URL
// and returns an auth code and state upon approval.
type AuthorizationHandler func(authCodeURL string) (code string, state string, err error)

// TokenSource returns an oauth2.TokenSource that fetches access tokens
// using 3-legged-OAuth flow.
//
// The provided context.Context is used for oauth2 Exchange operation.
//
// The provided oauth2.Config should be a full configuration containing AuthURL,
// TokenURL, and Scope.
//
// An environment-specific AuthorizationHandler is used to obtain user consent.
//
// Per the OAuth protocol, a unique "state" string should be specified here.
// This token source will verify that the "state" is identical in the request
// and response before exchanging the auth code for OAuth token to prevent CSRF
// attacks.
func TokenSource(ctx context.Context, config *oauth2.Config, state string, authHandler AuthorizationHandler) oauth2.TokenSource {
	return oauth2.ReuseTokenSource(nil, authHandlerSource{config: config, ctx: ctx, authHandler: authHandler, state: state})
}

type authHandlerSource struct {
	ctx         context.Context
	config      *oauth2.Config
	authHandler AuthorizationHandler
	state       string
}

func (source authHandlerSource) Token() (*oauth2.Token, error) {
	url := source.config.AuthCodeURL(source.state)
	code, state, err := source.authHandler(url)
	if err != nil {
		return nil, err
	}
	if state != source.state {
		return nil, errors.New("state mismatch in 3-legged-OAuth flow")
	}
	return source.config.Exchange(source.ctx, code)
}
