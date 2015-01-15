// Copyright 2014 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine,!appenginevm

package google

import (
	"time"

	"appengine"

	"golang.org/x/oauth2"
)

// AppEngineTokenSource returns a token source that fetches tokens
// issued to the current App Engine application's service account.
// If you are implementing a 3-legged OAuth 2.0 flow on App Engine
// that involves user accounts, see oauth2.Config instead.
//
// You are required to provide a valid appengine.Context as context.
func AppEngineTokenSource(ctx appengine.Context, scope ...string) oauth2.TokenSource {
	return &appEngineTokenSource{
		ctx:         ctx,
		scopes:      scope,
		fetcherFunc: aeFetcherFunc,
	}
}

var aeFetcherFunc = func(ctx oauth2.Context, scope ...string) (string, time.Time, error) {
	c, ok := ctx.(appengine.Context)
	if !ok {
		return "", time.Time{}, errInvalidContext
	}
	return appengine.AccessToken(c, scope...)
}
