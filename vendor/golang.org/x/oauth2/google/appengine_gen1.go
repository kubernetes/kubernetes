// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build appengine

// This file applies to App Engine first generation runtimes (<= Go 1.9).

package google

import (
	"context"
	"sort"
	"strings"
	"sync"

	"golang.org/x/oauth2"
	"google.golang.org/appengine"
)

func init() {
	appengineTokenFunc = appengine.AccessToken
	appengineAppIDFunc = appengine.AppID
}

// See comment on AppEngineTokenSource in appengine.go.
func appEngineTokenSource(ctx context.Context, scope ...string) oauth2.TokenSource {
	scopes := append([]string{}, scope...)
	sort.Strings(scopes)
	return &gaeTokenSource{
		ctx:    ctx,
		scopes: scopes,
		key:    strings.Join(scopes, " "),
	}
}

// aeTokens helps the fetched tokens to be reused until their expiration.
var (
	aeTokensMu sync.Mutex
	aeTokens   = make(map[string]*tokenLock) // key is space-separated scopes
)

type tokenLock struct {
	mu sync.Mutex // guards t; held while fetching or updating t
	t  *oauth2.Token
}

type gaeTokenSource struct {
	ctx    context.Context
	scopes []string
	key    string // to aeTokens map; space-separated scopes
}

func (ts *gaeTokenSource) Token() (*oauth2.Token, error) {
	aeTokensMu.Lock()
	tok, ok := aeTokens[ts.key]
	if !ok {
		tok = &tokenLock{}
		aeTokens[ts.key] = tok
	}
	aeTokensMu.Unlock()

	tok.mu.Lock()
	defer tok.mu.Unlock()
	if tok.t.Valid() {
		return tok.t, nil
	}
	access, exp, err := appengineTokenFunc(ts.ctx, ts.scopes...)
	if err != nil {
		return nil, err
	}
	tok.t = &oauth2.Token{
		AccessToken: access,
		Expiry:      exp,
	}
	return tok.t, nil
}
