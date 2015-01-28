// Copyright 2014 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google

import (
	"errors"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/oauth2"
)

var (
	aeTokensMu sync.Mutex // guards aeTokens and appEngineTokenSource.key

	// aeTokens helps the fetched tokens to be reused until their expiration.
	aeTokens = make(map[string]*tokenLock) // key is '\0'-separated scopes
)

var errInvalidContext = errors.New("oauth2: a valid appengine.Context is required")

type tokenLock struct {
	mu sync.Mutex // guards t; held while updating t
	t  *oauth2.Token
}

type appEngineTokenSource struct {
	ctx oauth2.Context

	// fetcherFunc makes the actual RPC to fetch a new access
	// token with an expiry time.  Provider of this function is
	// responsible to assert that the given context is valid.
	fetcherFunc func(ctx oauth2.Context, scope ...string) (accessToken string, expiry time.Time, err error)

	// scopes and key are guarded by the package-level mutex aeTokensMu
	scopes []string
	key    string
}

func (ts *appEngineTokenSource) Token() (*oauth2.Token, error) {
	aeTokensMu.Lock()
	if ts.key == "" {
		sort.Sort(sort.StringSlice(ts.scopes))
		ts.key = strings.Join(ts.scopes, string(0))
	}
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
	access, exp, err := ts.fetcherFunc(ts.ctx, ts.scopes...)
	if err != nil {
		return nil, err
	}
	tok.t = &oauth2.Token{
		AccessToken: access,
		Expiry:      exp,
	}
	return tok.t, nil
}
