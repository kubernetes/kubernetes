// Copyright 2013 The goauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package serviceaccount

import (
	"time"

	"appengine"
	"appengine/memcache"

	"code.google.com/p/goauth2/oauth"
)

// cache implementss TokenCache using memcache to store AccessToken
// for the application service account.
type cache struct {
	Context appengine.Context
	Key     string
}

func (m cache) Token() (*oauth.Token, error) {
	item, err := memcache.Get(m.Context, m.Key)
	if err != nil {
		return nil, err
	}
	return &oauth.Token{
		AccessToken: string(item.Value),
		Expiry:      time.Now().Add(item.Expiration),
	}, nil
}

func (m cache) PutToken(tok *oauth.Token) error {
	return memcache.Set(m.Context, &memcache.Item{
		Key:        m.Key,
		Value:      []byte(tok.AccessToken),
		Expiration: tok.Expiry.Sub(time.Now()),
	})
}
