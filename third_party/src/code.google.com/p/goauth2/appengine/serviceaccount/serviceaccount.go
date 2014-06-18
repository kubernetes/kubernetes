// Copyright 2013 The goauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

// The serviceaccount package provides support for making
// OAuth2-authorized HTTP requests from App Engine using service
// accounts.
//
// See: https://developers.google.com/appengine/docs/go/reference#AccessToken
//
// Example usage:
//
//	c := appengine.NewContext()
//	client, err := serviceaccount.NewClient(c, "https://www.googleapis.com/auth/compute", "https://www.googleapis.com/auth/bigquery")
//	if err != nil {
//		c.Errorf("failed to create service account client: %q", err)
//		return err
//	}
//	client.Post("https://www.googleapis.com/compute/...", ...)
//	client.Post("https://www.googleapis.com/bigquery/...", ...)
//
package serviceaccount

import (
	"net/http"
	"strings"

	"appengine"
	"appengine/urlfetch"

	"code.google.com/p/goauth2/oauth"
)

// NewClient returns an *http.Client authorized for the
// given scopes with the service account owned by the application.
// Tokens are cached in memcache until they expire.
func NewClient(c appengine.Context, scopes ...string) (*http.Client, error) {
	t := &transport{
		Context: c,
		Scopes:  scopes,
		Transport: &urlfetch.Transport{
			Context:                       c,
			Deadline:                      0,
			AllowInvalidServerCertificate: false,
		},
		TokenCache: &cache{
			Context: c,
			Key:     "goauth2_serviceaccount_" + strings.Join(scopes, "_"),
		},
	}
	// Get the initial access token.
	if err := t.FetchToken(); err != nil {
		return nil, err
	}
	return &http.Client{
		Transport: t,
	}, nil
}

// transport is an oauth.Transport with a custom Refresh and RoundTrip implementation.
type transport struct {
	*oauth.Token
	Context    appengine.Context
	Scopes     []string
	Transport  http.RoundTripper
	TokenCache oauth.Cache
}

func (t *transport) Refresh() error {
	// Get a new access token for the application service account.
	tok, expiry, err := appengine.AccessToken(t.Context, t.Scopes...)
	if err != nil {
		return err
	}
	t.Token = &oauth.Token{
		AccessToken: tok,
		Expiry:      expiry,
	}
	if t.TokenCache != nil {
		// Cache the token and ignore error (as we can always get a new one).
		t.TokenCache.PutToken(t.Token)
	}
	return nil
}

// Fetch token from cache or generate a new one if cache miss or expired.
func (t *transport) FetchToken() error {
	// Try to get the Token from the cache if enabled.
	if t.Token == nil && t.TokenCache != nil {
		// Ignore cache error as we can always get a new token with Refresh.
		t.Token, _ = t.TokenCache.Token()
	}

	// Get a new token using Refresh in case of a cache miss of if it has expired.
	if t.Token == nil || t.Expired() {
		if err := t.Refresh(); err != nil {
			return err
		}
	}
	return nil
}

// cloneRequest returns a clone of the provided *http.Request.
// The clone is a shallow copy of the struct and its Header map.
func cloneRequest(r *http.Request) *http.Request {
	// shallow copy of the struct
	r2 := new(http.Request)
	*r2 = *r
	// deep copy of the Header
	r2.Header = make(http.Header)
	for k, s := range r.Header {
		r2.Header[k] = s
	}
	return r2
}

// RoundTrip issues an authorized HTTP request and returns its response.
func (t *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	if err := t.FetchToken(); err != nil {
		return nil, err
	}

	// To set the Authorization header, we must make a copy of the Request
	// so that we don't modify the Request we were given.
	// This is required by the specification of http.RoundTripper.
	newReq := cloneRequest(req)
	newReq.Header.Set("Authorization", "Bearer "+t.AccessToken)

	// Make the HTTP request.
	return t.Transport.RoundTrip(newReq)
}
