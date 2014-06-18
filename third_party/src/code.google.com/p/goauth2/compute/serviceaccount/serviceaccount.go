// Copyright 2013 The goauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package serviceaccount provides support for making OAuth2-authorized
// HTTP requests from Google Compute Engine instances using service accounts.
//
// See: https://developers.google.com/compute/docs/authentication
//
// Example usage:
//
//	client, err := serviceaccount.NewClient(&serviceaccount.Options{})
//	if err != nil {
//		c.Errorf("failed to create service account client: %q", err)
//		return err
//	}
//	client.Post("https://www.googleapis.com/compute/...", ...)
//	client.Post("https://www.googleapis.com/bigquery/...", ...)
//
package serviceaccount

import (
	"encoding/json"
	"net/http"
	"net/url"
	"path"
	"sync"
	"time"

	"code.google.com/p/goauth2/oauth"
)

const (
	metadataServer     = "metadata"
	serviceAccountPath = "/computeMetadata/v1/instance/service-accounts"
)

// Options configures a service account Client.
type Options struct {
	// Underlying transport of service account Client.
	// If nil, http.DefaultTransport is used.
	Transport http.RoundTripper

	// Service account name.
	// If empty, "default" is used.
	Account string
}

// NewClient returns an *http.Client authorized with the service account
// configured in the Google Compute Engine instance.
func NewClient(opt *Options) (*http.Client, error) {
	tr := http.DefaultTransport
	account := "default"
	if opt != nil {
		if opt.Transport != nil {
			tr = opt.Transport
		}
		if opt.Account != "" {
			account = opt.Account
		}
	}
	t := &transport{
		Transport: tr,
		Account:   account,
	}
	// Get the initial access token.
	if _, err := fetchToken(t); err != nil {
		return nil, err
	}
	return &http.Client{
		Transport: t,
	}, nil
}

type tokenData struct {
	AccessToken string  `json:"access_token"`
	ExpiresIn   float64 `json:"expires_in"`
	TokenType   string  `json:"token_type"`
}

// transport is an oauth.Transport with a custom Refresh and RoundTrip implementation.
type transport struct {
	Transport http.RoundTripper
	Account   string

	mu sync.Mutex
	*oauth.Token
}

// Refresh renews the transport's AccessToken.
// t.mu sould be held when this is called.
func (t *transport) refresh() error {
	// https://developers.google.com/compute/docs/metadata
	// v1 requires "X-Google-Metadata-Request: True" header.
	tokenURL := &url.URL{
		Scheme: "http",
		Host:   metadataServer,
		Path:   path.Join(serviceAccountPath, t.Account, "token"),
	}
	req, err := http.NewRequest("GET", tokenURL.String(), nil)
	if err != nil {
		return err
	}
	req.Header.Add("X-Google-Metadata-Request", "True")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	d := json.NewDecoder(resp.Body)
	var token tokenData
	err = d.Decode(&token)
	if err != nil {
		return err
	}
	t.Token = &oauth.Token{
		AccessToken: token.AccessToken,
		Expiry:      time.Now().Add(time.Duration(token.ExpiresIn) * time.Second),
	}
	return nil
}

// Refresh renews the transport's AccessToken.
func (t *transport) Refresh() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.refresh()
}

// Fetch token from cache or generate a new one if cache miss or expired.
func fetchToken(t *transport) (*oauth.Token, error) {
	// Get a new token using Refresh in case of a cache miss of if it has expired.
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.Token == nil || t.Expired() {
		if err := t.refresh(); err != nil {
			return nil, err
		}
	}
	return t.Token, nil
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
	token, err := fetchToken(t)
	if err != nil {
		return nil, err
	}

	// To set the Authorization header, we must make a copy of the Request
	// so that we don't modify the Request we were given.
	// This is required by the specification of http.RoundTripper.
	newReq := cloneRequest(req)
	newReq.Header.Set("Authorization", "Bearer "+token.AccessToken)

	// Make the HTTP request.
	return t.Transport.RoundTrip(newReq)
}
