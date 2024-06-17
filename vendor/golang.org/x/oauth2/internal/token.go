// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"mime"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Token represents the credentials used to authorize
// the requests to access protected resources on the OAuth 2.0
// provider's backend.
//
// This type is a mirror of oauth2.Token and exists to break
// an otherwise-circular dependency. Other internal packages
// should convert this Token into an oauth2.Token before use.
type Token struct {
	// AccessToken is the token that authorizes and authenticates
	// the requests.
	AccessToken string

	// TokenType is the type of token.
	// The Type method returns either this or "Bearer", the default.
	TokenType string

	// RefreshToken is a token that's used by the application
	// (as opposed to the user) to refresh the access token
	// if it expires.
	RefreshToken string

	// Expiry is the optional expiration time of the access token.
	//
	// If zero, TokenSource implementations will reuse the same
	// token forever and RefreshToken or equivalent
	// mechanisms for that TokenSource will not be used.
	Expiry time.Time

	// Raw optionally contains extra metadata from the server
	// when updating a token.
	Raw interface{}
}

// tokenJSON is the struct representing the HTTP response from OAuth2
// providers returning a token or error in JSON form.
// https://datatracker.ietf.org/doc/html/rfc6749#section-5.1
type tokenJSON struct {
	AccessToken  string         `json:"access_token"`
	TokenType    string         `json:"token_type"`
	RefreshToken string         `json:"refresh_token"`
	ExpiresIn    expirationTime `json:"expires_in"` // at least PayPal returns string, while most return number
	// error fields
	// https://datatracker.ietf.org/doc/html/rfc6749#section-5.2
	ErrorCode        string `json:"error"`
	ErrorDescription string `json:"error_description"`
	ErrorURI         string `json:"error_uri"`
}

func (e *tokenJSON) expiry() (t time.Time) {
	if v := e.ExpiresIn; v != 0 {
		return time.Now().Add(time.Duration(v) * time.Second)
	}
	return
}

type expirationTime int32

func (e *expirationTime) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	var n json.Number
	err := json.Unmarshal(b, &n)
	if err != nil {
		return err
	}
	i, err := n.Int64()
	if err != nil {
		return err
	}
	if i > math.MaxInt32 {
		i = math.MaxInt32
	}
	*e = expirationTime(i)
	return nil
}

// RegisterBrokenAuthHeaderProvider previously did something. It is now a no-op.
//
// Deprecated: this function no longer does anything. Caller code that
// wants to avoid potential extra HTTP requests made during
// auto-probing of the provider's auth style should set
// Endpoint.AuthStyle.
func RegisterBrokenAuthHeaderProvider(tokenURL string) {}

// AuthStyle is a copy of the golang.org/x/oauth2 package's AuthStyle type.
type AuthStyle int

const (
	AuthStyleUnknown  AuthStyle = 0
	AuthStyleInParams AuthStyle = 1
	AuthStyleInHeader AuthStyle = 2
)

// LazyAuthStyleCache is a backwards compatibility compromise to let Configs
// have a lazily-initialized AuthStyleCache.
//
// The two users of this, oauth2.Config and oauth2/clientcredentials.Config,
// both would ideally just embed an unexported AuthStyleCache but because both
// were historically allowed to be copied by value we can't retroactively add an
// uncopyable Mutex to them.
//
// We could use an atomic.Pointer, but that was added recently enough (in Go
// 1.18) that we'd break Go 1.17 users where the tests as of 2023-08-03
// still pass. By using an atomic.Value, it supports both Go 1.17 and
// copying by value, even if that's not ideal.
type LazyAuthStyleCache struct {
	v atomic.Value // of *AuthStyleCache
}

func (lc *LazyAuthStyleCache) Get() *AuthStyleCache {
	if c, ok := lc.v.Load().(*AuthStyleCache); ok {
		return c
	}
	c := new(AuthStyleCache)
	if !lc.v.CompareAndSwap(nil, c) {
		c = lc.v.Load().(*AuthStyleCache)
	}
	return c
}

// AuthStyleCache is the set of tokenURLs we've successfully used via
// RetrieveToken and which style auth we ended up using.
// It's called a cache, but it doesn't (yet?) shrink. It's expected that
// the set of OAuth2 servers a program contacts over time is fixed and
// small.
type AuthStyleCache struct {
	mu sync.Mutex
	m  map[string]AuthStyle // keyed by tokenURL
}

// lookupAuthStyle reports which auth style we last used with tokenURL
// when calling RetrieveToken and whether we have ever done so.
func (c *AuthStyleCache) lookupAuthStyle(tokenURL string) (style AuthStyle, ok bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	style, ok = c.m[tokenURL]
	return
}

// setAuthStyle adds an entry to authStyleCache, documented above.
func (c *AuthStyleCache) setAuthStyle(tokenURL string, v AuthStyle) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.m == nil {
		c.m = make(map[string]AuthStyle)
	}
	c.m[tokenURL] = v
}

// newTokenRequest returns a new *http.Request to retrieve a new token
// from tokenURL using the provided clientID, clientSecret, and POST
// body parameters.
//
// inParams is whether the clientID & clientSecret should be encoded
// as the POST body. An 'inParams' value of true means to send it in
// the POST body (along with any values in v); false means to send it
// in the Authorization header.
func newTokenRequest(tokenURL, clientID, clientSecret string, v url.Values, authStyle AuthStyle) (*http.Request, error) {
	if authStyle == AuthStyleInParams {
		v = cloneURLValues(v)
		if clientID != "" {
			v.Set("client_id", clientID)
		}
		if clientSecret != "" {
			v.Set("client_secret", clientSecret)
		}
	}
	req, err := http.NewRequest("POST", tokenURL, strings.NewReader(v.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if authStyle == AuthStyleInHeader {
		req.SetBasicAuth(url.QueryEscape(clientID), url.QueryEscape(clientSecret))
	}
	return req, nil
}

func cloneURLValues(v url.Values) url.Values {
	v2 := make(url.Values, len(v))
	for k, vv := range v {
		v2[k] = append([]string(nil), vv...)
	}
	return v2
}

func RetrieveToken(ctx context.Context, clientID, clientSecret, tokenURL string, v url.Values, authStyle AuthStyle, styleCache *AuthStyleCache) (*Token, error) {
	needsAuthStyleProbe := authStyle == 0
	if needsAuthStyleProbe {
		if style, ok := styleCache.lookupAuthStyle(tokenURL); ok {
			authStyle = style
			needsAuthStyleProbe = false
		} else {
			authStyle = AuthStyleInHeader // the first way we'll try
		}
	}
	req, err := newTokenRequest(tokenURL, clientID, clientSecret, v, authStyle)
	if err != nil {
		return nil, err
	}
	token, err := doTokenRoundTrip(ctx, req)
	if err != nil && needsAuthStyleProbe {
		// If we get an error, assume the server wants the
		// clientID & clientSecret in a different form.
		// See https://code.google.com/p/goauth2/issues/detail?id=31 for background.
		// In summary:
		// - Reddit only accepts client secret in the Authorization header
		// - Dropbox accepts either it in URL param or Auth header, but not both.
		// - Google only accepts URL param (not spec compliant?), not Auth header
		// - Stripe only accepts client secret in Auth header with Bearer method, not Basic
		//
		// We used to maintain a big table in this code of all the sites and which way
		// they went, but maintaining it didn't scale & got annoying.
		// So just try both ways.
		authStyle = AuthStyleInParams // the second way we'll try
		req, _ = newTokenRequest(tokenURL, clientID, clientSecret, v, authStyle)
		token, err = doTokenRoundTrip(ctx, req)
	}
	if needsAuthStyleProbe && err == nil {
		styleCache.setAuthStyle(tokenURL, authStyle)
	}
	// Don't overwrite `RefreshToken` with an empty value
	// if this was a token refreshing request.
	if token != nil && token.RefreshToken == "" {
		token.RefreshToken = v.Get("refresh_token")
	}
	return token, err
}

func doTokenRoundTrip(ctx context.Context, req *http.Request) (*Token, error) {
	r, err := ContextClient(ctx).Do(req.WithContext(ctx))
	if err != nil {
		return nil, err
	}
	body, err := ioutil.ReadAll(io.LimitReader(r.Body, 1<<20))
	r.Body.Close()
	if err != nil {
		return nil, fmt.Errorf("oauth2: cannot fetch token: %v", err)
	}

	failureStatus := r.StatusCode < 200 || r.StatusCode > 299
	retrieveError := &RetrieveError{
		Response: r,
		Body:     body,
		// attempt to populate error detail below
	}

	var token *Token
	content, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	switch content {
	case "application/x-www-form-urlencoded", "text/plain":
		// some endpoints return a query string
		vals, err := url.ParseQuery(string(body))
		if err != nil {
			if failureStatus {
				return nil, retrieveError
			}
			return nil, fmt.Errorf("oauth2: cannot parse response: %v", err)
		}
		retrieveError.ErrorCode = vals.Get("error")
		retrieveError.ErrorDescription = vals.Get("error_description")
		retrieveError.ErrorURI = vals.Get("error_uri")
		token = &Token{
			AccessToken:  vals.Get("access_token"),
			TokenType:    vals.Get("token_type"),
			RefreshToken: vals.Get("refresh_token"),
			Raw:          vals,
		}
		e := vals.Get("expires_in")
		expires, _ := strconv.Atoi(e)
		if expires != 0 {
			token.Expiry = time.Now().Add(time.Duration(expires) * time.Second)
		}
	default:
		var tj tokenJSON
		if err = json.Unmarshal(body, &tj); err != nil {
			if failureStatus {
				return nil, retrieveError
			}
			return nil, fmt.Errorf("oauth2: cannot parse json: %v", err)
		}
		retrieveError.ErrorCode = tj.ErrorCode
		retrieveError.ErrorDescription = tj.ErrorDescription
		retrieveError.ErrorURI = tj.ErrorURI
		token = &Token{
			AccessToken:  tj.AccessToken,
			TokenType:    tj.TokenType,
			RefreshToken: tj.RefreshToken,
			Expiry:       tj.expiry(),
			Raw:          make(map[string]interface{}),
		}
		json.Unmarshal(body, &token.Raw) // no error checks for optional fields
	}
	// according to spec, servers should respond status 400 in error case
	// https://www.rfc-editor.org/rfc/rfc6749#section-5.2
	// but some unorthodox servers respond 200 in error case
	if failureStatus || retrieveError.ErrorCode != "" {
		return nil, retrieveError
	}
	if token.AccessToken == "" {
		return nil, errors.New("oauth2: server response missing access_token")
	}
	return token, nil
}

// mirrors oauth2.RetrieveError
type RetrieveError struct {
	Response         *http.Response
	Body             []byte
	ErrorCode        string
	ErrorDescription string
	ErrorURI         string
}

func (r *RetrieveError) Error() string {
	if r.ErrorCode != "" {
		s := fmt.Sprintf("oauth2: %q", r.ErrorCode)
		if r.ErrorDescription != "" {
			s += fmt.Sprintf(" %q", r.ErrorDescription)
		}
		if r.ErrorURI != "" {
			s += fmt.Sprintf(" %q", r.ErrorURI)
		}
		return s
	}
	return fmt.Sprintf("oauth2: cannot fetch token: %v\nResponse: %s", r.Response.Status, r.Body)
}
