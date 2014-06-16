// Copyright 2011 The goauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The oauth package provides support for making
// OAuth2-authenticated HTTP requests.
//
// Example usage:
//
//	// Specify your configuration. (typically as a global variable)
//	var config = &oauth.Config{
//		ClientId:     YOUR_CLIENT_ID,
//		ClientSecret: YOUR_CLIENT_SECRET,
//		Scope:        "https://www.googleapis.com/auth/buzz",
//		AuthURL:      "https://accounts.google.com/o/oauth2/auth",
//		TokenURL:     "https://accounts.google.com/o/oauth2/token",
//		RedirectURL:  "http://you.example.org/handler",
//	}
//
//	// A landing page redirects to the OAuth provider to get the auth code.
//	func landing(w http.ResponseWriter, r *http.Request) {
//		http.Redirect(w, r, config.AuthCodeURL("foo"), http.StatusFound)
//	}
//
//	// The user will be redirected back to this handler, that takes the
//	// "code" query parameter and Exchanges it for an access token.
//	func handler(w http.ResponseWriter, r *http.Request) {
//		t := &oauth.Transport{Config: config}
//		t.Exchange(r.FormValue("code"))
//		// The Transport now has a valid Token. Create an *http.Client
//		// with which we can make authenticated API requests.
//		c := t.Client()
//		c.Post(...)
//		// ...
//		// btw, r.FormValue("state") == "foo"
//	}
//
package oauth

import (
	"encoding/json"
	"io/ioutil"
	"mime"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

type OAuthError struct {
	prefix string
	msg    string
}

func (oe OAuthError) Error() string {
	return "OAuthError: " + oe.prefix + ": " + oe.msg
}

// Cache specifies the methods that implement a Token cache.
type Cache interface {
	Token() (*Token, error)
	PutToken(*Token) error
}

// CacheFile implements Cache. Its value is the name of the file in which
// the Token is stored in JSON format.
type CacheFile string

func (f CacheFile) Token() (*Token, error) {
	file, err := os.Open(string(f))
	if err != nil {
		return nil, OAuthError{"CacheFile.Token", err.Error()}
	}
	defer file.Close()
	tok := &Token{}
	if err := json.NewDecoder(file).Decode(tok); err != nil {
		return nil, OAuthError{"CacheFile.Token", err.Error()}
	}
	return tok, nil
}

func (f CacheFile) PutToken(tok *Token) error {
	file, err := os.OpenFile(string(f), os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0600)
	if err != nil {
		return OAuthError{"CacheFile.PutToken", err.Error()}
	}
	if err := json.NewEncoder(file).Encode(tok); err != nil {
		file.Close()
		return OAuthError{"CacheFile.PutToken", err.Error()}
	}
	if err := file.Close(); err != nil {
		return OAuthError{"CacheFile.PutToken", err.Error()}
	}
	return nil
}

// Config is the configuration of an OAuth consumer.
type Config struct {
	// ClientId is the OAuth client identifier used when communicating with
	// the configured OAuth provider.
	ClientId string

	// ClientSecret is the OAuth client secret used when communicating with
	// the configured OAuth provider.
	ClientSecret string

	// Scope identifies the level of access being requested. Multiple scope
	// values should be provided as a space-delimited string.
	Scope string

	// AuthURL is the URL the user will be directed to in order to grant
	// access.
	AuthURL string

	// TokenURL is the URL used to retrieve OAuth tokens.
	TokenURL string

	// RedirectURL is the URL to which the user will be returned after
	// granting (or denying) access.
	RedirectURL string

	// TokenCache allows tokens to be cached for subsequent requests.
	TokenCache Cache

	AccessType string // Optional, "online" (default) or "offline", no refresh token if "online"

	// ApprovalPrompt indicates whether the user should be
	// re-prompted for consent. If set to "auto" (default) the
	// user will be prompted only if they haven't previously
	// granted consent and the code can only be exchanged for an
	// access token.
	// If set to "force" the user will always be prompted, and the
	// code can be exchanged for a refresh token.
	ApprovalPrompt string
}

// Token contains an end-user's tokens.
// This is the data you must store to persist authentication.
type Token struct {
	AccessToken  string
	RefreshToken string
	Expiry       time.Time         // If zero the token has no (known) expiry time.
	Extra        map[string]string // May be nil.
}

func (t *Token) Expired() bool {
	if t.Expiry.IsZero() {
		return false
	}
	return t.Expiry.Before(time.Now())
}

// Transport implements http.RoundTripper. When configured with a valid
// Config and Token it can be used to make authenticated HTTP requests.
//
//	t := &oauth.Transport{config}
//      t.Exchange(code)
//      // t now contains a valid Token
//	r, _, err := t.Client().Get("http://example.org/url/requiring/auth")
//
// It will automatically refresh the Token if it can,
// updating the supplied Token in place.
type Transport struct {
	*Config
	*Token

	// Transport is the HTTP transport to use when making requests.
	// It will default to http.DefaultTransport if nil.
	// (It should never be an oauth.Transport.)
	Transport http.RoundTripper
}

// Client returns an *http.Client that makes OAuth-authenticated requests.
func (t *Transport) Client() *http.Client {
	return &http.Client{Transport: t}
}

func (t *Transport) transport() http.RoundTripper {
	if t.Transport != nil {
		return t.Transport
	}
	return http.DefaultTransport
}

// AuthCodeURL returns a URL that the end-user should be redirected to,
// so that they may obtain an authorization code.
func (c *Config) AuthCodeURL(state string) string {
	url_, err := url.Parse(c.AuthURL)
	if err != nil {
		panic("AuthURL malformed: " + err.Error())
	}
	q := url.Values{
		"response_type":   {"code"},
		"client_id":       {c.ClientId},
		"redirect_uri":    {c.RedirectURL},
		"scope":           {c.Scope},
		"state":           {state},
		"access_type":     {c.AccessType},
		"approval_prompt": {c.ApprovalPrompt},
	}.Encode()
	if url_.RawQuery == "" {
		url_.RawQuery = q
	} else {
		url_.RawQuery += "&" + q
	}
	return url_.String()
}

// Exchange takes a code and gets access Token from the remote server.
func (t *Transport) Exchange(code string) (*Token, error) {
	if t.Config == nil {
		return nil, OAuthError{"Exchange", "no Config supplied"}
	}

	// If the transport or the cache already has a token, it is
	// passed to `updateToken` to preserve existing refresh token.
	tok := t.Token
	if tok == nil && t.TokenCache != nil {
		tok, _ = t.TokenCache.Token()
	}
	if tok == nil {
		tok = new(Token)
	}
	err := t.updateToken(tok, url.Values{
		"grant_type":   {"authorization_code"},
		"redirect_uri": {t.RedirectURL},
		"scope":        {t.Scope},
		"code":         {code},
	})
	if err != nil {
		return nil, err
	}
	t.Token = tok
	if t.TokenCache != nil {
		return tok, t.TokenCache.PutToken(tok)
	}
	return tok, nil
}

// RoundTrip executes a single HTTP transaction using the Transport's
// Token as authorization headers.
//
// This method will attempt to renew the Token if it has expired and may return
// an error related to that Token renewal before attempting the client request.
// If the Token cannot be renewed a non-nil os.Error value will be returned.
// If the Token is invalid callers should expect HTTP-level errors,
// as indicated by the Response's StatusCode.
func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	if t.Token == nil {
		if t.Config == nil {
			return nil, OAuthError{"RoundTrip", "no Config supplied"}
		}
		if t.TokenCache == nil {
			return nil, OAuthError{"RoundTrip", "no Token supplied"}
		}
		var err error
		t.Token, err = t.TokenCache.Token()
		if err != nil {
			return nil, err
		}
	}

	// Refresh the Token if it has expired.
	if t.Expired() {
		if err := t.Refresh(); err != nil {
			return nil, err
		}
	}

	// To set the Authorization header, we must make a copy of the Request
	// so that we don't modify the Request we were given.
	// This is required by the specification of http.RoundTripper.
	req = cloneRequest(req)
	req.Header.Set("Authorization", "Bearer "+t.AccessToken)

	// Make the HTTP request.
	return t.transport().RoundTrip(req)
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

// Refresh renews the Transport's AccessToken using its RefreshToken.
func (t *Transport) Refresh() error {
	if t.Token == nil {
		return OAuthError{"Refresh", "no existing Token"}
	}
	if t.RefreshToken == "" {
		return OAuthError{"Refresh", "Token expired; no Refresh Token"}
	}
	if t.Config == nil {
		return OAuthError{"Refresh", "no Config supplied"}
	}

	err := t.updateToken(t.Token, url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {t.RefreshToken},
	})
	if err != nil {
		return err
	}
	if t.TokenCache != nil {
		return t.TokenCache.PutToken(t.Token)
	}
	return nil
}

func (t *Transport) updateToken(tok *Token, v url.Values) error {
	v.Set("client_id", t.ClientId)
	v.Set("client_secret", t.ClientSecret)
	client := &http.Client{Transport: t.transport()}
	req, err := http.NewRequest("POST", t.TokenURL, strings.NewReader(v.Encode()))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.SetBasicAuth(t.ClientId, t.ClientSecret)
	r, err := client.Do(req)
	if err != nil {
		return err
	}
	defer r.Body.Close()
	if r.StatusCode != 200 {
		return OAuthError{"updateToken", r.Status}
	}
	var b struct {
		Access    string        `json:"access_token"`
		Refresh   string        `json:"refresh_token"`
		ExpiresIn time.Duration `json:"expires_in"`
		Id        string        `json:"id_token"`
	}

	content, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	switch content {
	case "application/x-www-form-urlencoded", "text/plain":
		body, err := ioutil.ReadAll(r.Body)
		if err != nil {
			return err
		}
		vals, err := url.ParseQuery(string(body))
		if err != nil {
			return err
		}

		b.Access = vals.Get("access_token")
		b.Refresh = vals.Get("refresh_token")
		b.ExpiresIn, _ = time.ParseDuration(vals.Get("expires_in") + "s")
		b.Id = vals.Get("id_token")
	default:
		if err = json.NewDecoder(r.Body).Decode(&b); err != nil {
			return err
		}
		// The JSON parser treats the unitless ExpiresIn like 'ns' instead of 's' as above,
		// so compensate here.
		b.ExpiresIn *= time.Second
	}
	tok.AccessToken = b.Access
	// Don't overwrite `RefreshToken` with an empty value
	if len(b.Refresh) > 0 {
		tok.RefreshToken = b.Refresh
	}
	if b.ExpiresIn == 0 {
		tok.Expiry = time.Time{}
	} else {
		tok.Expiry = time.Now().Add(b.ExpiresIn)
	}
	if b.Id != "" {
		if tok.Extra == nil {
			tok.Extra = make(map[string]string)
		}
		tok.Extra["id_token"] = b.Id
	}
	return nil
}
