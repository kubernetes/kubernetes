// Copyright 2014 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package oauth2 provides support for making
// OAuth2 authorized and authenticated HTTP requests.
// It can additionally grant authorization with Bearer JWT.
package oauth2

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"mime"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coreos/etcd/Godeps/_workspace/src/golang.org/x/net/context"
)

// NoContext is the default context you should supply if not using
// your own context.Context (see https://golang.org/x/net/context).
var NoContext = context.TODO()

// Config describes a typical 3-legged OAuth2 flow, with both the
// client application information and the server's endpoint URLs.
type Config struct {
	// ClientID is the application's ID.
	ClientID string

	// ClientSecret is the application's secret.
	ClientSecret string

	// Endpoint contains the resource server's token endpoint
	// URLs. These are constants specific to each server and are
	// often available via site-specific packages, such as
	// google.Endpoint or github.Endpoint.
	Endpoint Endpoint

	// RedirectURL is the URL to redirect users going through
	// the OAuth flow, after the resource owner's URLs.
	RedirectURL string

	// Scope specifies optional requested permissions.
	Scopes []string
}

// A TokenSource is anything that can return a token.
type TokenSource interface {
	// Token returns a token or an error.
	// Token must be safe for concurrent use by multiple goroutines.
	// The returned Token must not be modified.
	Token() (*Token, error)
}

// Endpoint contains the OAuth 2.0 provider's authorization and token
// endpoint URLs.
type Endpoint struct {
	AuthURL  string
	TokenURL string
}

var (
	// AccessTypeOnline and AccessTypeOffline are options passed
	// to the Options.AuthCodeURL method. They modify the
	// "access_type" field that gets sent in the URL returned by
	// AuthCodeURL.
	//
	// Online is the default if neither is specified. If your
	// application needs to refresh access tokens when the user
	// is not present at the browser, then use offline. This will
	// result in your application obtaining a refresh token the
	// first time your application exchanges an authorization
	// code for a user.
	AccessTypeOnline  AuthCodeOption = SetParam("access_type", "online")
	AccessTypeOffline AuthCodeOption = SetParam("access_type", "offline")

	// ApprovalForce forces the users to view the consent dialog
	// and confirm the permissions request at the URL returned
	// from AuthCodeURL, even if they've already done so.
	ApprovalForce AuthCodeOption = SetParam("approval_prompt", "force")
)

// An AuthCodeOption is passed to Config.AuthCodeURL.
type AuthCodeOption interface {
	setValue(url.Values)
}

type setParam struct{ k, v string }

func (p setParam) setValue(m url.Values) { m.Set(p.k, p.v) }

// SetParam builds an AuthCodeOption which passes key/value parameters
// to a provider's authorization endpoint.
func SetParam(key, value string) AuthCodeOption {
	return setParam{key, value}
}

// AuthCodeURL returns a URL to OAuth 2.0 provider's consent page
// that asks for permissions for the required scopes explicitly.
//
// State is a token to protect the user from CSRF attacks. You must
// always provide a non-zero string and validate that it matches the
// the state query parameter on your redirect callback.
// See http://tools.ietf.org/html/rfc6749#section-10.12 for more info.
//
// Opts may include AccessTypeOnline or AccessTypeOffline, as well
// as ApprovalForce.
func (c *Config) AuthCodeURL(state string, opts ...AuthCodeOption) string {
	var buf bytes.Buffer
	buf.WriteString(c.Endpoint.AuthURL)
	v := url.Values{
		"response_type": {"code"},
		"client_id":     {c.ClientID},
		"redirect_uri":  condVal(c.RedirectURL),
		"scope":         condVal(strings.Join(c.Scopes, " ")),
		"state":         condVal(state),
	}
	for _, opt := range opts {
		opt.setValue(v)
	}
	if strings.Contains(c.Endpoint.AuthURL, "?") {
		buf.WriteByte('&')
	} else {
		buf.WriteByte('?')
	}
	buf.WriteString(v.Encode())
	return buf.String()
}

// PasswordCredentialsToken converts a resource owner username and password
// pair into a token.
//
// Per the RFC, this grant type should only be used "when there is a high
// degree of trust between the resource owner and the client (e.g., the client
// is part of the device operating system or a highly privileged application),
// and when other authorization grant types are not available."
// See https://tools.ietf.org/html/rfc6749#section-4.3 for more info.
//
// The HTTP client to use is derived from the context.
// If nil, http.DefaultClient is used.
func (c *Config) PasswordCredentialsToken(ctx context.Context, username, password string) (*Token, error) {
	return retrieveToken(ctx, c, url.Values{
		"grant_type": {"password"},
		"username":   {username},
		"password":   {password},
		"scope":      condVal(strings.Join(c.Scopes, " ")),
	})
}

// Exchange converts an authorization code into a token.
//
// It is used after a resource provider redirects the user back
// to the Redirect URI (the URL obtained from AuthCodeURL).
//
// The HTTP client to use is derived from the context.
// If a client is not provided via the context, http.DefaultClient is used.
//
// The code will be in the *http.Request.FormValue("code"). Before
// calling Exchange, be sure to validate FormValue("state").
func (c *Config) Exchange(ctx context.Context, code string) (*Token, error) {
	return retrieveToken(ctx, c, url.Values{
		"grant_type":   {"authorization_code"},
		"code":         {code},
		"redirect_uri": condVal(c.RedirectURL),
		"scope":        condVal(strings.Join(c.Scopes, " ")),
	})
}

// contextClientFunc is a func which tries to return an *http.Client
// given a Context value. If it returns an error, the search stops
// with that error.  If it returns (nil, nil), the search continues
// down the list of registered funcs.
type contextClientFunc func(context.Context) (*http.Client, error)

var contextClientFuncs []contextClientFunc

func registerContextClientFunc(fn contextClientFunc) {
	contextClientFuncs = append(contextClientFuncs, fn)
}

func contextClient(ctx context.Context) (*http.Client, error) {
	for _, fn := range contextClientFuncs {
		c, err := fn(ctx)
		if err != nil {
			return nil, err
		}
		if c != nil {
			return c, nil
		}
	}
	if hc, ok := ctx.Value(HTTPClient).(*http.Client); ok {
		return hc, nil
	}
	return http.DefaultClient, nil
}

func contextTransport(ctx context.Context) http.RoundTripper {
	hc, err := contextClient(ctx)
	if err != nil {
		// This is a rare error case (somebody using nil on App Engine),
		// so I'd rather not everybody do an error check on this Client
		// method. They can get the error that they're doing it wrong
		// later, at client.Get/PostForm time.
		return errorTransport{err}
	}
	return hc.Transport
}

// Client returns an HTTP client using the provided token.
// The token will auto-refresh as necessary. The underlying
// HTTP transport will be obtained using the provided context.
// The returned client and its Transport should not be modified.
func (c *Config) Client(ctx context.Context, t *Token) *http.Client {
	return NewClient(ctx, c.TokenSource(ctx, t))
}

// TokenSource returns a TokenSource that returns t until t expires,
// automatically refreshing it as necessary using the provided context.
//
// Most users will use Config.Client instead.
func (c *Config) TokenSource(ctx context.Context, t *Token) TokenSource {
	tkr := &tokenRefresher{
		ctx:  ctx,
		conf: c,
	}
	if t != nil {
		tkr.refreshToken = t.RefreshToken
	}
	return &reuseTokenSource{
		t:   t,
		new: tkr,
	}
}

// tokenRefresher is a TokenSource that makes "grant_type"=="refresh_token"
// HTTP requests to renew a token using a RefreshToken.
type tokenRefresher struct {
	ctx          context.Context // used to get HTTP requests
	conf         *Config
	refreshToken string
}

// WARNING: Token is not safe for concurrent access, as it
// updates the tokenRefresher's refreshToken field.
// Within this package, it is used by reuseTokenSource which
// synchronizes calls to this method with its own mutex.
func (tf *tokenRefresher) Token() (*Token, error) {
	if tf.refreshToken == "" {
		return nil, errors.New("oauth2: token expired and refresh token is not set")
	}

	tk, err := retrieveToken(tf.ctx, tf.conf, url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {tf.refreshToken},
	})

	if err != nil {
		return nil, err
	}
	if tf.refreshToken != tk.RefreshToken {
		tf.refreshToken = tk.RefreshToken
	}
	return tk, err
}

// reuseTokenSource is a TokenSource that holds a single token in memory
// and validates its expiry before each call to retrieve it with
// Token. If it's expired, it will be auto-refreshed using the
// new TokenSource.
type reuseTokenSource struct {
	new TokenSource // called when t is expired.

	mu sync.Mutex // guards t
	t  *Token
}

// Token returns the current token if it's still valid, else will
// refresh the current token (using r.Context for HTTP client
// information) and return the new one.
func (s *reuseTokenSource) Token() (*Token, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.t.Valid() {
		return s.t, nil
	}
	t, err := s.new.Token()
	if err != nil {
		return nil, err
	}
	s.t = t
	return t, nil
}

func retrieveToken(ctx context.Context, c *Config, v url.Values) (*Token, error) {
	hc, err := contextClient(ctx)
	if err != nil {
		return nil, err
	}
	v.Set("client_id", c.ClientID)
	bustedAuth := !providerAuthHeaderWorks(c.Endpoint.TokenURL)
	if bustedAuth && c.ClientSecret != "" {
		v.Set("client_secret", c.ClientSecret)
	}
	req, err := http.NewRequest("POST", c.Endpoint.TokenURL, strings.NewReader(v.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if !bustedAuth {
		req.SetBasicAuth(c.ClientID, c.ClientSecret)
	}
	r, err := hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer r.Body.Close()
	body, err := ioutil.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("oauth2: cannot fetch token: %v", err)
	}
	if code := r.StatusCode; code < 200 || code > 299 {
		return nil, fmt.Errorf("oauth2: cannot fetch token: %v\nResponse: %s", r.Status, body)
	}

	var token *Token
	content, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	switch content {
	case "application/x-www-form-urlencoded", "text/plain":
		vals, err := url.ParseQuery(string(body))
		if err != nil {
			return nil, err
		}
		token = &Token{
			AccessToken:  vals.Get("access_token"),
			TokenType:    vals.Get("token_type"),
			RefreshToken: vals.Get("refresh_token"),
			raw:          vals,
		}
		e := vals.Get("expires_in")
		if e == "" {
			// TODO(jbd): Facebook's OAuth2 implementation is broken and
			// returns expires_in field in expires. Remove the fallback to expires,
			// when Facebook fixes their implementation.
			e = vals.Get("expires")
		}
		expires, _ := strconv.Atoi(e)
		if expires != 0 {
			token.Expiry = time.Now().Add(time.Duration(expires) * time.Second)
		}
	default:
		var tj tokenJSON
		if err = json.Unmarshal(body, &tj); err != nil {
			return nil, err
		}
		token = &Token{
			AccessToken:  tj.AccessToken,
			TokenType:    tj.TokenType,
			RefreshToken: tj.RefreshToken,
			Expiry:       tj.expiry(),
			raw:          make(map[string]interface{}),
		}
		json.Unmarshal(body, &token.raw) // no error checks for optional fields
	}
	// Don't overwrite `RefreshToken` with an empty value
	// if this was a token refreshing request.
	if token.RefreshToken == "" {
		token.RefreshToken = v.Get("refresh_token")
	}
	return token, nil
}

// tokenJSON is the struct representing the HTTP response from OAuth2
// providers returning a token in JSON form.
type tokenJSON struct {
	AccessToken  string         `json:"access_token"`
	TokenType    string         `json:"token_type"`
	RefreshToken string         `json:"refresh_token"`
	ExpiresIn    expirationTime `json:"expires_in"` // at least PayPal returns string, while most return number
	Expires      expirationTime `json:"expires"`    // broken Facebook spelling of expires_in
}

func (e *tokenJSON) expiry() (t time.Time) {
	if v := e.ExpiresIn; v != 0 {
		return time.Now().Add(time.Duration(v) * time.Second)
	}
	if v := e.Expires; v != 0 {
		return time.Now().Add(time.Duration(v) * time.Second)
	}
	return
}

type expirationTime int32

func (e *expirationTime) UnmarshalJSON(b []byte) error {
	var n json.Number
	err := json.Unmarshal(b, &n)
	if err != nil {
		return err
	}
	i, err := n.Int64()
	if err != nil {
		return err
	}
	*e = expirationTime(i)
	return nil
}

func condVal(v string) []string {
	if v == "" {
		return nil
	}
	return []string{v}
}

var brokenAuthHeaderProviders = []string{
	"https://accounts.google.com/",
	"https://www.googleapis.com/",
	"https://github.com/",
	"https://api.instagram.com/",
	"https://www.douban.com/",
	"https://api.dropbox.com/",
	"https://api.soundcloud.com/",
	"https://www.linkedin.com/",
	"https://api.twitch.tv/",
	"https://oauth.vk.com/",
	"https://api.odnoklassniki.ru/",
	"https://connect.stripe.com/",
	"https://api.pushbullet.com/",
	"https://oauth.sandbox.trainingpeaks.com/",
	"https://oauth.trainingpeaks.com/",
	"https://www.strava.com/oauth/",
}

// providerAuthHeaderWorks reports whether the OAuth2 server identified by the tokenURL
// implements the OAuth2 spec correctly
// See https://code.google.com/p/goauth2/issues/detail?id=31 for background.
// In summary:
// - Reddit only accepts client secret in the Authorization header
// - Dropbox accepts either it in URL param or Auth header, but not both.
// - Google only accepts URL param (not spec compliant?), not Auth header
// - Stripe only accepts client secret in Auth header with Bearer method, not Basic
func providerAuthHeaderWorks(tokenURL string) bool {
	for _, s := range brokenAuthHeaderProviders {
		if strings.HasPrefix(tokenURL, s) {
			// Some sites fail to implement the OAuth2 spec fully.
			return false
		}
	}

	// Assume the provider implements the spec properly
	// otherwise. We can add more exceptions as they're
	// discovered. We will _not_ be adding configurable hooks
	// to this package to let users select server bugs.
	return true
}

// HTTPClient is the context key to use with golang.org/x/net/context's
// WithValue function to associate an *http.Client value with a context.
var HTTPClient contextKey

// contextKey is just an empty struct. It exists so HTTPClient can be
// an immutable public variable with a unique type. It's immutable
// because nobody else can create a contextKey, being unexported.
type contextKey struct{}

// NewClient creates an *http.Client from a Context and TokenSource.
// The returned client is not valid beyond the lifetime of the context.
//
// As a special case, if src is nil, a non-OAuth2 client is returned
// using the provided context. This exists to support related OAuth2
// packages.
func NewClient(ctx context.Context, src TokenSource) *http.Client {
	if src == nil {
		c, err := contextClient(ctx)
		if err != nil {
			return &http.Client{Transport: errorTransport{err}}
		}
		return c
	}
	return &http.Client{
		Transport: &Transport{
			Base:   contextTransport(ctx),
			Source: ReuseTokenSource(nil, src),
		},
	}
}

// ReuseTokenSource returns a TokenSource which repeatedly returns the
// same token as long as it's valid, starting with t.
// When its cached token is invalid, a new token is obtained from src.
//
// ReuseTokenSource is typically used to reuse tokens from a cache
// (such as a file on disk) between runs of a program, rather than
// obtaining new tokens unnecessarily.
//
// The initial token t may be nil, in which case the TokenSource is
// wrapped in a caching version if it isn't one already. This also
// means it's always safe to wrap ReuseTokenSource around any other
// TokenSource without adverse effects.
func ReuseTokenSource(t *Token, src TokenSource) TokenSource {
	// Don't wrap a reuseTokenSource in itself. That would work,
	// but cause an unnecessary number of mutex operations.
	// Just build the equivalent one.
	if rt, ok := src.(*reuseTokenSource); ok {
		if t == nil {
			// Just use it directly.
			return rt
		}
		src = rt.new
	}
	return &reuseTokenSource{
		t:   t,
		new: src,
	}
}
