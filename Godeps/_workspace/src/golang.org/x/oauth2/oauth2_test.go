// Copyright 2014 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oauth2

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"testing"
	"time"

	"golang.org/x/net/context"
)

type mockTransport struct {
	rt func(req *http.Request) (resp *http.Response, err error)
}

func (t *mockTransport) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	return t.rt(req)
}

type mockCache struct {
	token   *Token
	readErr error
}

func (c *mockCache) ReadToken() (*Token, error) {
	return c.token, c.readErr
}

func (c *mockCache) WriteToken(*Token) {
	// do nothing
}

func newConf(url string) *Config {
	return &Config{
		ClientID:     "CLIENT_ID",
		ClientSecret: "CLIENT_SECRET",
		RedirectURL:  "REDIRECT_URL",
		Scopes:       []string{"scope1", "scope2"},
		Endpoint: Endpoint{
			AuthURL:  url + "/auth",
			TokenURL: url + "/token",
		},
	}
}

func TestAuthCodeURL(t *testing.T) {
	conf := newConf("server")
	url := conf.AuthCodeURL("foo", AccessTypeOffline, ApprovalForce)
	if url != "server/auth?access_type=offline&approval_prompt=force&client_id=CLIENT_ID&redirect_uri=REDIRECT_URL&response_type=code&scope=scope1+scope2&state=foo" {
		t.Errorf("Auth code URL doesn't match the expected, found: %v", url)
	}
}

func TestAuthCodeURL_CustomParam(t *testing.T) {
	conf := newConf("server")
	param := SetAuthURLParam("foo", "bar")
	url := conf.AuthCodeURL("baz", param)
	if url != "server/auth?client_id=CLIENT_ID&foo=bar&redirect_uri=REDIRECT_URL&response_type=code&scope=scope1+scope2&state=baz" {
		t.Errorf("Auth code URL doesn't match the expected, found: %v", url)
	}
}

func TestAuthCodeURL_Optional(t *testing.T) {
	conf := &Config{
		ClientID: "CLIENT_ID",
		Endpoint: Endpoint{
			AuthURL:  "/auth-url",
			TokenURL: "/token-url",
		},
	}
	url := conf.AuthCodeURL("")
	if url != "/auth-url?client_id=CLIENT_ID&response_type=code" {
		t.Fatalf("Auth code URL doesn't match the expected, found: %v", url)
	}
}

func TestExchangeRequest(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.String() != "/token" {
			t.Errorf("Unexpected exchange request URL, %v is found.", r.URL)
		}
		headerAuth := r.Header.Get("Authorization")
		if headerAuth != "Basic Q0xJRU5UX0lEOkNMSUVOVF9TRUNSRVQ=" {
			t.Errorf("Unexpected authorization header, %v is found.", headerAuth)
		}
		headerContentType := r.Header.Get("Content-Type")
		if headerContentType != "application/x-www-form-urlencoded" {
			t.Errorf("Unexpected Content-Type header, %v is found.", headerContentType)
		}
		body, err := ioutil.ReadAll(r.Body)
		if err != nil {
			t.Errorf("Failed reading request body: %s.", err)
		}
		if string(body) != "client_id=CLIENT_ID&code=exchange-code&grant_type=authorization_code&redirect_uri=REDIRECT_URL&scope=scope1+scope2" {
			t.Errorf("Unexpected exchange payload, %v is found.", string(body))
		}
		w.Header().Set("Content-Type", "application/x-www-form-urlencoded")
		w.Write([]byte("access_token=90d64460d14870c08c81352a05dedd3465940a7c&scope=user&token_type=bearer"))
	}))
	defer ts.Close()
	conf := newConf(ts.URL)
	tok, err := conf.Exchange(NoContext, "exchange-code")
	if err != nil {
		t.Error(err)
	}
	if !tok.Valid() {
		t.Fatalf("Token invalid. Got: %#v", tok)
	}
	if tok.AccessToken != "90d64460d14870c08c81352a05dedd3465940a7c" {
		t.Errorf("Unexpected access token, %#v.", tok.AccessToken)
	}
	if tok.TokenType != "bearer" {
		t.Errorf("Unexpected token type, %#v.", tok.TokenType)
	}
	scope := tok.Extra("scope")
	if scope != "user" {
		t.Errorf("Unexpected value for scope: %v", scope)
	}
}

func TestExchangeRequest_JSONResponse(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.String() != "/token" {
			t.Errorf("Unexpected exchange request URL, %v is found.", r.URL)
		}
		headerAuth := r.Header.Get("Authorization")
		if headerAuth != "Basic Q0xJRU5UX0lEOkNMSUVOVF9TRUNSRVQ=" {
			t.Errorf("Unexpected authorization header, %v is found.", headerAuth)
		}
		headerContentType := r.Header.Get("Content-Type")
		if headerContentType != "application/x-www-form-urlencoded" {
			t.Errorf("Unexpected Content-Type header, %v is found.", headerContentType)
		}
		body, err := ioutil.ReadAll(r.Body)
		if err != nil {
			t.Errorf("Failed reading request body: %s.", err)
		}
		if string(body) != "client_id=CLIENT_ID&code=exchange-code&grant_type=authorization_code&redirect_uri=REDIRECT_URL&scope=scope1+scope2" {
			t.Errorf("Unexpected exchange payload, %v is found.", string(body))
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"access_token": "90d64460d14870c08c81352a05dedd3465940a7c", "scope": "user", "token_type": "bearer", "expires_in": 86400}`))
	}))
	defer ts.Close()
	conf := newConf(ts.URL)
	tok, err := conf.Exchange(NoContext, "exchange-code")
	if err != nil {
		t.Error(err)
	}
	if !tok.Valid() {
		t.Fatalf("Token invalid. Got: %#v", tok)
	}
	if tok.AccessToken != "90d64460d14870c08c81352a05dedd3465940a7c" {
		t.Errorf("Unexpected access token, %#v.", tok.AccessToken)
	}
	if tok.TokenType != "bearer" {
		t.Errorf("Unexpected token type, %#v.", tok.TokenType)
	}
	scope := tok.Extra("scope")
	if scope != "user" {
		t.Errorf("Unexpected value for scope: %v", scope)
	}
}

const day = 24 * time.Hour

func TestExchangeRequest_JSONResponse_Expiry(t *testing.T) {
	seconds := int32(day.Seconds())
	jsonNumberType := reflect.TypeOf(json.Number("0"))
	for _, c := range []struct {
		expires string
		expect  error
	}{
		{fmt.Sprintf(`"expires_in": %d`, seconds), nil},
		{fmt.Sprintf(`"expires_in": "%d"`, seconds), nil},                                             // PayPal case
		{fmt.Sprintf(`"expires": %d`, seconds), nil},                                                  // Facebook case
		{`"expires": false`, &json.UnmarshalTypeError{Value: "bool", Type: jsonNumberType}},           // wrong type
		{`"expires": {}`, &json.UnmarshalTypeError{Value: "object", Type: jsonNumberType}},            // wrong type
		{`"expires": "zzz"`, &strconv.NumError{Func: "ParseInt", Num: "zzz", Err: strconv.ErrSyntax}}, // wrong value
	} {
		testExchangeRequest_JSONResponse_expiry(t, c.expires, c.expect)
	}
}

func testExchangeRequest_JSONResponse_expiry(t *testing.T, exp string, expect error) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(fmt.Sprintf(`{"access_token": "90d", "scope": "user", "token_type": "bearer", %s}`, exp)))
	}))
	defer ts.Close()
	conf := newConf(ts.URL)
	t1 := time.Now().Add(day)
	tok, err := conf.Exchange(NoContext, "exchange-code")
	t2 := time.Now().Add(day)
	// Do a fmt.Sprint comparison so either side can be
	// nil. fmt.Sprint just stringifies them to "<nil>", and no
	// non-nil expected error ever stringifies as "<nil>", so this
	// isn't terribly disgusting.  We do this because Go 1.4 and
	// Go 1.5 return a different deep value for
	// json.UnmarshalTypeError.  In Go 1.5, the
	// json.UnmarshalTypeError contains a new field with a new
	// non-zero value.  Rather than ignore it here with reflect or
	// add new files and +build tags, just look at the strings.
	if fmt.Sprint(err) != fmt.Sprint(expect) {
		t.Errorf("Error = %v; want %v", err, expect)
	}
	if err != nil {
		return
	}
	if !tok.Valid() {
		t.Fatalf("Token invalid. Got: %#v", tok)
	}
	expiry := tok.Expiry
	if expiry.Before(t1) || expiry.After(t2) {
		t.Errorf("Unexpected value for Expiry: %v (shold be between %v and %v)", expiry, t1, t2)
	}
}

func TestExchangeRequest_BadResponse(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"scope": "user", "token_type": "bearer"}`))
	}))
	defer ts.Close()
	conf := newConf(ts.URL)
	tok, err := conf.Exchange(NoContext, "code")
	if err != nil {
		t.Fatal(err)
	}
	if tok.AccessToken != "" {
		t.Errorf("Unexpected access token, %#v.", tok.AccessToken)
	}
}

func TestExchangeRequest_BadResponseType(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"access_token":123,  "scope": "user", "token_type": "bearer"}`))
	}))
	defer ts.Close()
	conf := newConf(ts.URL)
	_, err := conf.Exchange(NoContext, "exchange-code")
	if err == nil {
		t.Error("expected error from invalid access_token type")
	}
}

func TestExchangeRequest_NonBasicAuth(t *testing.T) {
	tr := &mockTransport{
		rt: func(r *http.Request) (w *http.Response, err error) {
			headerAuth := r.Header.Get("Authorization")
			if headerAuth != "" {
				t.Errorf("Unexpected authorization header, %v is found.", headerAuth)
			}
			return nil, errors.New("no response")
		},
	}
	c := &http.Client{Transport: tr}
	conf := &Config{
		ClientID: "CLIENT_ID",
		Endpoint: Endpoint{
			AuthURL:  "https://accounts.google.com/auth",
			TokenURL: "https://accounts.google.com/token",
		},
	}

	ctx := context.WithValue(context.Background(), HTTPClient, c)
	conf.Exchange(ctx, "code")
}

func TestPasswordCredentialsTokenRequest(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		expected := "/token"
		if r.URL.String() != expected {
			t.Errorf("URL = %q; want %q", r.URL, expected)
		}
		headerAuth := r.Header.Get("Authorization")
		expected = "Basic Q0xJRU5UX0lEOkNMSUVOVF9TRUNSRVQ="
		if headerAuth != expected {
			t.Errorf("Authorization header = %q; want %q", headerAuth, expected)
		}
		headerContentType := r.Header.Get("Content-Type")
		expected = "application/x-www-form-urlencoded"
		if headerContentType != expected {
			t.Errorf("Content-Type header = %q; want %q", headerContentType, expected)
		}
		body, err := ioutil.ReadAll(r.Body)
		if err != nil {
			t.Errorf("Failed reading request body: %s.", err)
		}
		expected = "client_id=CLIENT_ID&grant_type=password&password=password1&scope=scope1+scope2&username=user1"
		if string(body) != expected {
			t.Errorf("res.Body = %q; want %q", string(body), expected)
		}
		w.Header().Set("Content-Type", "application/x-www-form-urlencoded")
		w.Write([]byte("access_token=90d64460d14870c08c81352a05dedd3465940a7c&scope=user&token_type=bearer"))
	}))
	defer ts.Close()
	conf := newConf(ts.URL)
	tok, err := conf.PasswordCredentialsToken(NoContext, "user1", "password1")
	if err != nil {
		t.Error(err)
	}
	if !tok.Valid() {
		t.Fatalf("Token invalid. Got: %#v", tok)
	}
	expected := "90d64460d14870c08c81352a05dedd3465940a7c"
	if tok.AccessToken != expected {
		t.Errorf("AccessToken = %q; want %q", tok.AccessToken, expected)
	}
	expected = "bearer"
	if tok.TokenType != expected {
		t.Errorf("TokenType = %q; want %q", tok.TokenType, expected)
	}
}

func TestTokenRefreshRequest(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.String() == "/somethingelse" {
			return
		}
		if r.URL.String() != "/token" {
			t.Errorf("Unexpected token refresh request URL, %v is found.", r.URL)
		}
		headerContentType := r.Header.Get("Content-Type")
		if headerContentType != "application/x-www-form-urlencoded" {
			t.Errorf("Unexpected Content-Type header, %v is found.", headerContentType)
		}
		body, _ := ioutil.ReadAll(r.Body)
		if string(body) != "client_id=CLIENT_ID&grant_type=refresh_token&refresh_token=REFRESH_TOKEN" {
			t.Errorf("Unexpected refresh token payload, %v is found.", string(body))
		}
	}))
	defer ts.Close()
	conf := newConf(ts.URL)
	c := conf.Client(NoContext, &Token{RefreshToken: "REFRESH_TOKEN"})
	c.Get(ts.URL + "/somethingelse")
}

func TestFetchWithNoRefreshToken(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.String() == "/somethingelse" {
			return
		}
		if r.URL.String() != "/token" {
			t.Errorf("Unexpected token refresh request URL, %v is found.", r.URL)
		}
		headerContentType := r.Header.Get("Content-Type")
		if headerContentType != "application/x-www-form-urlencoded" {
			t.Errorf("Unexpected Content-Type header, %v is found.", headerContentType)
		}
		body, _ := ioutil.ReadAll(r.Body)
		if string(body) != "client_id=CLIENT_ID&grant_type=refresh_token&refresh_token=REFRESH_TOKEN" {
			t.Errorf("Unexpected refresh token payload, %v is found.", string(body))
		}
	}))
	defer ts.Close()
	conf := newConf(ts.URL)
	c := conf.Client(NoContext, nil)
	_, err := c.Get(ts.URL + "/somethingelse")
	if err == nil {
		t.Errorf("Fetch should return an error if no refresh token is set")
	}
}

func TestRefreshToken_RefreshTokenReplacement(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"access_token":"ACCESS TOKEN",  "scope": "user", "token_type": "bearer", "refresh_token": "NEW REFRESH TOKEN"}`))
		return
	}))
	defer ts.Close()
	conf := newConf(ts.URL)
	tkr := tokenRefresher{
		conf:         conf,
		ctx:          NoContext,
		refreshToken: "OLD REFRESH TOKEN",
	}
	tk, err := tkr.Token()
	if err != nil {
		t.Errorf("Unexpected refreshToken error returned: %v", err)
		return
	}
	if tk.RefreshToken != tkr.refreshToken {
		t.Errorf("tokenRefresher.refresh_token = %s; want %s", tkr.refreshToken, tk.RefreshToken)
	}
}

func TestConfigClientWithToken(t *testing.T) {
	tok := &Token{
		AccessToken: "abc123",
	}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.Header.Get("Authorization"), fmt.Sprintf("Bearer %s", tok.AccessToken); got != want {
			t.Errorf("Authorization header = %q; want %q", got, want)
		}
		return
	}))
	defer ts.Close()
	conf := newConf(ts.URL)

	c := conf.Client(NoContext, tok)
	req, err := http.NewRequest("GET", ts.URL, nil)
	if err != nil {
		t.Error(err)
	}
	_, err = c.Do(req)
	if err != nil {
		t.Error(err)
	}
}
