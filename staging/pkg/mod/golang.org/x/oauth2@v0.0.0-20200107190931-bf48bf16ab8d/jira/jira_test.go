// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jira

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/jws"
)

func TestJWTFetch_JSONResponse(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"access_token": "90d64460d14870c08c81352a05dedd3465940a7c",
			"token_type": "Bearer",
			"expires_in": 3600
		}`))
	}))
	defer ts.Close()

	conf := &Config{
		BaseURL: "https://my.app.com",
		Subject: "useraccountId",
		Config: oauth2.Config{
			ClientID:     "super_secret_client_id",
			ClientSecret: "super_shared_secret",
			Scopes:       []string{"read", "write"},
			Endpoint: oauth2.Endpoint{
				AuthURL:  "https://example.com",
				TokenURL: ts.URL,
			},
		},
	}

	tok, err := conf.TokenSource(context.Background()).Token()
	if err != nil {
		t.Fatal(err)
	}
	if !tok.Valid() {
		t.Errorf("got invalid token: %v", tok)
	}
	if got, want := tok.AccessToken, "90d64460d14870c08c81352a05dedd3465940a7c"; got != want {
		t.Errorf("access token = %q; want %q", got, want)
	}
	if got, want := tok.TokenType, "Bearer"; got != want {
		t.Errorf("token type = %q; want %q", got, want)
	}
	if got := tok.Expiry.IsZero(); got {
		t.Errorf("token expiry = %v, want none", got)
	}
}

func TestJWTFetch_BadResponse(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"token_type": "Bearer"}`))
	}))
	defer ts.Close()

	conf := &Config{
		BaseURL: "https://my.app.com",
		Subject: "useraccountId",
		Config: oauth2.Config{
			ClientID:     "super_secret_client_id",
			ClientSecret: "super_shared_secret",
			Scopes:       []string{"read", "write"},
			Endpoint: oauth2.Endpoint{
				AuthURL:  "https://example.com",
				TokenURL: ts.URL,
			},
		},
	}

	tok, err := conf.TokenSource(context.Background()).Token()
	if err != nil {
		t.Fatal(err)
	}
	if tok == nil {
		t.Fatalf("got nil token; want token")
	}
	if tok.Valid() {
		t.Errorf("got invalid token: %v", tok)
	}
	if got, want := tok.AccessToken, ""; got != want {
		t.Errorf("access token = %q; want %q", got, want)
	}
	if got, want := tok.TokenType, "Bearer"; got != want {
		t.Errorf("token type = %q; want %q", got, want)
	}
}

func TestJWTFetch_BadResponseType(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"access_token":123, "token_type": "Bearer"}`))
	}))
	defer ts.Close()

	conf := &Config{
		BaseURL: "https://my.app.com",
		Subject: "useraccountId",
		Config: oauth2.Config{
			ClientID:     "super_secret_client_id",
			ClientSecret: "super_shared_secret",
			Endpoint: oauth2.Endpoint{
				AuthURL:  "https://example.com",
				TokenURL: ts.URL,
			},
		},
	}

	tok, err := conf.TokenSource(context.Background()).Token()
	if err == nil {
		t.Error("got a token; expected error")
		if got, want := tok.AccessToken, ""; got != want {
			t.Errorf("access token = %q; want %q", got, want)
		}
	}
}

func TestJWTFetch_Assertion(t *testing.T) {
	var assertion string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.ParseForm()
		assertion = r.Form.Get("assertion")

		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"access_token": "90d64460d14870c08c81352a05dedd3465940a7c",
			"token_type": "Bearer",
			"expires_in": 3600
		}`))
	}))
	defer ts.Close()

	conf := &Config{
		BaseURL: "https://my.app.com",
		Subject: "useraccountId",
		Config: oauth2.Config{
			ClientID:     "super_secret_client_id",
			ClientSecret: "super_shared_secret",
			Endpoint: oauth2.Endpoint{
				AuthURL:  "https://example.com",
				TokenURL: ts.URL,
			},
		},
	}

	_, err := conf.TokenSource(context.Background()).Token()
	if err != nil {
		t.Fatalf("Failed to fetch token: %v", err)
	}

	parts := strings.Split(assertion, ".")
	if len(parts) != 3 {
		t.Fatalf("assertion = %q; want 3 parts", assertion)
	}
	gotjson, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		t.Fatalf("invalid token header; err = %v", err)
	}

	got := jws.Header{}
	if err := json.Unmarshal(gotjson, &got); err != nil {
		t.Errorf("failed to unmarshal json token header = %q; err = %v", gotjson, err)
	}

	want := jws.Header{
		Algorithm: "HS256",
		Typ:       "JWT",
	}
	if got != want {
		t.Errorf("access token header = %q; want %q", got, want)
	}
}
