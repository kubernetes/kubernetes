// Copyright 2011 The goauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oauth

import (
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

var requests = []struct {
	path, query, auth string // request
	contenttype, body string // response
}{
	{
		path:        "/token",
		query:       "grant_type=authorization_code&code=c0d3&client_id=cl13nt1d",
		contenttype: "application/json",
		auth:        "Basic Y2wxM250MWQ6czNjcjN0",
		body: `
			{
				"access_token":"token1",
				"refresh_token":"refreshtoken1",
				"id_token":"idtoken1",
				"expires_in":3600
			}
		`,
	},
	{path: "/secure", auth: "Bearer token1", body: "first payload"},
	{
		path:        "/token",
		query:       "grant_type=refresh_token&refresh_token=refreshtoken1&client_id=cl13nt1d",
		contenttype: "application/json",
		auth:        "Basic Y2wxM250MWQ6czNjcjN0",
		body: `
			{
				"access_token":"token2",
				"refresh_token":"refreshtoken2",
				"id_token":"idtoken2",
				"expires_in":3600
			}
		`,
	},
	{path: "/secure", auth: "Bearer token2", body: "second payload"},
	{
		path:        "/token",
		query:       "grant_type=refresh_token&refresh_token=refreshtoken2&client_id=cl13nt1d",
		contenttype: "application/x-www-form-urlencoded",
		body:        "access_token=token3&refresh_token=refreshtoken3&id_token=idtoken3&expires_in=3600",
		auth:        "Basic Y2wxM250MWQ6czNjcjN0",
	},
	{path: "/secure", auth: "Bearer token3", body: "third payload"},
	{
		path:        "/token",
		query:       "grant_type=client_credentials&client_id=cl13nt1d",
		contenttype: "application/json",
		auth:        "Basic Y2wxM250MWQ6czNjcjN0",
		body: `
			{
				"access_token":"token4",
				"expires_in":3600
			}
		`,
	},
	{path: "/secure", auth: "Bearer token4", body: "fourth payload"},
}

func TestOAuth(t *testing.T) {
	// Set up test server.
	n := 0
	handler := func(w http.ResponseWriter, r *http.Request) {
		if n >= len(requests) {
			t.Errorf("too many requests: %d", n)
			return
		}
		req := requests[n]
		n++

		// Check request.
		if g, w := r.URL.Path, req.path; g != w {
			t.Errorf("request[%d] got path %s, want %s", n, g, w)
		}
		want, _ := url.ParseQuery(req.query)
		for k := range want {
			if g, w := r.FormValue(k), want.Get(k); g != w {
				t.Errorf("query[%s] = %s, want %s", k, g, w)
			}
		}
		if g, w := r.Header.Get("Authorization"), req.auth; w != "" && g != w {
			t.Errorf("Authorization: %v, want %v", g, w)
		}

		// Send response.
		w.Header().Set("Content-Type", req.contenttype)
		io.WriteString(w, req.body)
	}
	server := httptest.NewServer(http.HandlerFunc(handler))
	defer server.Close()

	config := &Config{
		ClientId:     "cl13nt1d",
		ClientSecret: "s3cr3t",
		Scope:        "https://example.net/scope",
		AuthURL:      server.URL + "/auth",
		TokenURL:     server.URL + "/token",
	}

	// TODO(adg): test AuthCodeURL

	transport := &Transport{Config: config}
	_, err := transport.Exchange("c0d3")
	if err != nil {
		t.Fatalf("Exchange: %v", err)
	}
	checkToken(t, transport.Token, "token1", "refreshtoken1", "idtoken1")

	c := transport.Client()
	resp, err := c.Get(server.URL + "/secure")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	checkBody(t, resp, "first payload")

	// test automatic refresh
	transport.Expiry = time.Now().Add(-time.Hour)
	resp, err = c.Get(server.URL + "/secure")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	checkBody(t, resp, "second payload")
	checkToken(t, transport.Token, "token2", "refreshtoken2", "idtoken2")

	// refresh one more time, but get URL-encoded token instead of JSON
	transport.Expiry = time.Now().Add(-time.Hour)
	resp, err = c.Get(server.URL + "/secure")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	checkBody(t, resp, "third payload")
	checkToken(t, transport.Token, "token3", "refreshtoken3", "idtoken3")

	transport.Token = &Token{}
	err = transport.AuthenticateClient()
	if err != nil {
		t.Fatalf("AuthenticateClient: %v", err)
	}
	checkToken(t, transport.Token, "token4", "", "")
	resp, err = c.Get(server.URL + "/secure")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	checkBody(t, resp, "fourth payload")
}

func checkToken(t *testing.T, tok *Token, access, refresh, id string) {
	if g, w := tok.AccessToken, access; g != w {
		t.Errorf("AccessToken = %q, want %q", g, w)
	}
	if g, w := tok.RefreshToken, refresh; g != w {
		t.Errorf("RefreshToken = %q, want %q", g, w)
	}
	if g, w := tok.Extra["id_token"], id; g != w {
		t.Errorf("Extra['id_token'] = %q, want %q", g, w)
	}
	if tok.Expiry.IsZero() {
		t.Errorf("Expiry is zero; want ~1 hour")
	} else {
		exp := tok.Expiry.Sub(time.Now())
		const slop = 3 * time.Second // time moving during test
		if (time.Hour-slop) > exp || exp > time.Hour {
			t.Errorf("Expiry = %v, want ~1 hour", exp)
		}
	}
}

func checkBody(t *testing.T, r *http.Response, body string) {
	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		t.Errorf("reading reponse body: %v, want %q", err, body)
	}
	if g, w := string(b), body; g != w {
		t.Errorf("request body mismatch: got %q, want %q", g, w)
	}
}

func TestCachePermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		// Windows doesn't support file mode bits.
		return
	}

	td, err := ioutil.TempDir("", "oauth-test")
	if err != nil {
		t.Fatalf("ioutil.TempDir: %v", err)
	}
	defer os.RemoveAll(td)
	tempFile := filepath.Join(td, "cache-file")

	cf := CacheFile(tempFile)
	if err := cf.PutToken(new(Token)); err != nil {
		t.Fatalf("PutToken: %v", err)
	}
	fi, err := os.Stat(tempFile)
	if err != nil {
		t.Fatalf("os.Stat: %v", err)
	}
	if fi.Mode()&0077 != 0 {
		t.Errorf("Created cache file has mode %#o, want non-accessible to group+other", fi.Mode())
	}
}

func TestTokenExpired(t *testing.T) {
	tests := []struct {
		token   Token
		expired bool
	}{
		{Token{AccessToken: "foo"}, false},
		{Token{AccessToken: ""}, true},
		{Token{AccessToken: "foo", Expiry: time.Now().Add(-1 * time.Hour)}, true},
		{Token{AccessToken: "foo", Expiry: time.Now().Add(1 * time.Hour)}, false},
	}
	for _, tt := range tests {
		if got := tt.token.Expired(); got != tt.expired {
			t.Errorf("token %+v Expired = %v; want %v", tt.token, got, !got)
		}
	}
}
