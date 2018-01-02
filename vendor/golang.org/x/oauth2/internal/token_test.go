// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package internal contains support packages for oauth2 package.
package internal

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"golang.org/x/net/context"
)

func TestRegisterBrokenAuthHeaderProvider(t *testing.T) {
	RegisterBrokenAuthHeaderProvider("https://aaa.com/")
	tokenURL := "https://aaa.com/token"
	if providerAuthHeaderWorks(tokenURL) {
		t.Errorf("got %q as unbroken; want broken", tokenURL)
	}
}

func TestRetrieveTokenBustedNoSecret(t *testing.T) {
	const clientID = "client-id"

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.FormValue("client_id"), clientID; got != want {
			t.Errorf("client_id = %q; want %q", got, want)
		}
		if got, want := r.FormValue("client_secret"), ""; got != want {
			t.Errorf("client_secret = %q; want empty", got)
		}
	}))
	defer ts.Close()

	RegisterBrokenAuthHeaderProvider(ts.URL)
	_, err := RetrieveToken(context.Background(), clientID, "", ts.URL, url.Values{})
	if err != nil {
		t.Errorf("RetrieveToken = %v; want no error", err)
	}
}

func Test_providerAuthHeaderWorks(t *testing.T) {
	for _, p := range brokenAuthHeaderProviders {
		if providerAuthHeaderWorks(p) {
			t.Errorf("got %q as unbroken; want broken", p)
		}
		p := fmt.Sprintf("%ssomesuffix", p)
		if providerAuthHeaderWorks(p) {
			t.Errorf("got %q as unbroken; want broken", p)
		}
	}
	p := "https://api.not-in-the-list-example.com/"
	if !providerAuthHeaderWorks(p) {
		t.Errorf("got %q as unbroken; want broken", p)
	}
}

func TestProviderAuthHeaderWorksDomain(t *testing.T) {
	tests := []struct {
		tokenURL  string
		wantWorks bool
	}{
		{"https://dev-12345.okta.com/token-url", false},
		{"https://dev-12345.oktapreview.com/token-url", false},
		{"https://dev-12345.okta.org/token-url", true},
		{"https://foo.bar.force.com/token-url", false},
		{"https://foo.force.com/token-url", false},
		{"https://force.com/token-url", true},
	}

	for _, test := range tests {
		got := providerAuthHeaderWorks(test.tokenURL)
		if got != test.wantWorks {
			t.Errorf("providerAuthHeaderWorks(%q) = %v; want %v", test.tokenURL, got, test.wantWorks)
		}
	}
}
