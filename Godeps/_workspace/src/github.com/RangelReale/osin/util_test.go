package osin

import (
	"net/http"
	"net/url"
	"testing"
)

const (
	badAuthValue  = "Digest XHHHHHHH"
	goodAuthValue = "Basic dGVzdDp0ZXN0"
)

func TestBasicAuth(t *testing.T) {
	r := &http.Request{Header: make(http.Header)}

	// Without any header
	if b, err := CheckBasicAuth(r); b != nil || err != nil {
		t.Errorf("Validated basic auth without header")
	}

	// with invalid header
	r.Header.Set("Authorization", badAuthValue)
	b, err := CheckBasicAuth(r)
	if b != nil || err == nil {
		t.Errorf("Validated invalid auth")
		return
	}

	// with valid header
	r.Header.Set("Authorization", goodAuthValue)
	b, err = CheckBasicAuth(r)
	if b == nil || err != nil {
		t.Errorf("Could not extract basic auth")
		return
	}

	// check extracted auth data
	if b.Username != "test" || b.Password != "test" {
		t.Errorf("Error decoding basic auth")
	}
}

func TestGetClientAuth(t *testing.T) {

	urlWithSecret, _ := url.Parse("http://host.tld/path?client_id=xxx&client_secret=yyy")
	urlWithEmptySecret, _ := url.Parse("http://host.tld/path?client_id=xxx&client_secret=")
	urlNoSecret, _ := url.Parse("http://host.tld/path?client_id=xxx")

	headerNoAuth := make(http.Header)
	headerBadAuth := make(http.Header)
	headerBadAuth.Set("Authorization", badAuthValue)
	headerOKAuth := make(http.Header)
	headerOKAuth.Set("Authorization", goodAuthValue)

	var tests = []struct {
		header           http.Header
		url              *url.URL
		allowQueryParams bool
		expectAuth       bool
	}{
		{headerNoAuth, urlWithSecret, true, true},
		{headerNoAuth, urlWithSecret, false, false},
		{headerNoAuth, urlWithEmptySecret, true, true},
		{headerNoAuth, urlWithEmptySecret, false, false},
		{headerNoAuth, urlNoSecret, true, false},
		{headerNoAuth, urlNoSecret, false, false},

		{headerBadAuth, urlWithSecret, true, true},
		{headerBadAuth, urlWithSecret, false, false},
		{headerBadAuth, urlWithEmptySecret, true, true},
		{headerBadAuth, urlWithEmptySecret, false, false},
		{headerBadAuth, urlNoSecret, true, false},
		{headerBadAuth, urlNoSecret, false, false},

		{headerOKAuth, urlWithSecret, true, true},
		{headerOKAuth, urlWithSecret, false, true},
		{headerOKAuth, urlWithEmptySecret, true, true},
		{headerOKAuth, urlWithEmptySecret, false, true},
		{headerOKAuth, urlNoSecret, true, true},
		{headerOKAuth, urlNoSecret, false, true},
	}

	for _, tt := range tests {
		w := new(Response)
		r := &http.Request{Header: tt.header, URL: tt.url}
		r.ParseForm()
		auth := getClientAuth(w, r, tt.allowQueryParams)
		if tt.expectAuth && auth == nil {
			t.Errorf("Auth should not be nil for %v", tt)
		} else if !tt.expectAuth && auth != nil {
			t.Errorf("Auth should be nil for %v", tt)
		}
	}

}
