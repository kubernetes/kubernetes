package oauth2

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

type tokenSource struct{ token *Token }

func (t *tokenSource) Token() (*Token, error) {
	return t.token, nil
}

func TestTransportTokenSource(t *testing.T) {
	ts := &tokenSource{
		token: &Token{
			AccessToken: "abc",
		},
	}
	tr := &Transport{
		Source: ts,
	}
	server := newMockServer(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer abc" {
			t.Errorf("Transport doesn't set the Authorization header from the fetched token")
		}
	})
	defer server.Close()
	client := http.Client{Transport: tr}
	client.Get(server.URL)
}

// Test for case-sensitive token types, per https://github.com/golang/oauth2/issues/113
func TestTransportTokenSourceTypes(t *testing.T) {
	const val = "abc"
	tests := []struct {
		key  string
		val  string
		want string
	}{
		{key: "bearer", val: val, want: "Bearer abc"},
		{key: "mac", val: val, want: "MAC abc"},
		{key: "basic", val: val, want: "Basic abc"},
	}
	for _, tc := range tests {
		ts := &tokenSource{
			token: &Token{
				AccessToken: tc.val,
				TokenType:   tc.key,
			},
		}
		tr := &Transport{
			Source: ts,
		}
		server := newMockServer(func(w http.ResponseWriter, r *http.Request) {
			if got, want := r.Header.Get("Authorization"), tc.want; got != want {
				t.Errorf("Authorization header (%q) = %q; want %q", val, got, want)
			}
		})
		defer server.Close()
		client := http.Client{Transport: tr}
		client.Get(server.URL)
	}
}

func TestTokenValidNoAccessToken(t *testing.T) {
	token := &Token{}
	if token.Valid() {
		t.Errorf("Token should not be valid with no access token")
	}
}

func TestExpiredWithExpiry(t *testing.T) {
	token := &Token{
		Expiry: time.Now().Add(-5 * time.Hour),
	}
	if token.Valid() {
		t.Errorf("Token should not be valid if it expired in the past")
	}
}

func newMockServer(handler func(w http.ResponseWriter, r *http.Request)) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(handler))
}
