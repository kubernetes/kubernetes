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

func TestTransportNilTokenSource(t *testing.T) {
	tr := &Transport{}
	server := newMockServer(func(w http.ResponseWriter, r *http.Request) {})
	defer server.Close()
	client := &http.Client{Transport: tr}
	resp, err := client.Get(server.URL)
	if err == nil {
		t.Errorf("got no errors, want an error with nil token source")
	}
	if resp != nil {
		t.Errorf("Response = %v; want nil", resp)
	}
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
		if got, want := r.Header.Get("Authorization"), "Bearer abc"; got != want {
			t.Errorf("Authorization header = %q; want %q", got, want)
		}
	})
	defer server.Close()
	client := &http.Client{Transport: tr}
	res, err := client.Get(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
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
		client := &http.Client{Transport: tr}
		res, err := client.Get(server.URL)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
	}
}

func TestTokenValidNoAccessToken(t *testing.T) {
	token := &Token{}
	if token.Valid() {
		t.Errorf("got valid with no access token; want invalid")
	}
}

func TestExpiredWithExpiry(t *testing.T) {
	token := &Token{
		Expiry: time.Now().Add(-5 * time.Hour),
	}
	if token.Valid() {
		t.Errorf("got valid with expired token; want invalid")
	}
}

func newMockServer(handler func(w http.ResponseWriter, r *http.Request)) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(handler))
}
