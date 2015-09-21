package auth

import (
	"encoding/base64"
	"net/http"
	"testing"
)

func TestAuthBasic(t *testing.T) {
	secrets := HtpasswdFileProvider("test.htpasswd")
	a := &BasicAuth{Realm: "example.com", Secrets: secrets}
	r := &http.Request{}
	r.Method = "GET"
	if a.CheckAuth(r) != "" {
		t.Fatal("CheckAuth passed on empty headers")
	}
	r.Header = http.Header(make(map[string][]string))
	r.Header.Set("Authorization", "Digest blabla ololo")
	if a.CheckAuth(r) != "" {
		t.Fatal("CheckAuth passed on bad headers")
	}
	r.Header.Set("Authorization", "Basic !@#")
	if a.CheckAuth(r) != "" {
		t.Fatal("CheckAuth passed on bad base64 data")
	}

	data := [][]string{
		{"test", "hello"},
		{"test2", "hello2"},
	}
	for _, tc := range data {
		auth := base64.StdEncoding.EncodeToString([]byte(tc[0] + ":" + tc[1]))
		r.Header.Set("Authorization", "Basic "+auth)
		if a.CheckAuth(r) != tc[0] {
			t.Fatalf("CheckAuth failed for user '%s'", tc[0])
		}
	}
}
