package oidc

import (
	"fmt"
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/go-oidc/jose"
)

func TestCookieTokenExtractorInvalid(t *testing.T) {
	ckName := "tokenCookie"
	tests := []*http.Cookie{
		&http.Cookie{},
		&http.Cookie{Name: ckName},
		&http.Cookie{Name: ckName, Value: ""},
	}

	for i, tt := range tests {
		r, _ := http.NewRequest("", "", nil)
		r.AddCookie(tt)
		_, err := CookieTokenExtractor(ckName)(r)
		if err == nil {
			t.Errorf("case %d: want: error for invalid cookie token, got: no error.", i)
		}
	}
}

func TestCookieTokenExtractorValid(t *testing.T) {
	validToken := "eyJ0eXAiOiJKV1QiLA0KICJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJqb2UiLA0KICJleHAiOjEzMDA4MTkzODAsDQogImh0dHA6Ly9leGFtcGxlLmNvbS9pc19yb290Ijp0cnVlfQ.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
	ckName := "tokenCookie"
	tests := []*http.Cookie{
		&http.Cookie{Name: ckName, Value: "some non-empty value"},
		&http.Cookie{Name: ckName, Value: validToken},
	}

	for i, tt := range tests {
		r, _ := http.NewRequest("", "", nil)
		r.AddCookie(tt)
		_, err := CookieTokenExtractor(ckName)(r)
		if err != nil {
			t.Errorf("case %d: want: valid cookie with no error, got: %v", i, err)
		}
	}
}

func TestExtractBearerTokenInvalid(t *testing.T) {
	tests := []string{"", "x", "Bearer", "xxxxxxx", "Bearer "}

	for i, tt := range tests {
		r, _ := http.NewRequest("", "", nil)
		r.Header.Add("Authorization", tt)
		_, err := ExtractBearerToken(r)
		if err == nil {
			t.Errorf("case %d: want: invalid Authorization header, got: valid Authorization header.", i)
		}
	}
}

func TestExtractBearerTokenValid(t *testing.T) {
	validToken := "eyJ0eXAiOiJKV1QiLA0KICJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJqb2UiLA0KICJleHAiOjEzMDA4MTkzODAsDQogImh0dHA6Ly9leGFtcGxlLmNvbS9pc19yb290Ijp0cnVlfQ.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
	tests := []string{
		fmt.Sprintf("Bearer %s", validToken),
	}

	for i, tt := range tests {
		r, _ := http.NewRequest("", "", nil)
		r.Header.Add("Authorization", tt)
		_, err := ExtractBearerToken(r)
		if err != nil {
			t.Errorf("case %d: want: valid Authorization header, got: invalid Authorization header: %v.", i, err)
		}
	}
}

func TestNewClaims(t *testing.T) {
	issAt := time.Date(2, time.January, 1, 0, 0, 0, 0, time.UTC)
	expAt := time.Date(2, time.January, 1, 1, 0, 0, 0, time.UTC)

	want := jose.Claims{
		"iss": "https://example.com",
		"sub": "user-123",
		"aud": "client-abc",
		"iat": issAt.Unix(),
		"exp": expAt.Unix(),
	}

	got := NewClaims("https://example.com", "user-123", "client-abc", issAt, expAt)

	if !reflect.DeepEqual(want, got) {
		t.Fatalf("want=%#v got=%#v", want, got)
	}

	want2 := jose.Claims{
		"iss": "https://example.com",
		"sub": "user-123",
		"aud": []string{"client-abc", "client-def"},
		"iat": issAt.Unix(),
		"exp": expAt.Unix(),
	}

	got2 := NewClaims("https://example.com", "user-123", []string{"client-abc", "client-def"}, issAt, expAt)

	if !reflect.DeepEqual(want2, got2) {
		t.Fatalf("want=%#v got=%#v", want2, got2)
	}

}
