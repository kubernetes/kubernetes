package oidc

import (
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/coreos/go-oidc/jose"
)

// RequestTokenExtractor funcs extract a raw encoded token from a request.
type RequestTokenExtractor func(r *http.Request) (string, error)

// ExtractBearerToken is a RequestTokenExtractor which extracts a bearer token from a request's
// Authorization header.
func ExtractBearerToken(r *http.Request) (string, error) {
	ah := r.Header.Get("Authorization")
	if ah == "" {
		return "", errors.New("missing Authorization header")
	}

	if len(ah) <= 6 || strings.ToUpper(ah[0:6]) != "BEARER" {
		return "", errors.New("should be a bearer token")
	}

	val := ah[7:]
	if len(val) == 0 {
		return "", errors.New("bearer token is empty")
	}

	return val, nil
}

// CookieTokenExtractor returns a RequestTokenExtractor which extracts a token from the named cookie in a request.
func CookieTokenExtractor(cookieName string) RequestTokenExtractor {
	return func(r *http.Request) (string, error) {
		ck, err := r.Cookie(cookieName)
		if err != nil {
			return "", fmt.Errorf("token cookie not found in request: %v", err)
		}

		if ck.Value == "" {
			return "", errors.New("token cookie found but is empty")
		}

		return ck.Value, nil
	}
}

func NewClaims(iss, sub, aud string, iat, exp time.Time) jose.Claims {
	return jose.Claims{
		// required
		"iss": iss,
		"sub": sub,
		"aud": aud,
		"iat": float64(iat.Unix()),
		"exp": float64(exp.Unix()),
	}
}

func GenClientID(hostport string) (string, error) {
	b, err := randBytes(32)
	if err != nil {
		return "", err
	}

	var host string
	if strings.Contains(hostport, ":") {
		host, _, err = net.SplitHostPort(hostport)
		if err != nil {
			return "", err
		}
	} else {
		host = hostport
	}

	return fmt.Sprintf("%s@%s", base64.URLEncoding.EncodeToString(b), host), nil
}

func randBytes(n int) ([]byte, error) {
	b := make([]byte, n)
	got, err := rand.Read(b)
	if err != nil {
		return nil, err
	} else if n != got {
		return nil, errors.New("unable to generate enough random data")
	}
	return b, nil
}

// urlEqual checks two urls for equality using only the host and path portions.
func urlEqual(url1, url2 string) bool {
	u1, err := url.Parse(url1)
	if err != nil {
		return false
	}
	u2, err := url.Parse(url2)
	if err != nil {
		return false
	}

	return strings.ToLower(u1.Host+u1.Path) == strings.ToLower(u2.Host+u2.Path)
}
