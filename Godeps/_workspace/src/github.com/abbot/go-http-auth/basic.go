package auth

import (
	"crypto/sha1"
	"encoding/base64"
	"net/http"
	"strings"
)

type BasicAuth struct {
	Realm   string
	Secrets SecretProvider
}

/*
 Checks the username/password combination from the request. Returns
 either an empty string (authentication failed) or the name of the
 authenticated user.

 Supports MD5 and SHA1 password entries
*/
func (a *BasicAuth) CheckAuth(r *http.Request) string {
	s := strings.SplitN(r.Header.Get("Authorization"), " ", 2)
	if len(s) != 2 || s[0] != "Basic" {
		return ""
	}

	b, err := base64.StdEncoding.DecodeString(s[1])
	if err != nil {
		return ""
	}
	pair := strings.SplitN(string(b), ":", 2)
	if len(pair) != 2 {
		return ""
	}
	passwd := a.Secrets(pair[0], a.Realm)
	if passwd == "" {
		return ""
	}
	if strings.HasPrefix(passwd, "{SHA}") {
		d := sha1.New()
		d.Write([]byte(pair[1]))
		if passwd[5:] != base64.StdEncoding.EncodeToString(d.Sum(nil)) {
			return ""
		}
	} else {
		e := NewMD5Entry(passwd)
		if e == nil {
			return ""
		}
		if passwd != string(MD5Crypt([]byte(pair[1]), e.Salt, e.Magic)) {
			return ""
		}
	}
	return pair[0]
}

/*
 http.Handler for BasicAuth which initiates the authentication process
 (or requires reauthentication).
*/
func (a *BasicAuth) RequireAuth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("WWW-Authenticate", `Basic realm="`+a.Realm+`"`)
	w.WriteHeader(401)
	w.Write([]byte("401 Unauthorized\n"))
}

/*
 BasicAuthenticator returns a function, which wraps an
 AuthenticatedHandlerFunc converting it to http.HandlerFunc. This
 wrapper function checks the authentication and either sends back
 required authentication headers, or calls the wrapped function with
 authenticated username in the AuthenticatedRequest.
*/
func (a *BasicAuth) Wrap(wrapped AuthenticatedHandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if username := a.CheckAuth(r); username == "" {
			a.RequireAuth(w, r)
		} else {
			ar := &AuthenticatedRequest{Request: *r, Username: username}
			wrapped(w, ar)
		}
	}
}

func NewBasicAuthenticator(realm string, secrets SecretProvider) *BasicAuth {
	return &BasicAuth{Realm: realm, Secrets: secrets}
}
