// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package security

import (
	"net/http"
	"strings"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/runtime"
)

// httpAuthenticator is a function that authenticates a HTTP request
func httpAuthenticator(handler func(*http.Request) (bool, interface{}, error)) runtime.Authenticator {
	return runtime.AuthenticatorFunc(func(params interface{}) (bool, interface{}, error) {
		if request, ok := params.(*http.Request); ok {
			return handler(request)
		}
		if scoped, ok := params.(*ScopedAuthRequest); ok {
			return handler(scoped.Request)
		}
		return false, nil, nil
	})
}

func scopedAuthenticator(handler func(*ScopedAuthRequest) (bool, interface{}, error)) runtime.Authenticator {
	return runtime.AuthenticatorFunc(func(params interface{}) (bool, interface{}, error) {
		if request, ok := params.(*ScopedAuthRequest); ok {
			return handler(request)
		}
		return false, nil, nil
	})
}

// UserPassAuthentication authentication function
type UserPassAuthentication func(string, string) (interface{}, error)

// TokenAuthentication authentication function
type TokenAuthentication func(string) (interface{}, error)

// ScopedTokenAuthentication authentication function
type ScopedTokenAuthentication func(string, []string) (interface{}, error)

// BasicAuth creates a basic auth authenticator with the provided authentication function
func BasicAuth(authenticate UserPassAuthentication) runtime.Authenticator {
	return httpAuthenticator(func(r *http.Request) (bool, interface{}, error) {
		if usr, pass, ok := r.BasicAuth(); ok {
			p, err := authenticate(usr, pass)
			return true, p, err
		}

		return false, nil, nil
	})
}

// APIKeyAuth creates an authenticator that uses a token for authorization.
// This token can be obtained from either a header or a query string
func APIKeyAuth(name, in string, authenticate TokenAuthentication) runtime.Authenticator {
	inl := strings.ToLower(in)
	if inl != "query" && inl != "header" {
		// panic because this is most likely a typo
		panic(errors.New(500, "api key auth: in value needs to be either \"query\" or \"header\"."))
	}

	var getToken func(*http.Request) string
	switch inl {
	case "header":
		getToken = func(r *http.Request) string { return r.Header.Get(name) }
	case "query":
		getToken = func(r *http.Request) string { return r.URL.Query().Get(name) }
	}

	return httpAuthenticator(func(r *http.Request) (bool, interface{}, error) {
		token := getToken(r)
		if token == "" {
			return false, nil, nil
		}

		p, err := authenticate(token)
		return true, p, err
	})
}

// ScopedAuthRequest contains both a http request and the required scopes for a particular operation
type ScopedAuthRequest struct {
	Request        *http.Request
	RequiredScopes []string
}

// BearerAuth for use with oauth2 flows
func BearerAuth(name string, authenticate ScopedTokenAuthentication) runtime.Authenticator {
	const prefix = "Bearer "
	return scopedAuthenticator(func(r *ScopedAuthRequest) (bool, interface{}, error) {
		var token string
		hdr := r.Request.Header.Get("Authorization")
		if strings.HasPrefix(hdr, prefix) {
			token = strings.TrimPrefix(hdr, prefix)
		}
		if token == "" {
			qs := r.Request.URL.Query()
			token = qs.Get("access_token")
		}
		ct, _, _ := runtime.ContentType(r.Request.Header)
		if token == "" && (ct == "application/x-www-form-urlencoded" || ct == "multipart/form-data") {
			token = r.Request.FormValue("access_token")
		}

		if token == "" {
			return false, nil, nil
		}

		p, err := authenticate(token, r.RequiredScopes)
		return true, p, err
	})
}
