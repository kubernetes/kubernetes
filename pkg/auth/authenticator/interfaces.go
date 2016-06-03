/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package authenticator

import (
	"net/http"

	"k8s.io/kubernetes/pkg/auth/user"
)

// Token checks a string value against a backing authentication store and returns
// information about the current user and true if successful, false if not successful,
// or an error if the token could not be checked.
type Token interface {
	AuthenticateToken(token string) (user.Info, bool, error)
}

// Request attempts to extract authentication information from a request and returns
// information about the current user and true if successful, false if not successful,
// or an error if the request could not be checked.
type Request interface {
	AuthenticateRequest(req *http.Request) (user.Info, bool, error)
}

// Password checks a username and password against a backing authentication store and
// returns information about the user and true if successful, false if not successful,
// or an error if the username and password could not be checked
type Password interface {
	AuthenticatePassword(user, password string) (user.Info, bool, error)
}

// TokenFunc is a function that implements the Token interface.
type TokenFunc func(token string) (user.Info, bool, error)

// AuthenticateToken implements authenticator.Token.
func (f TokenFunc) AuthenticateToken(token string) (user.Info, bool, error) {
	return f(token)
}

// RequestFunc is a function that implements the Request interface.
type RequestFunc func(req *http.Request) (user.Info, bool, error)

// AuthenticateRequest implements authenticator.Request.
func (f RequestFunc) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	return f(req)
}

// PasswordFunc is a function that implements the Password interface.
type PasswordFunc func(user, password string) (user.Info, bool, error)

// AuthenticatePassword implements authenticator.Password.
func (f PasswordFunc) AuthenticatePassword(user, password string) (user.Info, bool, error) {
	return f(user, password)
}
