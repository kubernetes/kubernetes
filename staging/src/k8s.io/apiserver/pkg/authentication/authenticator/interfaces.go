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
	"context"
	"fmt"
	"net/http"

	"k8s.io/apiserver/pkg/authentication/user"
)

// Token checks a string value against a backing authentication store and
// returns a Response or an error if the token could not be checked.
type Token interface {
	AuthenticateToken(ctx context.Context, token string) (*Response, bool, error)
}

// Request attempts to extract authentication information from a request and
// returns a Response or an error if the request could not be checked.
type Request interface {
	AuthenticateRequest(req *http.Request) (*Response, bool, error)
}

// Password checks a username and password against a backing authentication
// store and returns a Response or an error if the password could not be
// checked.
type Password interface {
	AuthenticatePassword(ctx context.Context, user, password string) (*Response, bool, error)
}

// TokenFunc is a function that implements the Token interface.
type TokenFunc func(ctx context.Context, token string) (*Response, bool, error)

// AuthenticateToken implements authenticator.Token.
func (f TokenFunc) AuthenticateToken(ctx context.Context, token string) (*Response, bool, error) {
	return f(ctx, token)
}

// RequestFunc is a function that implements the Request interface.
type RequestFunc func(req *http.Request) (*Response, bool, error)

// AuthenticateRequest implements authenticator.Request.
func (f RequestFunc) AuthenticateRequest(req *http.Request) (*Response, bool, error) {
	return f(req)
}

// PasswordFunc is a function that implements the Password interface.
type PasswordFunc func(ctx context.Context, user, password string) (*Response, bool, error)

// AuthenticatePassword implements authenticator.Password.
func (f PasswordFunc) AuthenticatePassword(ctx context.Context, user, password string) (*Response, bool, error) {
	return f(ctx, user, password)
}

// Response is the struct returned by authenticator interfaces upon successful
// authentication. It contains information about whether the authenticator
// authenticated the request, information about the context of the
// authentication, and information about the authenticated user.
type Response struct {
	// Audiences is the set of audiences the authenticator was able to validate
	// the token against. If the authenticator is not audience aware, this field
	// will be empty.
	Audiences Audiences
	// User is the UserInfo associated with the authentication context.
	User user.Info
	// AuthMethod identifies the method used to authenticate the request (ex. token or x509).
	// This field is intended for grouping (via a label) authentication metrics, thus allowing
	// aggregation of metrics based on the authentication method.
	// It is intended to be visible to to metrics consumers only, and should remain stable
	// over time. Therefore, it does not need to be human readable.
	// Its value should not affect the result of authentication in any way.
	// This field should neither be logged nor passed to the authorization layer.
	AuthMethod string
	// AuthenticatorName identifies the authenticator that authenticated the request.
	// This field is intended for grouping (via a label) authentication metrics, thus allowing
	// aggregation of metrics based on the authenticator.
	// It is intended to be visible to to metrics consumers only, and should remain stable
	// over time. Therefore, it does not need to be human readable.
	// Its value should not affect the result of authentication in any way.
	// This field should neither be logged nor passed to the authorization layer.
	AuthenticatorName string
}

// Error is the error returned by an authenticator upon failures.
type Error struct {
	// Err is the error produced by an authenticator.
	Err error
	// AuthMethod identifies the method used to authenticate the request (ex. token or x509).
	AuthMethod string
	// AuthenticatorName identifies the authenticator that failed an authentication request.
	AuthenticatorName string
}

// Error implements the errors.Error interface.
func (a Error) Error() string {
	return fmt.Sprintf("authentcation method: %q by authenticator: %q failed to process authentication request: %v", a.AuthMethod, a.AuthenticatorName, a.Err)
}

// Unwrap unpacks the wrapped error.
func (a Error) Unwrap() error {
	return a.Err
}

// NewError constructs Error.
func NewError(authMethod, authenticatorName string, err error) error {
	return &Error{AuthenticatorName: authenticatorName, AuthMethod: authMethod, Err: err}
}

// Errorf constructs Error.
func Errorf(authMethod, authenticatorName, format string, a ...interface{}) error {
	return &Error{AuthenticatorName: authenticatorName, AuthMethod: authMethod, Err: fmt.Errorf(format, a...)}
}
