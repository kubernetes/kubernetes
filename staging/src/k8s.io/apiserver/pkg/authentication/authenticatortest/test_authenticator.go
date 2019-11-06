/*
Copyright 2019 The Kubernetes Authors.

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

// Package authenticatortest provides implementations of various authenticators for testing purposes.
package authenticatortest

import (
	"context"
	"net/http"

	"k8s.io/apiserver/pkg/authentication/authenticator"
)

// RequestAuthenticator implements authenticator.Request for testing purposes only.
type RequestAuthenticator struct {
	authFunc func(req *http.Request) (*authenticator.Response, bool, error)
}

// NewRequestAuth constructs RequestAuthenticator.
func NewRequestAuth(auth func(req *http.Request) (*authenticator.Response, bool, error)) *RequestAuthenticator {
	return &RequestAuthenticator{authFunc: auth}
}

// AuthenticateRequest fakes AuthenticateRequest method.
func (a *RequestAuthenticator) AuthenticateRequest(req *http.Request) (*authenticator.Response, bool, error) {
	return a.authFunc(req)
}

// AuthenticatorID fakes AuthenticatorID method.
func (a *RequestAuthenticator) AuthenticatorID() string {
	return "testAuthenticator"
}

// TokenAuthenticator implements authenticator.Token for testing purposes only.
type TokenAuthenticator struct {
	authFunc func(ctx context.Context, token string) (*authenticator.Response, bool, error)
}

// NewTokenAuth constructs TokenAuthenticator method.
func NewTokenAuth(auth func(ctx context.Context, token string) (*authenticator.Response, bool, error)) *TokenAuthenticator {
	return &TokenAuthenticator{authFunc: auth}
}

// AuthenticatorID fakes AuthenticatorID method.
func (a *TokenAuthenticator) AuthenticatorID() string {
	return "test-token-auth"
}

// AuthenticateToken fakes AuthenticateToken method.
func (a *TokenAuthenticator) AuthenticateToken(ctx context.Context, token string) (*authenticator.Response, bool, error) {
	return a.authFunc(ctx, token)
}
