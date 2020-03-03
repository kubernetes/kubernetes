/*
Copyright 2020 The Kubernetes Authors.

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

package instrumentedauthenticator

import (
	"context"
	"net/http"
	"time"

	"k8s.io/apiserver/pkg/authentication/authenticator"
)

// WrapRequest wraps the input Request in the instrumentedRequest.
func WrapRequest(delegate authenticator.Request, authenticatorName string) authenticator.Request {
	return authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
		authenticationStart := time.Now()
		resp, ok, err := delegate.AuthenticateRequest(req)
		recordAuthMetrics(req.Context(), resp, authenticatorName, ok, err, authenticationStart)
		return resp, ok, err
	})
}

// WrapToken wraps the input Token in the instrumentedToken.
func WrapToken(delegate authenticator.Token, authenticatorName string) authenticator.Token {
	return authenticator.TokenFunc(func(ctx context.Context, token string) (*authenticator.Response, bool, error) {
		authenticationStart := time.Now()
		resp, ok, err := delegate.AuthenticateToken(ctx, token)
		recordAuthMetrics(ctx, resp, authenticatorName, ok, err, authenticationStart)
		return resp, ok, err
	})
}
