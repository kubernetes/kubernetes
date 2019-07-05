/*
Copyright 2017 The Kubernetes Authors.

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

package union

import (
	"context"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/authentication/authenticator"
)

// unionAuthTokenHandler authenticates tokens using a chain of authenticator.Token objects
type unionAuthTokenHandler struct {
	// Handlers is a chain of request authenticators to delegate to
	Handlers []authenticator.Token
	// FailOnError determines whether an error returns short-circuits the chain
	FailOnError bool
}

// New returns a token authenticator that validates credentials using a chain of authenticator.Token objects.
// The entire chain is tried until one succeeds. If all fail, an aggregate error is returned.
func New(authTokenHandlers ...authenticator.Token) authenticator.Token {
	if len(authTokenHandlers) == 1 {
		return authTokenHandlers[0]
	}
	return &unionAuthTokenHandler{Handlers: authTokenHandlers, FailOnError: false}
}

// NewFailOnError returns a token authenticator that validates credentials using a chain of authenticator.Token objects.
// The first error short-circuits the chain.
func NewFailOnError(authTokenHandlers ...authenticator.Token) authenticator.Token {
	if len(authTokenHandlers) == 1 {
		return authTokenHandlers[0]
	}
	return &unionAuthTokenHandler{Handlers: authTokenHandlers, FailOnError: true}
}

// AuthenticateToken authenticates the token using a chain of authenticator.Token objects.
func (authHandler *unionAuthTokenHandler) AuthenticateToken(ctx context.Context, token string) (*authenticator.Response, bool, error) {
	var errlist []error
	for _, currAuthRequestHandler := range authHandler.Handlers {
		info, ok, err := currAuthRequestHandler.AuthenticateToken(ctx, token)
		if err != nil {
			if authHandler.FailOnError {
				return info, ok, err
			}
			errlist = append(errlist, err)
			continue
		}

		if ok {
			return info, ok, err
		}
	}

	return nil, false, utilerrors.NewAggregate(errlist)
}
