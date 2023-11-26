/*
 *
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package tokenmanager provides tokens for authenticating to S2A.
package tokenmanager

import (
	"fmt"
	"os"

	commonpb "github.com/google/s2a-go/internal/proto/common_go_proto"
)

const (
	s2aAccessTokenEnvironmentVariable = "S2A_ACCESS_TOKEN"
)

// AccessTokenManager manages tokens for authenticating to S2A.
type AccessTokenManager interface {
	// DefaultToken returns a token that an application with no specified local
	// identity must use to authenticate to S2A.
	DefaultToken() (token string, err error)
	// Token returns a token that an application with local identity equal to
	// identity must use to authenticate to S2A.
	Token(identity *commonpb.Identity) (token string, err error)
}

type singleTokenAccessTokenManager struct {
	token string
}

// NewSingleTokenAccessTokenManager returns a new AccessTokenManager instance
// that will always manage the same token.
//
// The token to be managed is read from the s2aAccessTokenEnvironmentVariable
// environment variable. If this environment variable is not set, then this
// function returns an error.
func NewSingleTokenAccessTokenManager() (AccessTokenManager, error) {
	token, variableExists := os.LookupEnv(s2aAccessTokenEnvironmentVariable)
	if !variableExists {
		return nil, fmt.Errorf("%s environment variable is not set", s2aAccessTokenEnvironmentVariable)
	}
	return &singleTokenAccessTokenManager{token: token}, nil
}

// DefaultToken always returns the token managed by the
// singleTokenAccessTokenManager.
func (m *singleTokenAccessTokenManager) DefaultToken() (string, error) {
	return m.token, nil
}

// Token always returns the token managed by the singleTokenAccessTokenManager.
func (m *singleTokenAccessTokenManager) Token(*commonpb.Identity) (string, error) {
	return m.token, nil
}
