// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This package provides authentication utilities for Google Compute Engine (GCE)
package gce

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	"google.golang.org/cloud/compute/metadata"
)

const (
	metadataAuthScopes       = "instance/service-accounts/default/scopes"
	metadataAuthToken        = "instance/service-accounts/default/token"
	earlyRefreshTokenSeconds = 60
	waitForTokenInterval     = 1 * time.Second
	waitForTokenTimeout      = 30 * time.Second
)

// GCE specific representation of authorization token. For parsing purposes.
type authToken struct {
	// The actual token.
	AccessToken string `json:"access_token"`

	// Number of seconds in which the token will expire
	ExpiresIn int `json:"expires_in"`

	// Type of token.
	TokenType string `json:"token_type"`
}

type AuthTokenProvider interface {
	GetToken() (string, error)
	WaitForToken() (string, error)
}

// AuthTokenProvider is thread-safe.
type realAuthTokenProvider struct {
	sync.RWMutex

	// Token to use for authentication.
	token string
}

func NewAuthTokenProvider(expectedAuthScope string) (AuthTokenProvider, error) {
	// Retry OnGCE call for 15 seconds before declaring failure.
	onGCE := false
	for start := time.Now(); time.Since(start) < 15*time.Second; time.Sleep(time.Second) {
		if metadata.OnGCE() {
			onGCE = true
			break
		}
	}
	// Only support GCE for now.
	if !onGCE {
		return nil, fmt.Errorf("authorization to GCE is currently only supported on GCE")
	}

	// Check for required auth scopes
	err := verifyAuthScope(expectedAuthScope)
	if err != nil {
		return nil, err
	}

	t := &realAuthTokenProvider{
		RWMutex: sync.RWMutex{},
	}
	go t.refreshTokenWhenExpires()
	return t, nil
}

func (t *realAuthTokenProvider) GetToken() (string, error) {
	defer t.RUnlock()
	t.RLock()
	if t.token == "" {
		return "", fmt.Errorf("No valid GCE token")
	}
	return t.token, nil
}

func (t *realAuthTokenProvider) WaitForToken() (string, error) {
	for start := time.Now(); time.Since(start) < waitForTokenTimeout; time.Sleep(waitForTokenInterval) {
		if token, err := t.GetToken(); err == nil {
			return token, nil
		}
	}
	return "", fmt.Errorf("Timeout after %v while waiting for GCE token", waitForTokenTimeout)
}

// Get a token for performing GCE requests.
func getAuthToken() (authToken, error) {
	rawToken, err := metadata.Get(metadataAuthToken)
	if err != nil {
		return authToken{}, err
	}

	var token authToken
	err = json.Unmarshal([]byte(rawToken), &token)
	if err != nil {
		return authToken{}, fmt.Errorf("failed to unmarshal service account token with output %q: %v", rawToken, err)
	}

	return token, err
}

func (t *realAuthTokenProvider) refreshTokenWhenExpires() {
	for {
		token, err := getAuthToken()

		// If token was successfully obtained update local copy and wait until it will expire. Otherwise log error and try again.
		if err == nil {
			t.Lock()
			if token.ExpiresIn > earlyRefreshTokenSeconds {
				token.ExpiresIn -= earlyRefreshTokenSeconds
			}
			t.token = token.AccessToken
			t.Unlock()
			time.Sleep(time.Duration(token.ExpiresIn) * time.Second)
		} else {
			glog.Errorf("Error occured while refreshing GCE token: %v", err)
		}
	}
}

// Checks that the required auth scope is present.
func verifyAuthScope(expectedScope string) error {
	scopes, err := metadata.Get(metadataAuthScopes)
	if err != nil {
		return err
	}

	for _, scope := range strings.Fields(scopes) {
		if scope == expectedScope {
			return nil
		}
	}

	return fmt.Errorf("Current instance does not have the expected scope (%q). Actual scopes: %v", expectedScope, scopes)
}
