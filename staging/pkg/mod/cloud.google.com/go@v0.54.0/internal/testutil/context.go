// Copyright 2014 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package testutil contains helper functions for writing tests.
package testutil

import (
	"context"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"golang.org/x/oauth2/jwt"
)

const (
	envProjID     = "GCLOUD_TESTS_GOLANG_PROJECT_ID"
	envPrivateKey = "GCLOUD_TESTS_GOLANG_KEY"
)

// ProjID returns the project ID to use in integration tests, or the empty
// string if none is configured.
func ProjID() string {
	return os.Getenv(envProjID)
}

// Credentials returns the credentials to use in integration tests, or nil if
// none is configured. It uses the standard environment variable for tests in
// this repo.
func Credentials(ctx context.Context, scopes ...string) *google.Credentials {
	return CredentialsEnv(ctx, envPrivateKey, scopes...)
}

// CredentialsEnv returns the credentials to use in integration tests, or nil
// if none is configured. If the environment variable is unset, CredentialsEnv
// will try to find 'Application Default Credentials'. Else, CredentialsEnv
// will return nil. CredentialsEnv will log.Fatal if the token source is
// specified but missing or invalid.
func CredentialsEnv(ctx context.Context, envVar string, scopes ...string) *google.Credentials {
	key := os.Getenv(envVar)
	if key == "" { // Try for application default credentials.
		creds, err := google.FindDefaultCredentials(ctx, scopes...)
		if err != nil {
			log.Println("No 'Application Default Credentials' found.")
			return nil
		}
		return creds
	}

	data, err := ioutil.ReadFile(key)
	if err != nil {
		log.Fatal(err)
	}

	creds, err := google.CredentialsFromJSON(ctx, data, scopes...)
	if err != nil {
		log.Fatal(err)
	}
	return creds
}

// TokenSource returns the OAuth2 token source to use in integration tests,
// or nil if none is configured. It uses the standard environment variable
// for tests in this repo.
func TokenSource(ctx context.Context, scopes ...string) oauth2.TokenSource {
	return TokenSourceEnv(ctx, envPrivateKey, scopes...)
}

// TokenSourceEnv returns the OAuth2 token source to use in integration tests. or nil
// if none is configured. It tries to get credentials from the filename in the
// environment variable envVar. If the environment variable is unset, TokenSourceEnv
// will try to find 'Application Default Credentials'. Else, TokenSourceEnv will
// return nil. TokenSourceEnv will log.Fatal if the token source is specified but
// missing or invalid.
func TokenSourceEnv(ctx context.Context, envVar string, scopes ...string) oauth2.TokenSource {
	key := os.Getenv(envVar)
	if key == "" { // Try for application default credentials.
		ts, err := google.DefaultTokenSource(ctx, scopes...)
		if err != nil {
			log.Println("No 'Application Default Credentials' found.")
			return nil
		}
		return ts
	}
	conf, err := jwtConfigFromFile(key, scopes)
	if err != nil {
		log.Fatal(err)
	}
	return conf.TokenSource(ctx)
}

// JWTConfig reads the JSON private key file whose name is in the default
// environment variable, and returns the jwt.Config it contains. It ignores
// scopes.
// If the environment variable is empty, it returns (nil, nil).
func JWTConfig() (*jwt.Config, error) {
	return jwtConfigFromFile(os.Getenv(envPrivateKey), nil)
}

// jwtConfigFromFile reads the given JSON private key file, and returns the
// jwt.Config it contains.
// If the filename is empty, it returns (nil, nil).
func jwtConfigFromFile(filename string, scopes []string) (*jwt.Config, error) {
	if filename == "" {
		return nil, nil
	}
	jsonKey, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("cannot read the JSON key file, err: %v", err)
	}
	conf, err := google.JWTConfigFromJSON(jsonKey, scopes...)
	if err != nil {
		return nil, fmt.Errorf("google.JWTConfigFromJSON: %v", err)
	}
	return conf, nil
}

// CanReplay reports whether an integration test can be run in replay mode.
// The replay file must exist, and the GCLOUD_TESTS_GOLANG_ENABLE_REPLAY
// environment variable must be non-empty.
func CanReplay(replayFilename string) bool {
	if os.Getenv("GCLOUD_TESTS_GOLANG_ENABLE_REPLAY") == "" {
		return false
	}
	_, err := os.Stat(replayFilename)
	return err == nil
}

// ErroringTokenSource is a token source for testing purposes,
// to always return a non-nil error to its caller. It is useful
// when testing error responses with bad oauth2 credentials.
type ErroringTokenSource struct{}

// Token implements oauth2.TokenSource, returning a nil oauth2.Token and a non-nil error.
func (fts ErroringTokenSource) Token() (*oauth2.Token, error) {
	return nil, errors.New("intentional error")
}
