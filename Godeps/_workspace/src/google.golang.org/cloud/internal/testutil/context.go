// Copyright 2014 Google Inc. All Rights Reserved.
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
	"io/ioutil"
	"log"
	"os"

	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

const (
	envProjID     = "GCLOUD_TESTS_GOLANG_PROJECT_ID"
	envPrivateKey = "GCLOUD_TESTS_GOLANG_KEY"
)

// ProjID returns the project ID to use in integration tests, or the empty
// string if none is configured.
func ProjID() string {
	projID := os.Getenv(envProjID)
	if projID == "" {
		return ""
	}
	return projID
}

// TokenSource returns the OAuth2 token source to use in integration tests,
// or nil if none is configured. TokenSource will log.Fatal if the token
// source is specified but missing or invalid.
func TokenSource(ctx context.Context, scopes ...string) oauth2.TokenSource {
	key := os.Getenv(envPrivateKey)
	if key == "" {
		return nil
	}
	jsonKey, err := ioutil.ReadFile(key)
	if err != nil {
		log.Fatalf("Cannot read the JSON key file, err: %v", err)
	}
	conf, err := google.JWTConfigFromJSON(jsonKey, scopes...)
	if err != nil {
		log.Fatalf("google.JWTConfigFromJSON: %v", err)
	}
	return conf.TokenSource(ctx)
}
