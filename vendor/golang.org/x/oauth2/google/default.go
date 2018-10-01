// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"runtime"

	"cloud.google.com/go/compute/metadata"
	"golang.org/x/net/context"
	"golang.org/x/oauth2"
)

// DefaultClient returns an HTTP Client that uses the
// DefaultTokenSource to obtain authentication credentials.
func DefaultClient(ctx context.Context, scope ...string) (*http.Client, error) {
	ts, err := DefaultTokenSource(ctx, scope...)
	if err != nil {
		return nil, err
	}
	return oauth2.NewClient(ctx, ts), nil
}

// DefaultTokenSource returns the token source for
// "Application Default Credentials".
// It is a shortcut for FindDefaultCredentials(ctx, scope).TokenSource.
func DefaultTokenSource(ctx context.Context, scope ...string) (oauth2.TokenSource, error) {
	creds, err := FindDefaultCredentials(ctx, scope...)
	if err != nil {
		return nil, err
	}
	return creds.TokenSource, nil
}

// Common implementation for FindDefaultCredentials.
func findDefaultCredentials(ctx context.Context, scopes []string) (*DefaultCredentials, error) {
	// First, try the environment variable.
	const envVar = "GOOGLE_APPLICATION_CREDENTIALS"
	if filename := os.Getenv(envVar); filename != "" {
		creds, err := readCredentialsFile(ctx, filename, scopes)
		if err != nil {
			return nil, fmt.Errorf("google: error getting credentials using %v environment variable: %v", envVar, err)
		}
		return creds, nil
	}

	// Second, try a well-known file.
	filename := wellKnownFile()
	if creds, err := readCredentialsFile(ctx, filename, scopes); err == nil {
		return creds, nil
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("google: error getting credentials using well-known file (%v): %v", filename, err)
	}

	// Third, if we're on Google App Engine use those credentials.
	if appengineTokenFunc != nil && !appengineFlex {
		return &DefaultCredentials{
			ProjectID:   appengineAppIDFunc(ctx),
			TokenSource: AppEngineTokenSource(ctx, scopes...),
		}, nil
	}

	// Fourth, if we're on Google Compute Engine use the metadata server.
	if metadata.OnGCE() {
		id, _ := metadata.ProjectID()
		return &DefaultCredentials{
			ProjectID:   id,
			TokenSource: ComputeTokenSource(""),
		}, nil
	}

	// None are found; return helpful error.
	const url = "https://developers.google.com/accounts/docs/application-default-credentials"
	return nil, fmt.Errorf("google: could not find default credentials. See %v for more information.", url)
}

// Common implementation for CredentialsFromJSON.
func credentialsFromJSON(ctx context.Context, jsonData []byte, scopes []string) (*DefaultCredentials, error) {
	var f credentialsFile
	if err := json.Unmarshal(jsonData, &f); err != nil {
		return nil, err
	}
	ts, err := f.tokenSource(ctx, append([]string(nil), scopes...))
	if err != nil {
		return nil, err
	}
	return &DefaultCredentials{
		ProjectID:   f.ProjectID,
		TokenSource: ts,
		JSON:        jsonData,
	}, nil
}

func wellKnownFile() string {
	const f = "application_default_credentials.json"
	if runtime.GOOS == "windows" {
		return filepath.Join(os.Getenv("APPDATA"), "gcloud", f)
	}
	return filepath.Join(guessUnixHomeDir(), ".config", "gcloud", f)
}

func readCredentialsFile(ctx context.Context, filename string, scopes []string) (*DefaultCredentials, error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return CredentialsFromJSON(ctx, b, scopes...)
}
