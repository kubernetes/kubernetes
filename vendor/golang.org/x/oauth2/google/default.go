// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"runtime"

	"cloud.google.com/go/compute/metadata"
	"golang.org/x/oauth2"
)

// Credentials holds Google credentials, including "Application Default Credentials".
// For more details, see:
// https://developers.google.com/accounts/docs/application-default-credentials
type Credentials struct {
	ProjectID   string // may be empty
	TokenSource oauth2.TokenSource

	// JSON contains the raw bytes from a JSON credentials file.
	// This field may be nil if authentication is provided by the
	// environment and not with a credentials file, e.g. when code is
	// running on Google Cloud Platform.
	JSON []byte
}

// DefaultCredentials is the old name of Credentials.
//
// Deprecated: use Credentials instead.
type DefaultCredentials = Credentials

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

// FindDefaultCredentials searches for "Application Default Credentials".
//
// It looks for credentials in the following places,
// preferring the first location found:
//
//   1. A JSON file whose path is specified by the
//      GOOGLE_APPLICATION_CREDENTIALS environment variable.
//   2. A JSON file in a location known to the gcloud command-line tool.
//      On Windows, this is %APPDATA%/gcloud/application_default_credentials.json.
//      On other systems, $HOME/.config/gcloud/application_default_credentials.json.
//   3. On Google App Engine standard first generation runtimes (<= Go 1.9) it uses
//      the appengine.AccessToken function.
//   4. On Google Compute Engine, Google App Engine standard second generation runtimes
//      (>= Go 1.11), and Google App Engine flexible environment, it fetches
//      credentials from the metadata server.
func FindDefaultCredentials(ctx context.Context, scopes ...string) (*Credentials, error) {
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

	// Third, if we're on a Google App Engine standard first generation runtime (<= Go 1.9)
	// use those credentials. App Engine standard second generation runtimes (>= Go 1.11)
	// and App Engine flexible use ComputeTokenSource and the metadata server.
	if appengineTokenFunc != nil {
		return &DefaultCredentials{
			ProjectID:   appengineAppIDFunc(ctx),
			TokenSource: AppEngineTokenSource(ctx, scopes...),
		}, nil
	}

	// Fourth, if we're on Google Compute Engine, an App Engine standard second generation runtime,
	// or App Engine flexible, use the metadata server.
	if metadata.OnGCE() {
		id, _ := metadata.ProjectID()
		return &DefaultCredentials{
			ProjectID:   id,
			TokenSource: ComputeTokenSource("", scopes...),
		}, nil
	}

	// None are found; return helpful error.
	const url = "https://developers.google.com/accounts/docs/application-default-credentials"
	return nil, fmt.Errorf("google: could not find default credentials. See %v for more information.", url)
}

// CredentialsFromJSON obtains Google credentials from a JSON value. The JSON can
// represent either a Google Developers Console client_credentials.json file (as in
// ConfigFromJSON) or a Google Developers service account key file (as in
// JWTConfigFromJSON).
func CredentialsFromJSON(ctx context.Context, jsonData []byte, scopes ...string) (*Credentials, error) {
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
