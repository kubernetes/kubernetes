// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"runtime"

	"cloud.google.com/go/compute/metadata"
	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/jwt"
)

// DefaultClient returns an HTTP Client that uses the
// DefaultTokenSource to obtain authentication credentials.
//
// This client should be used when developing services
// that run on Google App Engine or Google Compute Engine
// and use "Application Default Credentials."
//
// For more details, see:
// https://developers.google.com/accounts/docs/application-default-credentials
//
func DefaultClient(ctx context.Context, scope ...string) (*http.Client, error) {
	ts, err := DefaultTokenSource(ctx, scope...)
	if err != nil {
		return nil, err
	}
	return oauth2.NewClient(ctx, ts), nil
}

// DefaultTokenSource is a token source that uses
// "Application Default Credentials".
//
// It looks for credentials in the following places,
// preferring the first location found:
//
//   1. A JSON file whose path is specified by the
//      GOOGLE_APPLICATION_CREDENTIALS environment variable.
//   2. A JSON file in a location known to the gcloud command-line tool.
//      On Windows, this is %APPDATA%/gcloud/application_default_credentials.json.
//      On other systems, $HOME/.config/gcloud/application_default_credentials.json.
//   3. On Google App Engine it uses the appengine.AccessToken function.
//   4. On Google Compute Engine and Google App Engine Managed VMs, it fetches
//      credentials from the metadata server.
//      (In this final case any provided scopes are ignored.)
//
// For more details, see:
// https://developers.google.com/accounts/docs/application-default-credentials
//
func DefaultTokenSource(ctx context.Context, scope ...string) (oauth2.TokenSource, error) {
	// First, try the environment variable.
	const envVar = "GOOGLE_APPLICATION_CREDENTIALS"
	if filename := os.Getenv(envVar); filename != "" {
		ts, err := tokenSourceFromFile(ctx, filename, scope)
		if err != nil {
			return nil, fmt.Errorf("google: error getting credentials using %v environment variable: %v", envVar, err)
		}
		return ts, nil
	}

	// Second, try a well-known file.
	filename := wellKnownFile()
	_, err := os.Stat(filename)
	if err == nil {
		ts, err2 := tokenSourceFromFile(ctx, filename, scope)
		if err2 == nil {
			return ts, nil
		}
		err = err2
	} else if os.IsNotExist(err) {
		err = nil // ignore this error
	}
	if err != nil {
		return nil, fmt.Errorf("google: error getting credentials using well-known file (%v): %v", filename, err)
	}

	// Third, if we're on Google App Engine use those credentials.
	if appengineTokenFunc != nil && !appengineVM {
		return AppEngineTokenSource(ctx, scope...), nil
	}

	// Fourth, if we're on Google Compute Engine use the metadata server.
	if metadata.OnGCE() {
		return ComputeTokenSource(""), nil
	}

	// None are found; return helpful error.
	const url = "https://developers.google.com/accounts/docs/application-default-credentials"
	return nil, fmt.Errorf("google: could not find default credentials. See %v for more information.", url)
}

func wellKnownFile() string {
	const f = "application_default_credentials.json"
	if runtime.GOOS == "windows" {
		return filepath.Join(os.Getenv("APPDATA"), "gcloud", f)
	}
	return filepath.Join(guessUnixHomeDir(), ".config", "gcloud", f)
}

func tokenSourceFromFile(ctx context.Context, filename string, scopes []string) (oauth2.TokenSource, error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	var d struct {
		// Common fields
		Type     string
		ClientID string `json:"client_id"`

		// User Credential fields
		ClientSecret string `json:"client_secret"`
		RefreshToken string `json:"refresh_token"`

		// Service Account fields
		ClientEmail  string `json:"client_email"`
		PrivateKeyID string `json:"private_key_id"`
		PrivateKey   string `json:"private_key"`
	}
	if err := json.Unmarshal(b, &d); err != nil {
		return nil, err
	}
	switch d.Type {
	case "authorized_user":
		cfg := &oauth2.Config{
			ClientID:     d.ClientID,
			ClientSecret: d.ClientSecret,
			Scopes:       append([]string{}, scopes...), // copy
			Endpoint:     Endpoint,
		}
		tok := &oauth2.Token{RefreshToken: d.RefreshToken}
		return cfg.TokenSource(ctx, tok), nil
	case "service_account":
		cfg := &jwt.Config{
			Email:      d.ClientEmail,
			PrivateKey: []byte(d.PrivateKey),
			Scopes:     append([]string{}, scopes...), // copy
			TokenURL:   JWTTokenURL,
		}
		return cfg.TokenSource(ctx), nil
	case "":
		return nil, errors.New("missing 'type' field in credentials")
	default:
		return nil, fmt.Errorf("unknown credential type: %q", d.Type)
	}
}
