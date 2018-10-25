// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.9

package google

import (
	"golang.org/x/net/context"
	"golang.org/x/oauth2"
)

// DefaultCredentials holds Google credentials, including "Application Default Credentials".
// For more details, see:
// https://developers.google.com/accounts/docs/application-default-credentials
type DefaultCredentials struct {
	ProjectID   string // may be empty
	TokenSource oauth2.TokenSource

	// JSON contains the raw bytes from a JSON credentials file.
	// This field may be nil if authentication is provided by the
	// environment and not with a credentials file, e.g. when code is
	// running on Google Cloud Platform.
	JSON []byte
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
//   3. On Google App Engine it uses the appengine.AccessToken function.
//   4. On Google Compute Engine and Google App Engine Managed VMs, it fetches
//      credentials from the metadata server.
//      (In this final case any provided scopes are ignored.)
func FindDefaultCredentials(ctx context.Context, scopes ...string) (*DefaultCredentials, error) {
	return findDefaultCredentials(ctx, scopes)
}

// CredentialsFromJSON obtains Google credentials from a JSON value. The JSON can
// represent either a Google Developers Console client_credentials.json file (as in
// ConfigFromJSON) or a Google Developers service account key file (as in
// JWTConfigFromJSON).
//
// Note: despite the name, the returned credentials may not be Application Default Credentials.
func CredentialsFromJSON(ctx context.Context, jsonData []byte, scopes ...string) (*DefaultCredentials, error) {
	return credentialsFromJSON(ctx, jsonData, scopes)
}
