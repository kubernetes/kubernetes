// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package google provides support for making OAuth2 authorized and authenticated
// HTTP requests to Google APIs. It supports the Web server flow, client-side
// credentials, service accounts, Google Compute Engine service accounts, and Google
// App Engine service accounts.
//
// A brief overview of the package follows. For more information, please read
// https://developers.google.com/accounts/docs/OAuth2
// and
// https://developers.google.com/accounts/docs/application-default-credentials.
//
// OAuth2 Configs
//
// Two functions in this package return golang.org/x/oauth2.Config values from Google credential
// data. Google supports two JSON formats for OAuth2 credentials: one is handled by ConfigFromJSON,
// the other by JWTConfigFromJSON. The returned Config can be used to obtain a TokenSource or
// create an http.Client.
//
//
// Credentials
//
// The Credentials type represents Google credentials, including Application Default
// Credentials.
//
// Use FindDefaultCredentials to obtain Application Default Credentials.
// FindDefaultCredentials looks in some well-known places for a credentials file, and
// will call AppEngineTokenSource or ComputeTokenSource as needed.
//
// DefaultClient and DefaultTokenSource are convenience methods. They first call FindDefaultCredentials,
// then use the credentials to construct an http.Client or an oauth2.TokenSource.
//
// Use CredentialsFromJSON to obtain credentials from either of the two JSON formats
// described in OAuth2 Configs, above. The TokenSource in the returned value is the
// same as the one obtained from the oauth2.Config returned from ConfigFromJSON or
// JWTConfigFromJSON, but the Credentials may contain additional information
// that is useful is some circumstances.
package google // import "golang.org/x/oauth2/google"
