// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package google provides support for making OAuth2 authorized and authenticated
// HTTP requests to Google APIs. It supports the Web server flow, client-side
// credentials, service accounts, Google Compute Engine service accounts,
// Google App Engine service accounts and workload identity federation
// from non-Google cloud platforms.
//
// A brief overview of the package follows. For more information, please read
// https://developers.google.com/accounts/docs/OAuth2
// and
// https://developers.google.com/accounts/docs/application-default-credentials.
// For more information on using workload identity federation, refer to
// https://cloud.google.com/iam/docs/how-to#using-workload-identity-federation.
//
// OAuth2 Configs
//
// Two functions in this package return golang.org/x/oauth2.Config values from Google credential
// data. Google supports two JSON formats for OAuth2 credentials: one is handled by ConfigFromJSON,
// the other by JWTConfigFromJSON. The returned Config can be used to obtain a TokenSource or
// create an http.Client.
//
// Workload Identity Federation
//
// Using workload identity federation, your application can access Google Cloud
// resources from Amazon Web Services (AWS), Microsoft Azure or any identity
// provider that supports OpenID Connect (OIDC).
// Traditionally, applications running outside Google Cloud have used service
// account keys to access Google Cloud resources. Using identity federation,
// you can allow your workload to impersonate a service account.
// This lets you access Google Cloud resources directly, eliminating the
// maintenance and security burden associated with service account keys.
//
// Follow the detailed instructions on how to configure Workload Identity Federation
// in various platforms:
//
//   Amazon Web Services (AWS): https://cloud.google.com/iam/docs/access-resources-aws
//   Microsoft Azure: https://cloud.google.com/iam/docs/access-resources-azure
//   OIDC identity provider: https://cloud.google.com/iam/docs/access-resources-oidc
//
// For OIDC providers, the library can retrieve OIDC tokens either from a
// local file location (file-sourced credentials) or from a local server
// (URL-sourced credentials).
// For file-sourced credentials, a background process needs to be continuously
// refreshing the file location with a new OIDC token prior to expiration.
// For tokens with one hour lifetimes, the token needs to be updated in the file
// every hour. The token can be stored directly as plain text or in JSON format.
// For URL-sourced credentials, a local server needs to host a GET endpoint to
// return the OIDC token. The response can be in plain text or JSON.
// Additional required request headers can also be specified.
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
// Application Default Credentials also support workload identity federation to
// access Google Cloud resources from non-Google Cloud platforms including Amazon
// Web Services (AWS), Microsoft Azure or any identity provider that supports
// OpenID Connect (OIDC). Workload identity federation is recommended for
// non-Google Cloud environments as it avoids the need to download, manage and
// store service account private keys locally.
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
