// Copyright 2017 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"

	"golang.org/x/oauth2"
	"google.golang.org/api/internal/impersonate"

	"golang.org/x/oauth2/google"
)

// Creds returns credential information obtained from DialSettings, or if none, then
// it returns default credential information.
func Creds(ctx context.Context, ds *DialSettings) (*google.Credentials, error) {
	creds, err := baseCreds(ctx, ds)
	if err != nil {
		return nil, err
	}
	if ds.ImpersonationConfig != nil {
		return impersonateCredentials(ctx, creds, ds)
	}
	return creds, nil
}

func baseCreds(ctx context.Context, ds *DialSettings) (*google.Credentials, error) {
	if ds.Credentials != nil {
		return ds.Credentials, nil
	}
	if ds.CredentialsJSON != nil {
		return credentialsFromJSON(ctx, ds.CredentialsJSON, ds)
	}
	if ds.CredentialsFile != "" {
		data, err := ioutil.ReadFile(ds.CredentialsFile)
		if err != nil {
			return nil, fmt.Errorf("cannot read credentials file: %v", err)
		}
		return credentialsFromJSON(ctx, data, ds)
	}
	if ds.TokenSource != nil {
		return &google.Credentials{TokenSource: ds.TokenSource}, nil
	}
	cred, err := google.FindDefaultCredentials(ctx, ds.GetScopes()...)
	if err != nil {
		return nil, err
	}
	if len(cred.JSON) > 0 {
		return credentialsFromJSON(ctx, cred.JSON, ds)
	}
	// For GAE and GCE, the JSON is empty so return the default credentials directly.
	return cred, nil
}

// JSON key file type.
const (
	serviceAccountKey = "service_account"
)

// credentialsFromJSON returns a google.Credentials based on the input.
//
// - A self-signed JWT auth flow will be executed if: the data file is a service
//   account, no user are scopes provided, an audience is provided, a user
//   specified endpoint is not provided, and credentials will not be
//   impersonated.
//
// - Otherwise, executes a stanard OAuth 2.0 flow.
func credentialsFromJSON(ctx context.Context, data []byte, ds *DialSettings) (*google.Credentials, error) {
	cred, err := google.CredentialsFromJSON(ctx, data, ds.GetScopes()...)
	if err != nil {
		return nil, err
	}
	// Standard OAuth 2.0 Flow
	if len(data) == 0 ||
		len(ds.Scopes) > 0 ||
		(ds.DefaultAudience == "" && len(ds.Audiences) == 0) ||
		ds.ImpersonationConfig != nil ||
		ds.Endpoint != "" {
		return cred, nil
	}

	// Check if JSON is a service account and if so create a self-signed JWT.
	var f struct {
		Type string `json:"type"`
		// The rest JSON fields are omitted because they are not used.
	}
	if err := json.Unmarshal(cred.JSON, &f); err != nil {
		return nil, err
	}
	if f.Type == serviceAccountKey {
		ts, err := selfSignedJWTTokenSource(data, ds.DefaultAudience, ds.Audiences)
		if err != nil {
			return nil, err
		}
		cred.TokenSource = ts
	}
	return cred, err
}

func selfSignedJWTTokenSource(data []byte, defaultAudience string, audiences []string) (oauth2.TokenSource, error) {
	audience := defaultAudience
	if len(audiences) > 0 {
		// TODO(shinfan): Update golang oauth to support multiple audiences.
		if len(audiences) > 1 {
			return nil, fmt.Errorf("multiple audiences support is not implemented")
		}
		audience = audiences[0]
	}
	return google.JWTAccessTokenSourceFromJSON(data, audience)
}

// QuotaProjectFromCreds returns the quota project from the JSON blob in the provided credentials.
//
// NOTE(cbro): consider promoting this to a field on google.Credentials.
func QuotaProjectFromCreds(cred *google.Credentials) string {
	var v struct {
		QuotaProject string `json:"quota_project_id"`
	}
	if err := json.Unmarshal(cred.JSON, &v); err != nil {
		return ""
	}
	return v.QuotaProject
}

func impersonateCredentials(ctx context.Context, creds *google.Credentials, ds *DialSettings) (*google.Credentials, error) {
	if len(ds.ImpersonationConfig.Scopes) == 0 {
		ds.ImpersonationConfig.Scopes = ds.GetScopes()
	}
	ts, err := impersonate.TokenSource(ctx, creds.TokenSource, ds.ImpersonationConfig)
	if err != nil {
		return nil, err
	}
	return &google.Credentials{
		TokenSource: ts,
		ProjectID:   creds.ProjectID,
	}, nil
}
