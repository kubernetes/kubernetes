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

	"golang.org/x/oauth2/google"
)

// Creds returns credential information obtained from DialSettings, or if none, then
// it returns default credential information.
func Creds(ctx context.Context, ds *DialSettings) (*google.Credentials, error) {
	if ds.Credentials != nil {
		return ds.Credentials, nil
	}
	if ds.CredentialsJSON != nil {
		return credentialsFromJSON(ctx, ds.CredentialsJSON, ds.Endpoint, ds.Scopes, ds.Audiences)
	}
	if ds.CredentialsFile != "" {
		data, err := ioutil.ReadFile(ds.CredentialsFile)
		if err != nil {
			return nil, fmt.Errorf("cannot read credentials file: %v", err)
		}
		return credentialsFromJSON(ctx, data, ds.Endpoint, ds.Scopes, ds.Audiences)
	}
	if ds.TokenSource != nil {
		return &google.Credentials{TokenSource: ds.TokenSource}, nil
	}
	cred, err := google.FindDefaultCredentials(ctx, ds.Scopes...)
	if err != nil {
		return nil, err
	}
	if len(cred.JSON) > 0 {
		return credentialsFromJSON(ctx, cred.JSON, ds.Endpoint, ds.Scopes, ds.Audiences)
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
// - If the JSON is a service account and no scopes provided, returns self-signed JWT auth flow
// - Otherwise, returns OAuth 2.0 flow.
func credentialsFromJSON(ctx context.Context, data []byte, endpoint string, scopes []string, audiences []string) (*google.Credentials, error) {
	cred, err := google.CredentialsFromJSON(ctx, data, scopes...)
	if err != nil {
		return nil, err
	}
	if len(data) > 0 && len(scopes) == 0 {
		var f struct {
			Type string `json:"type"`
			// The rest JSON fields are omitted because they are not used.
		}
		if err := json.Unmarshal(cred.JSON, &f); err != nil {
			return nil, err
		}
		if f.Type == serviceAccountKey {
			ts, err := selfSignedJWTTokenSource(data, endpoint, audiences)
			if err != nil {
				return nil, err
			}
			cred.TokenSource = ts
		}
	}
	return cred, err
}

func selfSignedJWTTokenSource(data []byte, endpoint string, audiences []string) (oauth2.TokenSource, error) {
	// Use the API endpoint as the default audience
	audience := endpoint
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
