// Copyright 2017 Google Inc. All Rights Reserved.
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

package internal

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

// Creds returns credential information obtained from DialSettings, or if none, then
// it returns default credential information.
func Creds(ctx context.Context, ds *DialSettings) (*google.DefaultCredentials, error) {
	if ds.CredentialsFile != "" {
		return credFileTokenSource(ctx, ds.CredentialsFile, ds.Scopes...)
	}
	if ds.TokenSource != nil {
		return &google.DefaultCredentials{TokenSource: ds.TokenSource}, nil
	}
	return google.FindDefaultCredentials(ctx, ds.Scopes...)
}

// credFileTokenSource reads a refresh token file or a service account and returns
// a TokenSource constructed from the config.
func credFileTokenSource(ctx context.Context, filename string, scope ...string) (*google.DefaultCredentials, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("cannot read credentials file: %v", err)
	}
	// See if it is a refresh token credentials file first.
	ts, ok, err := refreshTokenTokenSource(ctx, data, scope...)
	if err != nil {
		return nil, err
	}
	if ok {
		return &google.DefaultCredentials{
			TokenSource: ts,
			JSON:        data,
		}, nil
	}

	// If not, it should be a service account.
	cfg, err := google.JWTConfigFromJSON(data, scope...)
	if err != nil {
		return nil, fmt.Errorf("google.JWTConfigFromJSON: %v", err)
	}
	// jwt.Config does not expose the project ID, so re-unmarshal to get it.
	var pid struct {
		ProjectID string `json:"project_id"`
	}
	if err := json.Unmarshal(data, &pid); err != nil {
		return nil, err
	}
	return &google.DefaultCredentials{
		ProjectID:   pid.ProjectID,
		TokenSource: cfg.TokenSource(ctx),
		JSON:        data,
	}, nil
}

func refreshTokenTokenSource(ctx context.Context, data []byte, scope ...string) (oauth2.TokenSource, bool, error) {
	var c cred
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, false, fmt.Errorf("cannot unmarshal credentials file: %v", err)
	}
	if c.ClientID == "" || c.ClientSecret == "" || c.RefreshToken == "" || c.Type != "authorized_user" {
		return nil, false, nil
	}
	cfg := &oauth2.Config{
		ClientID:     c.ClientID,
		ClientSecret: c.ClientSecret,
		Endpoint:     google.Endpoint,
		RedirectURL:  "urn:ietf:wg:oauth:2.0:oob",
		Scopes:       scope,
	}
	return cfg.TokenSource(ctx, &oauth2.Token{
		RefreshToken: c.RefreshToken,
		Expiry:       time.Now(),
	}), true, nil
}

type cred struct {
	ClientID     string `json:"client_id"`
	ClientSecret string `json:"client_secret"`
	RefreshToken string `json:"refresh_token"`
	Type         string `json:"type"`
}
