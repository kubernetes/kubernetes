// Copyright 2019 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

const (
	mediaTypeAppManifestPreview = "application/vnd.github.fury-preview+json"
)

// AppConfig describes the configuration of a GitHub App.
type AppConfig struct {
	ID            *int64     `json:"id,omitempty"`
	NodeID        *string    `json:"node_id,omitempty"`
	Owner         *User      `json:"owner,omitempty"`
	Name          *string    `json:"name,omitempty"`
	Description   *string    `json:"description,omitempty"`
	ExternalURL   *string    `json:"external_url,omitempty"`
	HTMLURL       *string    `json:"html_url,omitempty"`
	CreatedAt     *Timestamp `json:"created_at,omitempty"`
	UpdatedAt     *Timestamp `json:"updated_at,omitempty"`
	ClientID      *string    `json:"client_id,omitempty"`
	ClientSecret  *string    `json:"client_secret,omitempty"`
	WebhookSecret *string    `json:"webhook_secret,omitempty"`
	PEM           *string    `json:"pem,omitempty"`
}

// CompleteAppManifest completes the App manifest handshake flow for the given
// code.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps/#create-a-github-app-from-a-manifest
func (s *AppsService) CompleteAppManifest(ctx context.Context, code string) (*AppConfig, *Response, error) {
	u := fmt.Sprintf("app-manifests/%s/conversions", code)
	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Accept", mediaTypeAppManifestPreview)

	cfg := new(AppConfig)
	resp, err := s.client.Do(ctx, req, cfg)
	if err != nil {
		return nil, resp, err
	}

	return cfg, resp, nil
}
