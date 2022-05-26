// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// LicensesService handles communication with the license related
// methods of the GitHub API.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/licenses/
type LicensesService service

// RepositoryLicense represents the license for a repository.
type RepositoryLicense struct {
	Name *string `json:"name,omitempty"`
	Path *string `json:"path,omitempty"`

	SHA         *string  `json:"sha,omitempty"`
	Size        *int     `json:"size,omitempty"`
	URL         *string  `json:"url,omitempty"`
	HTMLURL     *string  `json:"html_url,omitempty"`
	GitURL      *string  `json:"git_url,omitempty"`
	DownloadURL *string  `json:"download_url,omitempty"`
	Type        *string  `json:"type,omitempty"`
	Content     *string  `json:"content,omitempty"`
	Encoding    *string  `json:"encoding,omitempty"`
	License     *License `json:"license,omitempty"`
}

func (l RepositoryLicense) String() string {
	return Stringify(l)
}

// License represents an open source license.
type License struct {
	Key  *string `json:"key,omitempty"`
	Name *string `json:"name,omitempty"`
	URL  *string `json:"url,omitempty"`

	SPDXID         *string   `json:"spdx_id,omitempty"`
	HTMLURL        *string   `json:"html_url,omitempty"`
	Featured       *bool     `json:"featured,omitempty"`
	Description    *string   `json:"description,omitempty"`
	Implementation *string   `json:"implementation,omitempty"`
	Permissions    *[]string `json:"permissions,omitempty"`
	Conditions     *[]string `json:"conditions,omitempty"`
	Limitations    *[]string `json:"limitations,omitempty"`
	Body           *string   `json:"body,omitempty"`
}

func (l License) String() string {
	return Stringify(l)
}

// List popular open source licenses.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/licenses/#list-all-licenses
func (s *LicensesService) List(ctx context.Context) ([]*License, *Response, error) {
	req, err := s.client.NewRequest("GET", "licenses", nil)
	if err != nil {
		return nil, nil, err
	}

	var licenses []*License
	resp, err := s.client.Do(ctx, req, &licenses)
	if err != nil {
		return nil, resp, err
	}

	return licenses, resp, nil
}

// Get extended metadata for one license.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/licenses/#get-a-license
func (s *LicensesService) Get(ctx context.Context, licenseName string) (*License, *Response, error) {
	u := fmt.Sprintf("licenses/%s", licenseName)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	license := new(License)
	resp, err := s.client.Do(ctx, req, license)
	if err != nil {
		return nil, resp, err
	}

	return license, resp, nil
}
