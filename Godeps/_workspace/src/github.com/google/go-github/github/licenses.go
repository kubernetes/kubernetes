// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "fmt"

// LicensesService handles communication with the license related
// methods of the GitHub API.
//
// GitHub API docs: http://developer.github.com/v3/pulls/
type LicensesService struct {
	client *Client
}

// License represents an open source license.
type License struct {
	Key  *string `json:"key,omitempty"`
	Name *string `json:"name,omitempty"`
	URL  *string `json:"url,omitempty"`

	HTMLURL        *string   `json:"html_url,omitempty"`
	Featured       *bool     `json:"featured,omitempty"`
	Description    *string   `json:"description,omitempty"`
	Category       *string   `json:"category,omitempty"`
	Implementation *string   `json:"implementation,omitempty"`
	Required       *[]string `json:"required,omitempty"`
	Permitted      *[]string `json:"permitted,omitempty"`
	Forbidden      *[]string `json:"forbidden,omitempty"`
	Body           *string   `json:"body,omitempty"`
}

func (l License) String() string {
	return Stringify(l)
}

// List popular open source licenses.
//
// GitHub API docs: https://developer.github.com/v3/licenses/#list-all-licenses
func (s *LicensesService) List() ([]License, *Response, error) {
	req, err := s.client.NewRequest("GET", "licenses", nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches
	req.Header.Set("Accept", mediaTypeLicensesPreview)

	licenses := new([]License)
	resp, err := s.client.Do(req, licenses)
	if err != nil {
		return nil, resp, err
	}

	return *licenses, resp, err
}

// Get extended metadata for one license.
//
// GitHub API docs: https://developer.github.com/v3/licenses/#get-an-individual-license
func (s *LicensesService) Get(licenseName string) (*License, *Response, error) {
	u := fmt.Sprintf("licenses/%s", licenseName)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches
	req.Header.Set("Accept", mediaTypeLicensesPreview)

	license := new(License)
	resp, err := s.client.Do(req, license)
	if err != nil {
		return nil, resp, err
	}

	return license, resp, err
}
