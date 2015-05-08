// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "fmt"

// GitignoresService provides access to the gitignore related functions in the
// GitHub API.
//
// GitHub API docs: http://developer.github.com/v3/gitignore/
type GitignoresService struct {
	client *Client
}

// Gitignore represents a .gitignore file as returned by the GitHub API.
type Gitignore struct {
	Name   *string `json:"name,omitempty"`
	Source *string `json:"source,omitempty"`
}

func (g Gitignore) String() string {
	return Stringify(g)
}

// List all available Gitignore templates.
//
// http://developer.github.com/v3/gitignore/#listing-available-templates
func (s GitignoresService) List() ([]string, *Response, error) {
	req, err := s.client.NewRequest("GET", "gitignore/templates", nil)
	if err != nil {
		return nil, nil, err
	}

	availableTemplates := new([]string)
	resp, err := s.client.Do(req, availableTemplates)
	if err != nil {
		return nil, resp, err
	}

	return *availableTemplates, resp, err
}

// Get a Gitignore by name.
//
// http://developer.github.com/v3/gitignore/#get-a-single-template
func (s GitignoresService) Get(name string) (*Gitignore, *Response, error) {
	u := fmt.Sprintf("gitignore/templates/%v", name)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	gitignore := new(Gitignore)
	resp, err := s.client.Do(req, gitignore)
	if err != nil {
		return nil, resp, err
	}

	return gitignore, resp, err
}
