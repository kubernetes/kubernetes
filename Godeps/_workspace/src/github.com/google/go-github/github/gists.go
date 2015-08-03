// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// GistsService handles communication with the Gist related
// methods of the GitHub API.
//
// GitHub API docs: http://developer.github.com/v3/gists/
type GistsService struct {
	client *Client
}

// Gist represents a GitHub's gist.
type Gist struct {
	ID          *string                   `json:"id,omitempty"`
	Description *string                   `json:"description,omitempty"`
	Public      *bool                     `json:"public,omitempty"`
	Owner       *User                     `json:"owner,omitempty"`
	Files       map[GistFilename]GistFile `json:"files,omitempty"`
	Comments    *int                      `json:"comments,omitempty"`
	HTMLURL     *string                   `json:"html_url,omitempty"`
	GitPullURL  *string                   `json:"git_pull_url,omitempty"`
	GitPushURL  *string                   `json:"git_push_url,omitempty"`
	CreatedAt   *time.Time                `json:"created_at,omitempty"`
	UpdatedAt   *time.Time                `json:"updated_at,omitempty"`
}

func (g Gist) String() string {
	return Stringify(g)
}

// GistFilename represents filename on a gist.
type GistFilename string

// GistFile represents a file on a gist.
type GistFile struct {
	Size     *int    `json:"size,omitempty"`
	Filename *string `json:"filename,omitempty"`
	RawURL   *string `json:"raw_url,omitempty"`
	Content  *string `json:"content,omitempty"`
}

func (g GistFile) String() string {
	return Stringify(g)
}

// GistListOptions specifies the optional parameters to the
// GistsService.List, GistsService.ListAll, and GistsService.ListStarred methods.
type GistListOptions struct {
	// Since filters Gists by time.
	Since time.Time `url:"since,omitempty"`

	ListOptions
}

// List gists for a user. Passing the empty string will list
// all public gists if called anonymously. However, if the call
// is authenticated, it will returns all gists for the authenticated
// user.
//
// GitHub API docs: http://developer.github.com/v3/gists/#list-gists
func (s *GistsService) List(user string, opt *GistListOptions) ([]Gist, *Response, error) {
	var u string
	if user != "" {
		u = fmt.Sprintf("users/%v/gists", user)
	} else {
		u = "gists"
	}
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	gists := new([]Gist)
	resp, err := s.client.Do(req, gists)
	if err != nil {
		return nil, resp, err
	}

	return *gists, resp, err
}

// ListAll lists all public gists.
//
// GitHub API docs: http://developer.github.com/v3/gists/#list-gists
func (s *GistsService) ListAll(opt *GistListOptions) ([]Gist, *Response, error) {
	u, err := addOptions("gists/public", opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	gists := new([]Gist)
	resp, err := s.client.Do(req, gists)
	if err != nil {
		return nil, resp, err
	}

	return *gists, resp, err
}

// ListStarred lists starred gists of authenticated user.
//
// GitHub API docs: http://developer.github.com/v3/gists/#list-gists
func (s *GistsService) ListStarred(opt *GistListOptions) ([]Gist, *Response, error) {
	u, err := addOptions("gists/starred", opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	gists := new([]Gist)
	resp, err := s.client.Do(req, gists)
	if err != nil {
		return nil, resp, err
	}

	return *gists, resp, err
}

// Get a single gist.
//
// GitHub API docs: http://developer.github.com/v3/gists/#get-a-single-gist
func (s *GistsService) Get(id string) (*Gist, *Response, error) {
	u := fmt.Sprintf("gists/%v", id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}
	gist := new(Gist)
	resp, err := s.client.Do(req, gist)
	if err != nil {
		return nil, resp, err
	}

	return gist, resp, err
}

// GetRevision gets a specific revision of a gist.
//
// GitHub API docs: https://developer.github.com/v3/gists/#get-a-specific-revision-of-a-gist
func (s *GistsService) GetRevision(id, sha string) (*Gist, *Response, error) {
	u := fmt.Sprintf("gists/%v/%v", id, sha)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}
	gist := new(Gist)
	resp, err := s.client.Do(req, gist)
	if err != nil {
		return nil, resp, err
	}

	return gist, resp, err
}

// Create a gist for authenticated user.
//
// GitHub API docs: http://developer.github.com/v3/gists/#create-a-gist
func (s *GistsService) Create(gist *Gist) (*Gist, *Response, error) {
	u := "gists"
	req, err := s.client.NewRequest("POST", u, gist)
	if err != nil {
		return nil, nil, err
	}
	g := new(Gist)
	resp, err := s.client.Do(req, g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, err
}

// Edit a gist.
//
// GitHub API docs: http://developer.github.com/v3/gists/#edit-a-gist
func (s *GistsService) Edit(id string, gist *Gist) (*Gist, *Response, error) {
	u := fmt.Sprintf("gists/%v", id)
	req, err := s.client.NewRequest("PATCH", u, gist)
	if err != nil {
		return nil, nil, err
	}
	g := new(Gist)
	resp, err := s.client.Do(req, g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, err
}

// Delete a gist.
//
// GitHub API docs: http://developer.github.com/v3/gists/#delete-a-gist
func (s *GistsService) Delete(id string) (*Response, error) {
	u := fmt.Sprintf("gists/%v", id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// Star a gist on behalf of authenticated user.
//
// GitHub API docs: http://developer.github.com/v3/gists/#star-a-gist
func (s *GistsService) Star(id string) (*Response, error) {
	u := fmt.Sprintf("gists/%v/star", id)
	req, err := s.client.NewRequest("PUT", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// Unstar a gist on a behalf of authenticated user.
//
// Github API docs: http://developer.github.com/v3/gists/#unstar-a-gist
func (s *GistsService) Unstar(id string) (*Response, error) {
	u := fmt.Sprintf("gists/%v/star", id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// IsStarred checks if a gist is starred by authenticated user.
//
// GitHub API docs: http://developer.github.com/v3/gists/#check-if-a-gist-is-starred
func (s *GistsService) IsStarred(id string) (bool, *Response, error) {
	u := fmt.Sprintf("gists/%v/star", id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return false, nil, err
	}
	resp, err := s.client.Do(req, nil)
	starred, err := parseBoolResponse(err)
	return starred, resp, err
}

// Fork a gist.
//
// GitHub API docs: http://developer.github.com/v3/gists/#fork-a-gist
func (s *GistsService) Fork(id string) (*Gist, *Response, error) {
	u := fmt.Sprintf("gists/%v/forks", id)
	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, nil, err
	}

	g := new(Gist)
	resp, err := s.client.Do(req, g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, err
}
