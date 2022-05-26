// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
	"time"
)

// GistsService handles communication with the Gist related
// methods of the GitHub API.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/
type GistsService service

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
	NodeID      *string                   `json:"node_id,omitempty"`
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
	Language *string `json:"language,omitempty"`
	Type     *string `json:"type,omitempty"`
	RawURL   *string `json:"raw_url,omitempty"`
	Content  *string `json:"content,omitempty"`
}

func (g GistFile) String() string {
	return Stringify(g)
}

// GistCommit represents a commit on a gist.
type GistCommit struct {
	URL          *string      `json:"url,omitempty"`
	Version      *string      `json:"version,omitempty"`
	User         *User        `json:"user,omitempty"`
	ChangeStatus *CommitStats `json:"change_status,omitempty"`
	CommittedAt  *Timestamp   `json:"committed_at,omitempty"`
	NodeID       *string      `json:"node_id,omitempty"`
}

func (gc GistCommit) String() string {
	return Stringify(gc)
}

// GistFork represents a fork of a gist.
type GistFork struct {
	URL       *string    `json:"url,omitempty"`
	User      *User      `json:"user,omitempty"`
	ID        *string    `json:"id,omitempty"`
	CreatedAt *Timestamp `json:"created_at,omitempty"`
	UpdatedAt *Timestamp `json:"updated_at,omitempty"`
	NodeID    *string    `json:"node_id,omitempty"`
}

func (gf GistFork) String() string {
	return Stringify(gf)
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
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#list-gists-for-the-authenticated-user
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#list-gists-for-a-user
func (s *GistsService) List(ctx context.Context, user string, opts *GistListOptions) ([]*Gist, *Response, error) {
	var u string
	if user != "" {
		u = fmt.Sprintf("users/%v/gists", user)
	} else {
		u = "gists"
	}
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var gists []*Gist
	resp, err := s.client.Do(ctx, req, &gists)
	if err != nil {
		return nil, resp, err
	}

	return gists, resp, nil
}

// ListAll lists all public gists.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#list-public-gists
func (s *GistsService) ListAll(ctx context.Context, opts *GistListOptions) ([]*Gist, *Response, error) {
	u, err := addOptions("gists/public", opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var gists []*Gist
	resp, err := s.client.Do(ctx, req, &gists)
	if err != nil {
		return nil, resp, err
	}

	return gists, resp, nil
}

// ListStarred lists starred gists of authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#list-starred-gists
func (s *GistsService) ListStarred(ctx context.Context, opts *GistListOptions) ([]*Gist, *Response, error) {
	u, err := addOptions("gists/starred", opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var gists []*Gist
	resp, err := s.client.Do(ctx, req, &gists)
	if err != nil {
		return nil, resp, err
	}

	return gists, resp, nil
}

// Get a single gist.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#get-a-gist
func (s *GistsService) Get(ctx context.Context, id string) (*Gist, *Response, error) {
	u := fmt.Sprintf("gists/%v", id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	gist := new(Gist)
	resp, err := s.client.Do(ctx, req, gist)
	if err != nil {
		return nil, resp, err
	}

	return gist, resp, nil
}

// GetRevision gets a specific revision of a gist.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#get-a-gist-revision
func (s *GistsService) GetRevision(ctx context.Context, id, sha string) (*Gist, *Response, error) {
	u := fmt.Sprintf("gists/%v/%v", id, sha)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	gist := new(Gist)
	resp, err := s.client.Do(ctx, req, gist)
	if err != nil {
		return nil, resp, err
	}

	return gist, resp, nil
}

// Create a gist for authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#create-a-gist
func (s *GistsService) Create(ctx context.Context, gist *Gist) (*Gist, *Response, error) {
	u := "gists"
	req, err := s.client.NewRequest("POST", u, gist)
	if err != nil {
		return nil, nil, err
	}

	g := new(Gist)
	resp, err := s.client.Do(ctx, req, g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, nil
}

// Edit a gist.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#update-a-gist
func (s *GistsService) Edit(ctx context.Context, id string, gist *Gist) (*Gist, *Response, error) {
	u := fmt.Sprintf("gists/%v", id)
	req, err := s.client.NewRequest("PATCH", u, gist)
	if err != nil {
		return nil, nil, err
	}

	g := new(Gist)
	resp, err := s.client.Do(ctx, req, g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, nil
}

// ListCommits lists commits of a gist.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#list-gist-commits
func (s *GistsService) ListCommits(ctx context.Context, id string, opts *ListOptions) ([]*GistCommit, *Response, error) {
	u := fmt.Sprintf("gists/%v/commits", id)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var gistCommits []*GistCommit
	resp, err := s.client.Do(ctx, req, &gistCommits)
	if err != nil {
		return nil, resp, err
	}

	return gistCommits, resp, nil
}

// Delete a gist.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#delete-a-gist
func (s *GistsService) Delete(ctx context.Context, id string) (*Response, error) {
	u := fmt.Sprintf("gists/%v", id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}

// Star a gist on behalf of authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#star-a-gist
func (s *GistsService) Star(ctx context.Context, id string) (*Response, error) {
	u := fmt.Sprintf("gists/%v/star", id)
	req, err := s.client.NewRequest("PUT", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}

// Unstar a gist on a behalf of authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#unstar-a-gist
func (s *GistsService) Unstar(ctx context.Context, id string) (*Response, error) {
	u := fmt.Sprintf("gists/%v/star", id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}

// IsStarred checks if a gist is starred by authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#check-if-a-gist-is-starred
func (s *GistsService) IsStarred(ctx context.Context, id string) (bool, *Response, error) {
	u := fmt.Sprintf("gists/%v/star", id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return false, nil, err
	}
	resp, err := s.client.Do(ctx, req, nil)
	starred, err := parseBoolResponse(err)
	return starred, resp, err
}

// Fork a gist.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#fork-a-gist
func (s *GistsService) Fork(ctx context.Context, id string) (*Gist, *Response, error) {
	u := fmt.Sprintf("gists/%v/forks", id)
	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, nil, err
	}

	g := new(Gist)
	resp, err := s.client.Do(ctx, req, g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, nil
}

// ListForks lists forks of a gist.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/gists/#list-gist-forks
func (s *GistsService) ListForks(ctx context.Context, id string, opts *ListOptions) ([]*GistFork, *Response, error) {
	u := fmt.Sprintf("gists/%v/forks", id)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var gistForks []*GistFork
	resp, err := s.client.Do(ctx, req, &gistForks)
	if err != nil {
		return nil, resp, err
	}

	return gistForks, resp, nil
}
