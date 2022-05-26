// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
	"strings"
)

// StarredRepository is returned by ListStarred.
type StarredRepository struct {
	StarredAt  *Timestamp  `json:"starred_at,omitempty"`
	Repository *Repository `json:"repo,omitempty"`
}

// Stargazer represents a user that has starred a repository.
type Stargazer struct {
	StarredAt *Timestamp `json:"starred_at,omitempty"`
	User      *User      `json:"user,omitempty"`
}

// ListStargazers lists people who have starred the specified repo.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#list-stargazers
func (s *ActivityService) ListStargazers(ctx context.Context, owner, repo string, opts *ListOptions) ([]*Stargazer, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/stargazers", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches
	req.Header.Set("Accept", mediaTypeStarringPreview)

	var stargazers []*Stargazer
	resp, err := s.client.Do(ctx, req, &stargazers)
	if err != nil {
		return nil, resp, err
	}

	return stargazers, resp, nil
}

// ActivityListStarredOptions specifies the optional parameters to the
// ActivityService.ListStarred method.
type ActivityListStarredOptions struct {
	// How to sort the repository list. Possible values are: created, updated,
	// pushed, full_name. Default is "full_name".
	Sort string `url:"sort,omitempty"`

	// Direction in which to sort repositories. Possible values are: asc, desc.
	// Default is "asc" when sort is "full_name", otherwise default is "desc".
	Direction string `url:"direction,omitempty"`

	ListOptions
}

// ListStarred lists all the repos starred by a user. Passing the empty string
// will list the starred repositories for the authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#list-repositories-starred-by-the-authenticated-user
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#list-repositories-starred-by-a-user
func (s *ActivityService) ListStarred(ctx context.Context, user string, opts *ActivityListStarredOptions) ([]*StarredRepository, *Response, error) {
	var u string
	if user != "" {
		u = fmt.Sprintf("users/%v/starred", user)
	} else {
		u = "user/starred"
	}
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when APIs fully launch
	acceptHeaders := []string{mediaTypeStarringPreview, mediaTypeTopicsPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	var repos []*StarredRepository
	resp, err := s.client.Do(ctx, req, &repos)
	if err != nil {
		return nil, resp, err
	}

	return repos, resp, nil
}

// IsStarred checks if a repository is starred by authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#check-if-a-repository-is-starred-by-the-authenticated-user
func (s *ActivityService) IsStarred(ctx context.Context, owner, repo string) (bool, *Response, error) {
	u := fmt.Sprintf("user/starred/%v/%v", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return false, nil, err
	}
	resp, err := s.client.Do(ctx, req, nil)
	starred, err := parseBoolResponse(err)
	return starred, resp, err
}

// Star a repository as the authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#star-a-repository-for-the-authenticated-user
func (s *ActivityService) Star(ctx context.Context, owner, repo string) (*Response, error) {
	u := fmt.Sprintf("user/starred/%v/%v", owner, repo)
	req, err := s.client.NewRequest("PUT", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}

// Unstar a repository as the authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#unstar-a-repository-for-the-authenticated-user
func (s *ActivityService) Unstar(ctx context.Context, owner, repo string) (*Response, error) {
	u := fmt.Sprintf("user/starred/%v/%v", owner, repo)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}
