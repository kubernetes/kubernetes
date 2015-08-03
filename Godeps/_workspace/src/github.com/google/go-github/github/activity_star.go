// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "fmt"

// StarredRepository is returned by ListStarred.
type StarredRepository struct {
	StarredAt  *Timestamp  `json:"starred_at,omitempty"`
	Repository *Repository `json:"repo,omitempty"`
}

// ListStargazers lists people who have starred the specified repo.
//
// GitHub API Docs: https://developer.github.com/v3/activity/starring/#list-stargazers
func (s *ActivityService) ListStargazers(owner, repo string, opt *ListOptions) ([]User, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/stargazers", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	stargazers := new([]User)
	resp, err := s.client.Do(req, stargazers)
	if err != nil {
		return nil, resp, err
	}

	return *stargazers, resp, err
}

// ActivityListStarredOptions specifies the optional parameters to the
// ActivityService.ListStarred method.
type ActivityListStarredOptions struct {
	// How to sort the repository list.  Possible values are: created, updated,
	// pushed, full_name.  Default is "full_name".
	Sort string `url:"sort,omitempty"`

	// Direction in which to sort repositories.  Possible values are: asc, desc.
	// Default is "asc" when sort is "full_name", otherwise default is "desc".
	Direction string `url:"direction,omitempty"`

	ListOptions
}

// ListStarred lists all the repos starred by a user.  Passing the empty string
// will list the starred repositories for the authenticated user.
//
// GitHub API docs: http://developer.github.com/v3/activity/starring/#list-repositories-being-starred
func (s *ActivityService) ListStarred(user string, opt *ActivityListStarredOptions) ([]StarredRepository, *Response, error) {
	var u string
	if user != "" {
		u = fmt.Sprintf("users/%v/starred", user)
	} else {
		u = "user/starred"
	}
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches
	req.Header.Set("Accept", mediaTypeStarringPreview)

	repos := new([]StarredRepository)
	resp, err := s.client.Do(req, repos)
	if err != nil {
		return nil, resp, err
	}

	return *repos, resp, err
}

// IsStarred checks if a repository is starred by authenticated user.
//
// GitHub API docs: https://developer.github.com/v3/activity/starring/#check-if-you-are-starring-a-repository
func (s *ActivityService) IsStarred(owner, repo string) (bool, *Response, error) {
	u := fmt.Sprintf("user/starred/%v/%v", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return false, nil, err
	}
	resp, err := s.client.Do(req, nil)
	starred, err := parseBoolResponse(err)
	return starred, resp, err
}

// Star a repository as the authenticated user.
//
// GitHub API docs: https://developer.github.com/v3/activity/starring/#star-a-repository
func (s *ActivityService) Star(owner, repo string) (*Response, error) {
	u := fmt.Sprintf("user/starred/%v/%v", owner, repo)
	req, err := s.client.NewRequest("PUT", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// Unstar a repository as the authenticated user.
//
// GitHub API docs: https://developer.github.com/v3/activity/starring/#unstar-a-repository
func (s *ActivityService) Unstar(owner, repo string) (*Response, error) {
	u := fmt.Sprintf("user/starred/%v/%v", owner, repo)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}
