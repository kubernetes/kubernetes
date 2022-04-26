// Copyright 2016 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// ListRepos lists the repositories that are accessible to the authenticated installation.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps/#list-repositories-accessible-to-the-app-installation
func (s *AppsService) ListRepos(ctx context.Context, opts *ListOptions) ([]*Repository, *Response, error) {
	u, err := addOptions("installation/repositories", opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var r struct {
		Repositories []*Repository `json:"repositories"`
	}
	resp, err := s.client.Do(ctx, req, &r)
	if err != nil {
		return nil, resp, err
	}

	return r.Repositories, resp, nil
}

// ListUserRepos lists repositories that are accessible
// to the authenticated user for an installation.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps/#list-repositories-accessible-to-the-user-access-token
func (s *AppsService) ListUserRepos(ctx context.Context, id int64, opts *ListOptions) ([]*Repository, *Response, error) {
	u := fmt.Sprintf("user/installations/%v/repositories", id)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var r struct {
		Repositories []*Repository `json:"repositories"`
	}
	resp, err := s.client.Do(ctx, req, &r)
	if err != nil {
		return nil, resp, err
	}

	return r.Repositories, resp, nil
}

// AddRepository adds a single repository to an installation.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps/#add-a-repository-to-an-app-installation
func (s *AppsService) AddRepository(ctx context.Context, instID, repoID int64) (*Repository, *Response, error) {
	u := fmt.Sprintf("user/installations/%v/repositories/%v", instID, repoID)
	req, err := s.client.NewRequest("PUT", u, nil)
	if err != nil {
		return nil, nil, err
	}

	r := new(Repository)
	resp, err := s.client.Do(ctx, req, r)
	if err != nil {
		return nil, resp, err
	}

	return r, resp, nil
}

// RemoveRepository removes a single repository from an installation.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps/#remove-a-repository-from-an-app-installation
func (s *AppsService) RemoveRepository(ctx context.Context, instID, repoID int64) (*Response, error) {
	u := fmt.Sprintf("user/installations/%v/repositories/%v", instID, repoID)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// RevokeInstallationToken revokes an installation token.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps/#revoke-an-installation-access-token
func (s *AppsService) RevokeInstallationToken(ctx context.Context) (*Response, error) {
	u := "installation/token"
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
