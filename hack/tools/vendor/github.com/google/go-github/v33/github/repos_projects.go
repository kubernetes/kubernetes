// Copyright 2017 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// ProjectListOptions specifies the optional parameters to the
// OrganizationsService.ListProjects and RepositoriesService.ListProjects methods.
type ProjectListOptions struct {
	// Indicates the state of the projects to return. Can be either open, closed, or all. Default: open
	State string `url:"state,omitempty"`

	ListOptions
}

// ListProjects lists the projects for a repo.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#list-repository-projects
func (s *RepositoriesService) ListProjects(ctx context.Context, owner, repo string, opts *ProjectListOptions) ([]*Project, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/projects", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	var projects []*Project
	resp, err := s.client.Do(ctx, req, &projects)
	if err != nil {
		return nil, resp, err
	}

	return projects, resp, nil
}

// CreateProject creates a GitHub Project for the specified repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#create-a-repository-project
func (s *RepositoriesService) CreateProject(ctx context.Context, owner, repo string, opts *ProjectOptions) (*Project, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/projects", owner, repo)
	req, err := s.client.NewRequest("POST", u, opts)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	project := &Project{}
	resp, err := s.client.Do(ctx, req, project)
	if err != nil {
		return nil, resp, err
	}

	return project, resp, nil
}
