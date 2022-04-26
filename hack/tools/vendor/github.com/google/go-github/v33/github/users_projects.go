// Copyright 2019 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// ListProjects lists the projects for the specified user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#list-user-projects
func (s *UsersService) ListProjects(ctx context.Context, user string, opts *ProjectListOptions) ([]*Project, *Response, error) {
	u := fmt.Sprintf("users/%v/projects", user)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	var projects []*Project
	resp, err := s.client.Do(ctx, req, &projects)
	if err != nil {
		return nil, resp, err
	}

	return projects, resp, nil
}

// CreateUserProjectOptions specifies the parameters to the UsersService.CreateProject method.
type CreateUserProjectOptions struct {
	// The name of the project. (Required.)
	Name string `json:"name"`
	// The description of the project. (Optional.)
	Body *string `json:"body,omitempty"`
}

// CreateProject creates a GitHub Project for the current user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#create-a-user-project
func (s *UsersService) CreateProject(ctx context.Context, opts *CreateUserProjectOptions) (*Project, *Response, error) {
	u := "user/projects"
	req, err := s.client.NewRequest("POST", u, opts)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	project := &Project{}
	resp, err := s.client.Do(ctx, req, project)
	if err != nil {
		return nil, resp, err
	}

	return project, resp, nil
}
