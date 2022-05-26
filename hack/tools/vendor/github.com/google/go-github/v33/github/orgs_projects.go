// Copyright 2017 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// ListProjects lists the projects for an organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#list-organization-projects
func (s *OrganizationsService) ListProjects(ctx context.Context, org string, opts *ProjectListOptions) ([]*Project, *Response, error) {
	u := fmt.Sprintf("orgs/%v/projects", org)
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

// CreateProject creates a GitHub Project for the specified organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#create-an-organization-project
func (s *OrganizationsService) CreateProject(ctx context.Context, org string, opts *ProjectOptions) (*Project, *Response, error) {
	u := fmt.Sprintf("orgs/%v/projects", org)
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
