// Copyright 2017 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// ListOutsideCollaboratorsOptions specifies optional parameters to the
// OrganizationsService.ListOutsideCollaborators method.
type ListOutsideCollaboratorsOptions struct {
	// Filter outside collaborators returned in the list. Possible values are:
	// 2fa_disabled, all.  Default is "all".
	Filter string `url:"filter,omitempty"`

	ListOptions
}

// ListOutsideCollaborators lists outside collaborators of organization's repositories.
// This will only work if the authenticated
// user is an owner of the organization.
//
// Warning: The API may change without advance notice during the preview period.
// Preview features are not supported for production use.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#list-outside-collaborators-for-an-organization
func (s *OrganizationsService) ListOutsideCollaborators(ctx context.Context, org string, opts *ListOutsideCollaboratorsOptions) ([]*User, *Response, error) {
	u := fmt.Sprintf("orgs/%v/outside_collaborators", org)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var members []*User
	resp, err := s.client.Do(ctx, req, &members)
	if err != nil {
		return nil, resp, err
	}

	return members, resp, nil
}

// RemoveOutsideCollaborator removes a user from the list of outside collaborators;
// consequently, removing them from all the organization's repositories.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#remove-outside-collaborator-from-an-organization
func (s *OrganizationsService) RemoveOutsideCollaborator(ctx context.Context, org string, user string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/outside_collaborators/%v", org, user)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// ConvertMemberToOutsideCollaborator reduces the permission level of a member of the
// organization to that of an outside collaborator. Therefore, they will only
// have access to the repositories that their current team membership allows.
// Responses for converting a non-member or the last owner to an outside collaborator
// are listed in GitHub API docs.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#convert-an-organization-member-to-outside-collaborator
func (s *OrganizationsService) ConvertMemberToOutsideCollaborator(ctx context.Context, org string, user string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/outside_collaborators/%v", org, user)
	req, err := s.client.NewRequest("PUT", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
