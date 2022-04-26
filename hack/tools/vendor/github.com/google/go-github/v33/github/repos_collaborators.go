// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// ListCollaboratorsOptions specifies the optional parameters to the
// RepositoriesService.ListCollaborators method.
type ListCollaboratorsOptions struct {
	// Affiliation specifies how collaborators should be filtered by their affiliation.
	// Possible values are:
	//     outside - All outside collaborators of an organization-owned repository
	//     direct - All collaborators with permissions to an organization-owned repository,
	//              regardless of organization membership status
	//     all - All collaborators the authenticated user can see
	//
	// Default value is "all".
	Affiliation string `url:"affiliation,omitempty"`

	ListOptions
}

// CollaboratorInvitation represents an invitation created when adding a collaborator.
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/collaborators/#response-when-a-new-invitation-is-created
type CollaboratorInvitation struct {
	ID          *int64      `json:"id,omitempty"`
	Repo        *Repository `json:"repository,omitempty"`
	Invitee     *User       `json:"invitee,omitempty"`
	Inviter     *User       `json:"inviter,omitempty"`
	Permissions *string     `json:"permissions,omitempty"`
	CreatedAt   *Timestamp  `json:"created_at,omitempty"`
	URL         *string     `json:"url,omitempty"`
	HTMLURL     *string     `json:"html_url,omitempty"`
}

// ListCollaborators lists the GitHub users that have access to the repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#list-repository-collaborators
func (s *RepositoriesService) ListCollaborators(ctx context.Context, owner, repo string, opts *ListCollaboratorsOptions) ([]*User, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/collaborators", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var users []*User
	resp, err := s.client.Do(ctx, req, &users)
	if err != nil {
		return nil, resp, err
	}

	return users, resp, nil
}

// IsCollaborator checks whether the specified GitHub user has collaborator
// access to the given repo.
// Note: This will return false if the user is not a collaborator OR the user
// is not a GitHub user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#check-if-a-user-is-a-repository-collaborator
func (s *RepositoriesService) IsCollaborator(ctx context.Context, owner, repo, user string) (bool, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/collaborators/%v", owner, repo, user)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return false, nil, err
	}

	resp, err := s.client.Do(ctx, req, nil)
	isCollab, err := parseBoolResponse(err)
	return isCollab, resp, err
}

// RepositoryPermissionLevel represents the permission level an organization
// member has for a given repository.
type RepositoryPermissionLevel struct {
	// Possible values: "admin", "write", "read", "none"
	Permission *string `json:"permission,omitempty"`

	User *User `json:"user,omitempty"`
}

// GetPermissionLevel retrieves the specific permission level a collaborator has for a given repository.
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-repository-permissions-for-a-user
func (s *RepositoriesService) GetPermissionLevel(ctx context.Context, owner, repo, user string) (*RepositoryPermissionLevel, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/collaborators/%v/permission", owner, repo, user)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	rpl := new(RepositoryPermissionLevel)
	resp, err := s.client.Do(ctx, req, rpl)
	if err != nil {
		return nil, resp, err
	}
	return rpl, resp, nil
}

// RepositoryAddCollaboratorOptions specifies the optional parameters to the
// RepositoriesService.AddCollaborator method.
type RepositoryAddCollaboratorOptions struct {
	// Permission specifies the permission to grant the user on this repository.
	// Possible values are:
	//     pull - team members can pull, but not push to or administer this repository
	//     push - team members can pull and push, but not administer this repository
	//     admin - team members can pull, push and administer this repository
	//     maintain - team members can manage the repository without access to sensitive or destructive actions.
	//     triage - team members can proactively manage issues and pull requests without write access.
	//
	// Default value is "push". This option is only valid for organization-owned repositories.
	Permission string `json:"permission,omitempty"`
}

// AddCollaborator sends an invitation to the specified GitHub user
// to become a collaborator to the given repo.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#add-a-repository-collaborator
func (s *RepositoriesService) AddCollaborator(ctx context.Context, owner, repo, user string, opts *RepositoryAddCollaboratorOptions) (*CollaboratorInvitation, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/collaborators/%v", owner, repo, user)
	req, err := s.client.NewRequest("PUT", u, opts)
	if err != nil {
		return nil, nil, err
	}
	acr := new(CollaboratorInvitation)
	resp, err := s.client.Do(ctx, req, acr)
	if err != nil {
		return nil, resp, err
	}
	return acr, resp, nil
}

// RemoveCollaborator removes the specified GitHub user as collaborator from the given repo.
// Note: Does not return error if a valid user that is not a collaborator is removed.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#remove-a-repository-collaborator
func (s *RepositoriesService) RemoveCollaborator(ctx context.Context, owner, repo, user string) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/collaborators/%v", owner, repo, user)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}
