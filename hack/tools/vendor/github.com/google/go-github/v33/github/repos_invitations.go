// Copyright 2016 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// RepositoryInvitation represents an invitation to collaborate on a repo.
type RepositoryInvitation struct {
	ID      *int64      `json:"id,omitempty"`
	Repo    *Repository `json:"repository,omitempty"`
	Invitee *User       `json:"invitee,omitempty"`
	Inviter *User       `json:"inviter,omitempty"`

	// Permissions represents the permissions that the associated user will have
	// on the repository. Possible values are: "read", "write", "admin".
	Permissions *string    `json:"permissions,omitempty"`
	CreatedAt   *Timestamp `json:"created_at,omitempty"`
	URL         *string    `json:"url,omitempty"`
	HTMLURL     *string    `json:"html_url,omitempty"`
}

// ListInvitations lists all currently-open repository invitations.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#list-repository-invitations
func (s *RepositoriesService) ListInvitations(ctx context.Context, owner, repo string, opts *ListOptions) ([]*RepositoryInvitation, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/invitations", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	invites := []*RepositoryInvitation{}
	resp, err := s.client.Do(ctx, req, &invites)
	if err != nil {
		return nil, resp, err
	}

	return invites, resp, nil
}

// DeleteInvitation deletes a repository invitation.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#delete-a-repository-invitation
func (s *RepositoriesService) DeleteInvitation(ctx context.Context, owner, repo string, invitationID int64) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/invitations/%v", owner, repo, invitationID)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// UpdateInvitation updates the permissions associated with a repository
// invitation.
//
// permissions represents the permissions that the associated user will have
// on the repository. Possible values are: "read", "write", "admin".
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#update-a-repository-invitation
func (s *RepositoriesService) UpdateInvitation(ctx context.Context, owner, repo string, invitationID int64, permissions string) (*RepositoryInvitation, *Response, error) {
	opts := &struct {
		Permissions string `json:"permissions"`
	}{Permissions: permissions}
	u := fmt.Sprintf("repos/%v/%v/invitations/%v", owner, repo, invitationID)
	req, err := s.client.NewRequest("PATCH", u, opts)
	if err != nil {
		return nil, nil, err
	}

	invite := &RepositoryInvitation{}
	resp, err := s.client.Do(ctx, req, invite)
	if err != nil {
		return nil, resp, err
	}

	return invite, resp, nil
}
