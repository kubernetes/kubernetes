// Copyright 2018 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// TeamListTeamMembersOptions specifies the optional parameters to the
// TeamsService.ListTeamMembers method.
type TeamListTeamMembersOptions struct {
	// Role filters members returned by their role in the team. Possible
	// values are "all", "member", "maintainer". Default is "all".
	Role string `url:"role,omitempty"`

	ListOptions
}

// ListTeamMembersByID lists all of the users who are members of a team, given a specified
// organization ID, by team ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-team-members
func (s *TeamsService) ListTeamMembersByID(ctx context.Context, orgID, teamID int64, opts *TeamListTeamMembersOptions) ([]*User, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/members", orgID, teamID)
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

// ListTeamMembersBySlug lists all of the users who are members of a team, given a specified
// organization name, by team slug.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-team-members
func (s *TeamsService) ListTeamMembersBySlug(ctx context.Context, org, slug string, opts *TeamListTeamMembersOptions) ([]*User, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/members", org, slug)
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

// GetTeamMembershipByID returns the membership status for a user in a team, given a specified
// organization ID, by team ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#get-team-membership-for-a-user
func (s *TeamsService) GetTeamMembershipByID(ctx context.Context, orgID, teamID int64, user string) (*Membership, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/memberships/%v", orgID, teamID, user)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	t := new(Membership)
	resp, err := s.client.Do(ctx, req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, nil
}

// GetTeamMembershipBySlug returns the membership status for a user in a team, given a specified
// organization name, by team slug.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#get-team-membership-for-a-user
func (s *TeamsService) GetTeamMembershipBySlug(ctx context.Context, org, slug, user string) (*Membership, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/memberships/%v", org, slug, user)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	t := new(Membership)
	resp, err := s.client.Do(ctx, req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, nil
}

// TeamAddTeamMembershipOptions specifies the optional
// parameters to the TeamsService.AddTeamMembership method.
type TeamAddTeamMembershipOptions struct {
	// Role specifies the role the user should have in the team. Possible
	// values are:
	//     member - a normal member of the team
	//     maintainer - a team maintainer. Able to add/remove other team
	//                  members, promote other team members to team
	//                  maintainer, and edit the team’s name and description
	//
	// Default value is "member".
	Role string `json:"role,omitempty"`
}

// AddTeamMembershipByID adds or invites a user to a team, given a specified
// organization ID, by team ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#add-or-update-team-membership-for-a-user
func (s *TeamsService) AddTeamMembershipByID(ctx context.Context, orgID, teamID int64, user string, opts *TeamAddTeamMembershipOptions) (*Membership, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/memberships/%v", orgID, teamID, user)
	req, err := s.client.NewRequest("PUT", u, opts)
	if err != nil {
		return nil, nil, err
	}

	t := new(Membership)
	resp, err := s.client.Do(ctx, req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, nil
}

// AddTeamMembershipBySlug adds or invites a user to a team, given a specified
// organization name, by team slug.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#add-or-update-team-membership-for-a-user
func (s *TeamsService) AddTeamMembershipBySlug(ctx context.Context, org, slug, user string, opts *TeamAddTeamMembershipOptions) (*Membership, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/memberships/%v", org, slug, user)
	req, err := s.client.NewRequest("PUT", u, opts)
	if err != nil {
		return nil, nil, err
	}

	t := new(Membership)
	resp, err := s.client.Do(ctx, req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, nil
}

// RemoveTeamMembershipByID removes a user from a team, given a specified
// organization ID, by team ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#remove-team-membership-for-a-user
func (s *TeamsService) RemoveTeamMembershipByID(ctx context.Context, orgID, teamID int64, user string) (*Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/memberships/%v", orgID, teamID, user)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// RemoveTeamMembershipBySlug removes a user from a team, given a specified
// organization name, by team slug.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#remove-team-membership-for-a-user
func (s *TeamsService) RemoveTeamMembershipBySlug(ctx context.Context, org, slug, user string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/memberships/%v", org, slug, user)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// ListPendingTeamInvitationsByID gets pending invitation list of a team, given a specified
// organization ID, by team ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-pending-team-invitations
func (s *TeamsService) ListPendingTeamInvitationsByID(ctx context.Context, orgID, teamID int64, opts *ListOptions) ([]*Invitation, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/invitations", orgID, teamID)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var pendingInvitations []*Invitation
	resp, err := s.client.Do(ctx, req, &pendingInvitations)
	if err != nil {
		return nil, resp, err
	}

	return pendingInvitations, resp, nil
}

// ListPendingTeamInvitationsBySlug get pending invitation list of a team, given a specified
// organization name, by team slug.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-pending-team-invitations
func (s *TeamsService) ListPendingTeamInvitationsBySlug(ctx context.Context, org, slug string, opts *ListOptions) ([]*Invitation, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/invitations", org, slug)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var pendingInvitations []*Invitation
	resp, err := s.client.Do(ctx, req, &pendingInvitations)
	if err != nil {
		return nil, resp, err
	}

	return pendingInvitations, resp, nil
}
