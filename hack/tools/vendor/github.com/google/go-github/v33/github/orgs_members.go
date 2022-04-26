// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// Membership represents the status of a user's membership in an organization or team.
type Membership struct {
	URL *string `json:"url,omitempty"`

	// State is the user's status within the organization or team.
	// Possible values are: "active", "pending"
	State *string `json:"state,omitempty"`

	// Role identifies the user's role within the organization or team.
	// Possible values for organization membership:
	//     member - non-owner organization member
	//     admin - organization owner
	//
	// Possible values for team membership are:
	//     member - a normal member of the team
	//     maintainer - a team maintainer. Able to add/remove other team
	//                  members, promote other team members to team
	//                  maintainer, and edit the team’s name and description
	Role *string `json:"role,omitempty"`

	// For organization membership, the API URL of the organization.
	OrganizationURL *string `json:"organization_url,omitempty"`

	// For organization membership, the organization the membership is for.
	Organization *Organization `json:"organization,omitempty"`

	// For organization membership, the user the membership is for.
	User *User `json:"user,omitempty"`
}

func (m Membership) String() string {
	return Stringify(m)
}

// ListMembersOptions specifies optional parameters to the
// OrganizationsService.ListMembers method.
type ListMembersOptions struct {
	// If true (or if the authenticated user is not an owner of the
	// organization), list only publicly visible members.
	PublicOnly bool `url:"-"`

	// Filter members returned in the list. Possible values are:
	// 2fa_disabled, all. Default is "all".
	Filter string `url:"filter,omitempty"`

	// Role filters members returned by their role in the organization.
	// Possible values are:
	//     all - all members of the organization, regardless of role
	//     admin - organization owners
	//     member - non-owner organization members
	//
	// Default is "all".
	Role string `url:"role,omitempty"`

	ListOptions
}

// ListMembers lists the members for an organization. If the authenticated
// user is an owner of the organization, this will return both concealed and
// public members, otherwise it will only return public members.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#list-organization-members
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#list-public-organization-members
func (s *OrganizationsService) ListMembers(ctx context.Context, org string, opts *ListMembersOptions) ([]*User, *Response, error) {
	var u string
	if opts != nil && opts.PublicOnly {
		u = fmt.Sprintf("orgs/%v/public_members", org)
	} else {
		u = fmt.Sprintf("orgs/%v/members", org)
	}
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

// IsMember checks if a user is a member of an organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#check-organization-membership-for-a-user
func (s *OrganizationsService) IsMember(ctx context.Context, org, user string) (bool, *Response, error) {
	u := fmt.Sprintf("orgs/%v/members/%v", org, user)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return false, nil, err
	}

	resp, err := s.client.Do(ctx, req, nil)
	member, err := parseBoolResponse(err)
	return member, resp, err
}

// IsPublicMember checks if a user is a public member of an organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#check-public-organization-membership-for-a-user
func (s *OrganizationsService) IsPublicMember(ctx context.Context, org, user string) (bool, *Response, error) {
	u := fmt.Sprintf("orgs/%v/public_members/%v", org, user)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return false, nil, err
	}

	resp, err := s.client.Do(ctx, req, nil)
	member, err := parseBoolResponse(err)
	return member, resp, err
}

// RemoveMember removes a user from all teams of an organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#remove-an-organization-member
func (s *OrganizationsService) RemoveMember(ctx context.Context, org, user string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/members/%v", org, user)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// PublicizeMembership publicizes a user's membership in an organization. (A
// user cannot publicize the membership for another user.)
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#set-public-organization-membership-for-the-authenticated-user
func (s *OrganizationsService) PublicizeMembership(ctx context.Context, org, user string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/public_members/%v", org, user)
	req, err := s.client.NewRequest("PUT", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// ConcealMembership conceals a user's membership in an organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#remove-public-organization-membership-for-the-authenticated-user
func (s *OrganizationsService) ConcealMembership(ctx context.Context, org, user string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/public_members/%v", org, user)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// ListOrgMembershipsOptions specifies optional parameters to the
// OrganizationsService.ListOrgMemberships method.
type ListOrgMembershipsOptions struct {
	// Filter memberships to include only those with the specified state.
	// Possible values are: "active", "pending".
	State string `url:"state,omitempty"`

	ListOptions
}

// ListOrgMemberships lists the organization memberships for the authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#list-organization-memberships-for-the-authenticated-user
func (s *OrganizationsService) ListOrgMemberships(ctx context.Context, opts *ListOrgMembershipsOptions) ([]*Membership, *Response, error) {
	u := "user/memberships/orgs"
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var memberships []*Membership
	resp, err := s.client.Do(ctx, req, &memberships)
	if err != nil {
		return nil, resp, err
	}

	return memberships, resp, nil
}

// GetOrgMembership gets the membership for a user in a specified organization.
// Passing an empty string for user will get the membership for the
// authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#get-an-organization-membership-for-the-authenticated-user
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#get-organization-membership-for-a-user
func (s *OrganizationsService) GetOrgMembership(ctx context.Context, user, org string) (*Membership, *Response, error) {
	var u string
	if user != "" {
		u = fmt.Sprintf("orgs/%v/memberships/%v", org, user)
	} else {
		u = fmt.Sprintf("user/memberships/orgs/%v", org)
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	membership := new(Membership)
	resp, err := s.client.Do(ctx, req, membership)
	if err != nil {
		return nil, resp, err
	}

	return membership, resp, nil
}

// EditOrgMembership edits the membership for user in specified organization.
// Passing an empty string for user will edit the membership for the
// authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#update-an-organization-membership-for-the-authenticated-user
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#set-organization-membership-for-a-user
func (s *OrganizationsService) EditOrgMembership(ctx context.Context, user, org string, membership *Membership) (*Membership, *Response, error) {
	var u, method string
	if user != "" {
		u = fmt.Sprintf("orgs/%v/memberships/%v", org, user)
		method = "PUT"
	} else {
		u = fmt.Sprintf("user/memberships/orgs/%v", org)
		method = "PATCH"
	}

	req, err := s.client.NewRequest(method, u, membership)
	if err != nil {
		return nil, nil, err
	}

	m := new(Membership)
	resp, err := s.client.Do(ctx, req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, nil
}

// RemoveOrgMembership removes user from the specified organization. If the
// user has been invited to the organization, this will cancel their invitation.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#remove-organization-membership-for-a-user
func (s *OrganizationsService) RemoveOrgMembership(ctx context.Context, user, org string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/memberships/%v", org, user)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// ListPendingOrgInvitations returns a list of pending invitations.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#list-pending-organization-invitations
func (s *OrganizationsService) ListPendingOrgInvitations(ctx context.Context, org string, opts *ListOptions) ([]*Invitation, *Response, error) {
	u := fmt.Sprintf("orgs/%v/invitations", org)
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

// CreateOrgInvitationOptions specifies the parameters to the OrganizationService.Invite
// method.
type CreateOrgInvitationOptions struct {
	// GitHub user ID for the person you are inviting. Not required if you provide Email.
	InviteeID *int64 `json:"invitee_id,omitempty"`
	// Email address of the person you are inviting, which can be an existing GitHub user.
	// Not required if you provide InviteeID
	Email *string `json:"email,omitempty"`
	// Specify role for new member. Can be one of:
	// * admin - Organization owners with full administrative rights to the
	// 	 organization and complete access to all repositories and teams.
	// * direct_member - Non-owner organization members with ability to see
	//   other members and join teams by invitation.
	// * billing_manager - Non-owner organization members with ability to
	//   manage the billing settings of your organization.
	// Default is "direct_member".
	Role   *string `json:"role"`
	TeamID []int64 `json:"team_ids"`
}

// CreateOrgInvitation invites people to an organization by using their GitHub user ID or their email address.
// In order to create invitations in an organization,
// the authenticated user must be an organization owner.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#create-an-organization-invitation
func (s *OrganizationsService) CreateOrgInvitation(ctx context.Context, org string, opts *CreateOrgInvitationOptions) (*Invitation, *Response, error) {
	u := fmt.Sprintf("orgs/%v/invitations", org)

	req, err := s.client.NewRequest("POST", u, opts)
	if err != nil {
		return nil, nil, err
	}

	var invitation *Invitation
	resp, err := s.client.Do(ctx, req, &invitation)
	if err != nil {
		return nil, resp, err
	}
	return invitation, resp, nil
}

// ListOrgInvitationTeams lists all teams associated with an invitation. In order to see invitations in an organization,
// the authenticated user must be an organization owner.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#list-organization-invitation-teams
func (s *OrganizationsService) ListOrgInvitationTeams(ctx context.Context, org, invitationID string, opts *ListOptions) ([]*Team, *Response, error) {
	u := fmt.Sprintf("orgs/%v/invitations/%v/teams", org, invitationID)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var orgInvitationTeams []*Team
	resp, err := s.client.Do(ctx, req, &orgInvitationTeams)
	if err != nil {
		return nil, resp, err
	}
	return orgInvitationTeams, resp, nil
}
