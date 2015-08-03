// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "fmt"

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

	// Filter members returned in the list.  Possible values are:
	// 2fa_disabled, all.  Default is "all".
	Filter string `url:"filter,omitempty"`

	// Role filters memebers returned by their role in the organization.
	// Possible values are:
	//     all - all members of the organization, regardless of role
	//     admin - organization owners
	//     member - non-organization members
	//
	// Default is "all".
	Role string `url:"role,omitempty"`

	ListOptions
}

// ListMembers lists the members for an organization.  If the authenticated
// user is an owner of the organization, this will return both concealed and
// public members, otherwise it will only return public members.
//
// GitHub API docs: http://developer.github.com/v3/orgs/members/#members-list
func (s *OrganizationsService) ListMembers(org string, opt *ListMembersOptions) ([]User, *Response, error) {
	var u string
	if opt != nil && opt.PublicOnly {
		u = fmt.Sprintf("orgs/%v/public_members", org)
	} else {
		u = fmt.Sprintf("orgs/%v/members", org)
	}
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	if opt != nil && opt.Role != "" {
		req.Header.Set("Accept", mediaTypeOrgPermissionPreview)
	}

	members := new([]User)
	resp, err := s.client.Do(req, members)
	if err != nil {
		return nil, resp, err
	}

	return *members, resp, err
}

// IsMember checks if a user is a member of an organization.
//
// GitHub API docs: http://developer.github.com/v3/orgs/members/#check-membership
func (s *OrganizationsService) IsMember(org, user string) (bool, *Response, error) {
	u := fmt.Sprintf("orgs/%v/members/%v", org, user)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return false, nil, err
	}

	resp, err := s.client.Do(req, nil)
	member, err := parseBoolResponse(err)
	return member, resp, err
}

// IsPublicMember checks if a user is a public member of an organization.
//
// GitHub API docs: http://developer.github.com/v3/orgs/members/#check-public-membership
func (s *OrganizationsService) IsPublicMember(org, user string) (bool, *Response, error) {
	u := fmt.Sprintf("orgs/%v/public_members/%v", org, user)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return false, nil, err
	}

	resp, err := s.client.Do(req, nil)
	member, err := parseBoolResponse(err)
	return member, resp, err
}

// RemoveMember removes a user from all teams of an organization.
//
// GitHub API docs: http://developer.github.com/v3/orgs/members/#remove-a-member
func (s *OrganizationsService) RemoveMember(org, user string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/members/%v", org, user)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// PublicizeMembership publicizes a user's membership in an organization. (A
// user cannot publicize the membership for another user.)
//
// GitHub API docs: http://developer.github.com/v3/orgs/members/#publicize-a-users-membership
func (s *OrganizationsService) PublicizeMembership(org, user string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/public_members/%v", org, user)
	req, err := s.client.NewRequest("PUT", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ConcealMembership conceals a user's membership in an organization.
//
// GitHub API docs: http://developer.github.com/v3/orgs/members/#conceal-a-users-membership
func (s *OrganizationsService) ConcealMembership(org, user string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/public_members/%v", org, user)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ListOrgMembershipsOptions specifies optional parameters to the
// OrganizationsService.ListOrgMemberships method.
type ListOrgMembershipsOptions struct {
	// Filter memberships to include only those withe the specified state.
	// Possible values are: "active", "pending".
	State string `url:"state,omitempty"`

	ListOptions
}

// ListOrgMemberships lists the organization memberships for the authenticated user.
//
// GitHub API docs: https://developer.github.com/v3/orgs/members/#list-your-organization-memberships
func (s *OrganizationsService) ListOrgMemberships(opt *ListOrgMembershipsOptions) ([]Membership, *Response, error) {
	u := "user/memberships/orgs"
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var memberships []Membership
	resp, err := s.client.Do(req, &memberships)
	if err != nil {
		return nil, resp, err
	}

	return memberships, resp, err
}

// GetOrgMembership gets the membership for a user in a specified organization.
// Passing an empty string for user will get the membership for the
// authenticated user.
//
// GitHub API docs: https://developer.github.com/v3/orgs/members/#get-organization-membership
// GitHub API docs: https://developer.github.com/v3/orgs/members/#get-your-organization-membership
func (s *OrganizationsService) GetOrgMembership(user, org string) (*Membership, *Response, error) {
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
	resp, err := s.client.Do(req, membership)
	if err != nil {
		return nil, resp, err
	}

	return membership, resp, err
}

// EditOrgMembership edits the membership for user in specified organization.
// Passing an empty string for user will edit the membership for the
// authenticated user.
//
// GitHub API docs: https://developer.github.com/v3/orgs/members/#add-or-update-organization-membership
// GitHub API docs: https://developer.github.com/v3/orgs/members/#edit-your-organization-membership
func (s *OrganizationsService) EditOrgMembership(user, org string, membership *Membership) (*Membership, *Response, error) {
	var u string
	if user != "" {
		u = fmt.Sprintf("orgs/%v/memberships/%v", org, user)
	} else {
		u = fmt.Sprintf("user/memberships/orgs/%v", org)
	}

	req, err := s.client.NewRequest("PATCH", u, membership)
	if err != nil {
		return nil, nil, err
	}

	m := new(Membership)
	resp, err := s.client.Do(req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// RemoveOrgMembership removes user from the specified organization.  If the
// user has been invited to the organization, this will cancel their invitation.
//
// GitHub API docs: https://developer.github.com/v3/orgs/members/#remove-organization-membership
func (s *OrganizationsService) RemoveOrgMembership(user, org string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/memberships/%v", org, user)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
