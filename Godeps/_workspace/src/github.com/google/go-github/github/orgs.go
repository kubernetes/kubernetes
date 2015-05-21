// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// OrganizationsService provides access to the organization related functions
// in the GitHub API.
//
// GitHub API docs: http://developer.github.com/v3/orgs/
type OrganizationsService struct {
	client *Client
}

// Organization represents a GitHub organization account.
type Organization struct {
	Login             *string    `json:"login,omitempty"`
	ID                *int       `json:"id,omitempty"`
	AvatarURL         *string    `json:"avatar_url,omitempty"`
	HTMLURL           *string    `json:"html_url,omitempty"`
	Name              *string    `json:"name,omitempty"`
	Company           *string    `json:"company,omitempty"`
	Blog              *string    `json:"blog,omitempty"`
	Location          *string    `json:"location,omitempty"`
	Email             *string    `json:"email,omitempty"`
	PublicRepos       *int       `json:"public_repos,omitempty"`
	PublicGists       *int       `json:"public_gists,omitempty"`
	Followers         *int       `json:"followers,omitempty"`
	Following         *int       `json:"following,omitempty"`
	CreatedAt         *time.Time `json:"created_at,omitempty"`
	UpdatedAt         *time.Time `json:"updated_at,omitempty"`
	TotalPrivateRepos *int       `json:"total_private_repos,omitempty"`
	OwnedPrivateRepos *int       `json:"owned_private_repos,omitempty"`
	PrivateGists      *int       `json:"private_gists,omitempty"`
	DiskUsage         *int       `json:"disk_usage,omitempty"`
	Collaborators     *int       `json:"collaborators,omitempty"`
	BillingEmail      *string    `json:"billing_email,omitempty"`
	Type              *string    `json:"type,omitempty"`
	Plan              *Plan      `json:"plan,omitempty"`

	// API URLs
	URL              *string `json:"url,omitempty"`
	EventsURL        *string `json:"events_url,omitempty"`
	MembersURL       *string `json:"members_url,omitempty"`
	PublicMembersURL *string `json:"public_members_url,omitempty"`
	ReposURL         *string `json:"repos_url,omitempty"`
}

func (o Organization) String() string {
	return Stringify(o)
}

// Plan represents the payment plan for an account.  See plans at https://github.com/plans.
type Plan struct {
	Name          *string `json:"name,omitempty"`
	Space         *int    `json:"space,omitempty"`
	Collaborators *int    `json:"collaborators,omitempty"`
	PrivateRepos  *int    `json:"private_repos,omitempty"`
}

func (p Plan) String() string {
	return Stringify(p)
}

// List the organizations for a user.  Passing the empty string will list
// organizations for the authenticated user.
//
// GitHub API docs: http://developer.github.com/v3/orgs/#list-user-organizations
func (s *OrganizationsService) List(user string, opt *ListOptions) ([]Organization, *Response, error) {
	var u string
	if user != "" {
		u = fmt.Sprintf("users/%v/orgs", user)
	} else {
		u = "user/orgs"
	}
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	orgs := new([]Organization)
	resp, err := s.client.Do(req, orgs)
	if err != nil {
		return nil, resp, err
	}

	return *orgs, resp, err
}

// Get fetches an organization by name.
//
// GitHub API docs: http://developer.github.com/v3/orgs/#get-an-organization
func (s *OrganizationsService) Get(org string) (*Organization, *Response, error) {
	u := fmt.Sprintf("orgs/%v", org)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	organization := new(Organization)
	resp, err := s.client.Do(req, organization)
	if err != nil {
		return nil, resp, err
	}

	return organization, resp, err
}

// Edit an organization.
//
// GitHub API docs: http://developer.github.com/v3/orgs/#edit-an-organization
func (s *OrganizationsService) Edit(name string, org *Organization) (*Organization, *Response, error) {
	u := fmt.Sprintf("orgs/%v", name)
	req, err := s.client.NewRequest("PATCH", u, org)
	if err != nil {
		return nil, nil, err
	}

	o := new(Organization)
	resp, err := s.client.Do(req, o)
	if err != nil {
		return nil, resp, err
	}

	return o, resp, err
}
