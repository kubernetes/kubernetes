// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "fmt"

// UsersService handles communication with the user related
// methods of the GitHub API.
//
// GitHub API docs: http://developer.github.com/v3/users/
type UsersService struct {
	client *Client
}

// User represents a GitHub user.
type User struct {
	Login             *string    `json:"login,omitempty"`
	ID                *int       `json:"id,omitempty"`
	AvatarURL         *string    `json:"avatar_url,omitempty"`
	HTMLURL           *string    `json:"html_url,omitempty"`
	GravatarID        *string    `json:"gravatar_id,omitempty"`
	Name              *string    `json:"name,omitempty"`
	Company           *string    `json:"company,omitempty"`
	Blog              *string    `json:"blog,omitempty"`
	Location          *string    `json:"location,omitempty"`
	Email             *string    `json:"email,omitempty"`
	Hireable          *bool      `json:"hireable,omitempty"`
	Bio               *string    `json:"bio,omitempty"`
	PublicRepos       *int       `json:"public_repos,omitempty"`
	PublicGists       *int       `json:"public_gists,omitempty"`
	Followers         *int       `json:"followers,omitempty"`
	Following         *int       `json:"following,omitempty"`
	CreatedAt         *Timestamp `json:"created_at,omitempty"`
	UpdatedAt         *Timestamp `json:"updated_at,omitempty"`
	Type              *string    `json:"type,omitempty"`
	SiteAdmin         *bool      `json:"site_admin,omitempty"`
	TotalPrivateRepos *int       `json:"total_private_repos,omitempty"`
	OwnedPrivateRepos *int       `json:"owned_private_repos,omitempty"`
	PrivateGists      *int       `json:"private_gists,omitempty"`
	DiskUsage         *int       `json:"disk_usage,omitempty"`
	Collaborators     *int       `json:"collaborators,omitempty"`
	Plan              *Plan      `json:"plan,omitempty"`

	// API URLs
	URL               *string `json:"url,omitempty"`
	EventsURL         *string `json:"events_url,omitempty"`
	FollowingURL      *string `json:"following_url,omitempty"`
	FollowersURL      *string `json:"followers_url,omitempty"`
	GistsURL          *string `json:"gists_url,omitempty"`
	OrganizationsURL  *string `json:"organizations_url,omitempty"`
	ReceivedEventsURL *string `json:"received_events_url,omitempty"`
	ReposURL          *string `json:"repos_url,omitempty"`
	StarredURL        *string `json:"starred_url,omitempty"`
	SubscriptionsURL  *string `json:"subscriptions_url,omitempty"`

	// TextMatches is only populated from search results that request text matches
	// See: search.go and https://developer.github.com/v3/search/#text-match-metadata
	TextMatches []TextMatch `json:"text_matches,omitempty"`
}

func (u User) String() string {
	return Stringify(u)
}

// Get fetches a user.  Passing the empty string will fetch the authenticated
// user.
//
// GitHub API docs: http://developer.github.com/v3/users/#get-a-single-user
func (s *UsersService) Get(user string) (*User, *Response, error) {
	var u string
	if user != "" {
		u = fmt.Sprintf("users/%v", user)
	} else {
		u = "user"
	}
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	uResp := new(User)
	resp, err := s.client.Do(req, uResp)
	if err != nil {
		return nil, resp, err
	}

	return uResp, resp, err
}

// Edit the authenticated user.
//
// GitHub API docs: http://developer.github.com/v3/users/#update-the-authenticated-user
func (s *UsersService) Edit(user *User) (*User, *Response, error) {
	u := "user"
	req, err := s.client.NewRequest("PATCH", u, user)
	if err != nil {
		return nil, nil, err
	}

	uResp := new(User)
	resp, err := s.client.Do(req, uResp)
	if err != nil {
		return nil, resp, err
	}

	return uResp, resp, err
}

// UserListOptions specifies optional parameters to the UsersService.List
// method.
type UserListOptions struct {
	// ID of the last user seen
	Since int `url:"since,omitempty"`
}

// ListAll lists all GitHub users.
//
// GitHub API docs: http://developer.github.com/v3/users/#get-all-users
func (s *UsersService) ListAll(opt *UserListOptions) ([]User, *Response, error) {
	u, err := addOptions("users", opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	users := new([]User)
	resp, err := s.client.Do(req, users)
	if err != nil {
		return nil, resp, err
	}

	return *users, resp, err
}
