// Copyright 2018 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// TeamDiscussion represents a GitHub dicussion in a team.
type TeamDiscussion struct {
	Author        *User      `json:"author,omitempty"`
	Body          *string    `json:"body,omitempty"`
	BodyHTML      *string    `json:"body_html,omitempty"`
	BodyVersion   *string    `json:"body_version,omitempty"`
	CommentsCount *int       `json:"comments_count,omitempty"`
	CommentsURL   *string    `json:"comments_url,omitempty"`
	CreatedAt     *Timestamp `json:"created_at,omitempty"`
	LastEditedAt  *Timestamp `json:"last_edited_at,omitempty"`
	HTMLURL       *string    `json:"html_url,omitempty"`
	NodeID        *string    `json:"node_id,omitempty"`
	Number        *int       `json:"number,omitempty"`
	Pinned        *bool      `json:"pinned,omitempty"`
	Private       *bool      `json:"private,omitempty"`
	TeamURL       *string    `json:"team_url,omitempty"`
	Title         *string    `json:"title,omitempty"`
	UpdatedAt     *Timestamp `json:"updated_at,omitempty"`
	URL           *string    `json:"url,omitempty"`
	Reactions     *Reactions `json:"reactions,omitempty"`
}

func (d TeamDiscussion) String() string {
	return Stringify(d)
}

// DiscussionListOptions specifies optional parameters to the
// TeamServices.ListDiscussions method.
type DiscussionListOptions struct {
	// Sorts the discussion by the date they were created.
	// Accepted values are asc and desc. Default is desc.
	Direction string `url:"direction,omitempty"`

	ListOptions
}

// ListDiscussionsByID lists all discussions on team's page given Organization and Team ID.
// Authenticated user must grant read:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-discussions
func (s *TeamsService) ListDiscussionsByID(ctx context.Context, orgID, teamID int64, opts *DiscussionListOptions) ([]*TeamDiscussion, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/discussions", orgID, teamID)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var teamDiscussions []*TeamDiscussion
	resp, err := s.client.Do(ctx, req, &teamDiscussions)
	if err != nil {
		return nil, resp, err
	}

	return teamDiscussions, resp, nil
}

// ListDiscussionsBySlug lists all discussions on team's page given Organization name and Team's slug.
// Authenticated user must grant read:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-discussions
func (s *TeamsService) ListDiscussionsBySlug(ctx context.Context, org, slug string, opts *DiscussionListOptions) ([]*TeamDiscussion, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/discussions", org, slug)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var teamDiscussions []*TeamDiscussion
	resp, err := s.client.Do(ctx, req, &teamDiscussions)
	if err != nil {
		return nil, resp, err
	}

	return teamDiscussions, resp, nil
}

// GetDiscussionByID gets a specific discussion on a team's page given Organization and Team ID.
// Authenticated user must grant read:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#get-a-discussion
func (s *TeamsService) GetDiscussionByID(ctx context.Context, orgID, teamID int64, discussionNumber int) (*TeamDiscussion, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/discussions/%v", orgID, teamID, discussionNumber)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	teamDiscussion := &TeamDiscussion{}
	resp, err := s.client.Do(ctx, req, teamDiscussion)
	if err != nil {
		return nil, resp, err
	}

	return teamDiscussion, resp, nil
}

// GetDiscussionBySlug gets a specific discussion on a team's page given Organization name and Team's slug.
// Authenticated user must grant read:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#get-a-discussion
func (s *TeamsService) GetDiscussionBySlug(ctx context.Context, org, slug string, discussionNumber int) (*TeamDiscussion, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/discussions/%v", org, slug, discussionNumber)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	teamDiscussion := &TeamDiscussion{}
	resp, err := s.client.Do(ctx, req, teamDiscussion)
	if err != nil {
		return nil, resp, err
	}

	return teamDiscussion, resp, nil
}

// CreateDiscussionByID creates a new discussion post on a team's page given Organization and Team ID.
// Authenticated user must grant write:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#create-a-discussion
func (s *TeamsService) CreateDiscussionByID(ctx context.Context, orgID, teamID int64, discussion TeamDiscussion) (*TeamDiscussion, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/discussions", orgID, teamID)
	req, err := s.client.NewRequest("POST", u, discussion)
	if err != nil {
		return nil, nil, err
	}

	teamDiscussion := &TeamDiscussion{}
	resp, err := s.client.Do(ctx, req, teamDiscussion)
	if err != nil {
		return nil, resp, err
	}

	return teamDiscussion, resp, nil
}

// CreateDiscussionBySlug creates a new discussion post on a team's page given Organization name and Team's slug.
// Authenticated user must grant write:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#create-a-discussion
func (s *TeamsService) CreateDiscussionBySlug(ctx context.Context, org, slug string, discussion TeamDiscussion) (*TeamDiscussion, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/discussions", org, slug)
	req, err := s.client.NewRequest("POST", u, discussion)
	if err != nil {
		return nil, nil, err
	}

	teamDiscussion := &TeamDiscussion{}
	resp, err := s.client.Do(ctx, req, teamDiscussion)
	if err != nil {
		return nil, resp, err
	}

	return teamDiscussion, resp, nil
}

// EditDiscussionByID edits the title and body text of a discussion post given Organization and Team ID.
// Authenticated user must grant write:discussion scope.
// User is allowed to change Title and Body of a discussion only.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#update-a-discussion
func (s *TeamsService) EditDiscussionByID(ctx context.Context, orgID, teamID int64, discussionNumber int, discussion TeamDiscussion) (*TeamDiscussion, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/discussions/%v", orgID, teamID, discussionNumber)
	req, err := s.client.NewRequest("PATCH", u, discussion)
	if err != nil {
		return nil, nil, err
	}

	teamDiscussion := &TeamDiscussion{}
	resp, err := s.client.Do(ctx, req, teamDiscussion)
	if err != nil {
		return nil, resp, err
	}

	return teamDiscussion, resp, nil
}

// EditDiscussionBySlug edits the title and body text of a discussion post given Organization name and Team's slug.
// Authenticated user must grant write:discussion scope.
// User is allowed to change Title and Body of a discussion only.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#update-a-discussion
func (s *TeamsService) EditDiscussionBySlug(ctx context.Context, org, slug string, discussionNumber int, discussion TeamDiscussion) (*TeamDiscussion, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/discussions/%v", org, slug, discussionNumber)
	req, err := s.client.NewRequest("PATCH", u, discussion)
	if err != nil {
		return nil, nil, err
	}

	teamDiscussion := &TeamDiscussion{}
	resp, err := s.client.Do(ctx, req, teamDiscussion)
	if err != nil {
		return nil, resp, err
	}

	return teamDiscussion, resp, nil
}

// DeleteDiscussionByID deletes a discussion from team's page given Organization and Team ID.
// Authenticated user must grant write:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#delete-a-discussion
func (s *TeamsService) DeleteDiscussionByID(ctx context.Context, orgID, teamID int64, discussionNumber int) (*Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/discussions/%v", orgID, teamID, discussionNumber)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// DeleteDiscussionBySlug deletes a discussion from team's page given Organization name and Team's slug.
// Authenticated user must grant write:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#delete-a-discussion
func (s *TeamsService) DeleteDiscussionBySlug(ctx context.Context, org, slug string, discussionNumber int) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/discussions/%v", org, slug, discussionNumber)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
