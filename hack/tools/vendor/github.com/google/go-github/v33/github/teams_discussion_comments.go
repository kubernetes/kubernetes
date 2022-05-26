// Copyright 2018 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// DiscussionComment represents a GitHub dicussion in a team.
type DiscussionComment struct {
	Author        *User      `json:"author,omitempty"`
	Body          *string    `json:"body,omitempty"`
	BodyHTML      *string    `json:"body_html,omitempty"`
	BodyVersion   *string    `json:"body_version,omitempty"`
	CreatedAt     *Timestamp `json:"created_at,omitempty"`
	LastEditedAt  *Timestamp `json:"last_edited_at,omitempty"`
	DiscussionURL *string    `json:"discussion_url,omitempty"`
	HTMLURL       *string    `json:"html_url,omitempty"`
	NodeID        *string    `json:"node_id,omitempty"`
	Number        *int       `json:"number,omitempty"`
	UpdatedAt     *Timestamp `json:"updated_at,omitempty"`
	URL           *string    `json:"url,omitempty"`
	Reactions     *Reactions `json:"reactions,omitempty"`
}

func (c DiscussionComment) String() string {
	return Stringify(c)
}

// DiscussionCommentListOptions specifies optional parameters to the
// TeamServices.ListComments method.
type DiscussionCommentListOptions struct {
	// Sorts the discussion comments by the date they were created.
	// Accepted values are asc and desc. Default is desc.
	Direction string `url:"direction,omitempty"`
	ListOptions
}

// ListCommentsByID lists all comments on a team discussion by team ID.
// Authenticated user must grant read:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-discussion-comments
func (s *TeamsService) ListCommentsByID(ctx context.Context, orgID, teamID int64, discussionNumber int, options *DiscussionCommentListOptions) ([]*DiscussionComment, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/discussions/%v/comments", orgID, teamID, discussionNumber)
	u, err := addOptions(u, options)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var comments []*DiscussionComment
	resp, err := s.client.Do(ctx, req, &comments)
	if err != nil {
		return nil, resp, err
	}

	return comments, resp, nil
}

// ListCommentsBySlug lists all comments on a team discussion by team slug.
// Authenticated user must grant read:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-discussion-comments
func (s *TeamsService) ListCommentsBySlug(ctx context.Context, org, slug string, discussionNumber int, options *DiscussionCommentListOptions) ([]*DiscussionComment, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/discussions/%v/comments", org, slug, discussionNumber)
	u, err := addOptions(u, options)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var comments []*DiscussionComment
	resp, err := s.client.Do(ctx, req, &comments)
	if err != nil {
		return nil, resp, err
	}

	return comments, resp, nil
}

// GetCommentByID gets a specific comment on a team discussion by team ID.
// Authenticated user must grant read:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#get-a-discussion-comment
func (s *TeamsService) GetCommentByID(ctx context.Context, orgID, teamID int64, discussionNumber, commentNumber int) (*DiscussionComment, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/discussions/%v/comments/%v", orgID, teamID, discussionNumber, commentNumber)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	discussionComment := &DiscussionComment{}
	resp, err := s.client.Do(ctx, req, discussionComment)
	if err != nil {
		return nil, resp, err
	}

	return discussionComment, resp, nil
}

// GetCommentBySlug gets a specific comment on a team discussion by team slug.
// Authenticated user must grant read:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#get-a-discussion-comment
func (s *TeamsService) GetCommentBySlug(ctx context.Context, org, slug string, discussionNumber, commentNumber int) (*DiscussionComment, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/discussions/%v/comments/%v", org, slug, discussionNumber, commentNumber)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	discussionComment := &DiscussionComment{}
	resp, err := s.client.Do(ctx, req, discussionComment)
	if err != nil {
		return nil, resp, err
	}

	return discussionComment, resp, nil
}

// CreateCommentByID creates a new comment on a team discussion by team ID.
// Authenticated user must grant write:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#create-a-discussion-comment
func (s *TeamsService) CreateCommentByID(ctx context.Context, orgID, teamID int64, discsusionNumber int, comment DiscussionComment) (*DiscussionComment, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/discussions/%v/comments", orgID, teamID, discsusionNumber)
	req, err := s.client.NewRequest("POST", u, comment)
	if err != nil {
		return nil, nil, err
	}

	discussionComment := &DiscussionComment{}
	resp, err := s.client.Do(ctx, req, discussionComment)
	if err != nil {
		return nil, resp, err
	}

	return discussionComment, resp, nil
}

// CreateCommentBySlug creates a new comment on a team discussion by team slug.
// Authenticated user must grant write:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#create-a-discussion-comment
func (s *TeamsService) CreateCommentBySlug(ctx context.Context, org, slug string, discsusionNumber int, comment DiscussionComment) (*DiscussionComment, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/discussions/%v/comments", org, slug, discsusionNumber)
	req, err := s.client.NewRequest("POST", u, comment)
	if err != nil {
		return nil, nil, err
	}

	discussionComment := &DiscussionComment{}
	resp, err := s.client.Do(ctx, req, discussionComment)
	if err != nil {
		return nil, resp, err
	}

	return discussionComment, resp, nil
}

// EditCommentByID edits the body text of a discussion comment by team ID.
// Authenticated user must grant write:discussion scope.
// User is allowed to edit body of a comment only.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#update-a-discussion-comment
func (s *TeamsService) EditCommentByID(ctx context.Context, orgID, teamID int64, discussionNumber, commentNumber int, comment DiscussionComment) (*DiscussionComment, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/discussions/%v/comments/%v", orgID, teamID, discussionNumber, commentNumber)
	req, err := s.client.NewRequest("PATCH", u, comment)
	if err != nil {
		return nil, nil, err
	}

	discussionComment := &DiscussionComment{}
	resp, err := s.client.Do(ctx, req, discussionComment)
	if err != nil {
		return nil, resp, err
	}

	return discussionComment, resp, nil
}

// EditCommentBySlug edits the body text of a discussion comment by team slug.
// Authenticated user must grant write:discussion scope.
// User is allowed to edit body of a comment only.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#update-a-discussion-comment
func (s *TeamsService) EditCommentBySlug(ctx context.Context, org, slug string, discussionNumber, commentNumber int, comment DiscussionComment) (*DiscussionComment, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/discussions/%v/comments/%v", org, slug, discussionNumber, commentNumber)
	req, err := s.client.NewRequest("PATCH", u, comment)
	if err != nil {
		return nil, nil, err
	}

	discussionComment := &DiscussionComment{}
	resp, err := s.client.Do(ctx, req, discussionComment)
	if err != nil {
		return nil, resp, err
	}

	return discussionComment, resp, nil
}

// DeleteCommentByID deletes a comment on a team discussion by team ID.
// Authenticated user must grant write:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#delete-a-discussion-comment
func (s *TeamsService) DeleteCommentByID(ctx context.Context, orgID, teamID int64, discussionNumber, commentNumber int) (*Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/discussions/%v/comments/%v", orgID, teamID, discussionNumber, commentNumber)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// DeleteCommentBySlug deletes a comment on a team discussion by team slug.
// Authenticated user must grant write:discussion scope.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#delete-a-discussion-comment
func (s *TeamsService) DeleteCommentBySlug(ctx context.Context, org, slug string, discussionNumber, commentNumber int) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/discussions/%v/comments/%v", org, slug, discussionNumber, commentNumber)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
