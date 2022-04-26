// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
	"time"
)

// IssueComment represents a comment left on an issue.
type IssueComment struct {
	ID        *int64     `json:"id,omitempty"`
	NodeID    *string    `json:"node_id,omitempty"`
	Body      *string    `json:"body,omitempty"`
	User      *User      `json:"user,omitempty"`
	Reactions *Reactions `json:"reactions,omitempty"`
	CreatedAt *time.Time `json:"created_at,omitempty"`
	UpdatedAt *time.Time `json:"updated_at,omitempty"`
	// AuthorAssociation is the comment author's relationship to the issue's repository.
	// Possible values are "COLLABORATOR", "CONTRIBUTOR", "FIRST_TIMER", "FIRST_TIME_CONTRIBUTOR", "MEMBER", "OWNER", or "NONE".
	AuthorAssociation *string `json:"author_association,omitempty"`
	URL               *string `json:"url,omitempty"`
	HTMLURL           *string `json:"html_url,omitempty"`
	IssueURL          *string `json:"issue_url,omitempty"`
}

func (i IssueComment) String() string {
	return Stringify(i)
}

// IssueListCommentsOptions specifies the optional parameters to the
// IssuesService.ListComments method.
type IssueListCommentsOptions struct {
	// Since filters comments by time.
	Since *time.Time `url:"since,omitempty"`

	ListOptions
}

// ListComments lists all comments on the specified issue. Specifying an issue
// number of 0 will return all comments on all issues for the repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#list-issue-comments
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#list-issue-comments-for-a-repository
func (s *IssuesService) ListComments(ctx context.Context, owner string, repo string, number int, opts *IssueListCommentsOptions) ([]*IssueComment, *Response, error) {
	var u string
	if number == 0 {
		u = fmt.Sprintf("repos/%v/%v/issues/comments", owner, repo)
	} else {
		u = fmt.Sprintf("repos/%v/%v/issues/%d/comments", owner, repo, number)
	}
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeReactionsPreview)

	var comments []*IssueComment
	resp, err := s.client.Do(ctx, req, &comments)
	if err != nil {
		return nil, resp, err
	}

	return comments, resp, nil
}

// GetComment fetches the specified issue comment.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#get-an-issue-comment
func (s *IssuesService) GetComment(ctx context.Context, owner string, repo string, commentID int64) (*IssueComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/comments/%d", owner, repo, commentID)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeReactionsPreview)

	comment := new(IssueComment)
	resp, err := s.client.Do(ctx, req, comment)
	if err != nil {
		return nil, resp, err
	}

	return comment, resp, nil
}

// CreateComment creates a new comment on the specified issue.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#create-an-issue-comment
func (s *IssuesService) CreateComment(ctx context.Context, owner string, repo string, number int, comment *IssueComment) (*IssueComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d/comments", owner, repo, number)
	req, err := s.client.NewRequest("POST", u, comment)
	if err != nil {
		return nil, nil, err
	}
	c := new(IssueComment)
	resp, err := s.client.Do(ctx, req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, nil
}

// EditComment updates an issue comment.
// A non-nil comment.Body must be provided. Other comment fields should be left nil.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#update-an-issue-comment
func (s *IssuesService) EditComment(ctx context.Context, owner string, repo string, commentID int64, comment *IssueComment) (*IssueComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/comments/%d", owner, repo, commentID)
	req, err := s.client.NewRequest("PATCH", u, comment)
	if err != nil {
		return nil, nil, err
	}
	c := new(IssueComment)
	resp, err := s.client.Do(ctx, req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, nil
}

// DeleteComment deletes an issue comment.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#delete-an-issue-comment
func (s *IssuesService) DeleteComment(ctx context.Context, owner string, repo string, commentID int64) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/comments/%d", owner, repo, commentID)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}
