// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// PullRequestComment represents a comment left on a pull request.
type PullRequestComment struct {
	ID                  *int64     `json:"id,omitempty"`
	NodeID              *string    `json:"node_id,omitempty"`
	InReplyTo           *int64     `json:"in_reply_to_id,omitempty"`
	Body                *string    `json:"body,omitempty"`
	Path                *string    `json:"path,omitempty"`
	DiffHunk            *string    `json:"diff_hunk,omitempty"`
	PullRequestReviewID *int64     `json:"pull_request_review_id,omitempty"`
	Position            *int       `json:"position,omitempty"`
	OriginalPosition    *int       `json:"original_position,omitempty"`
	StartLine           *int       `json:"start_line,omitempty"`
	Line                *int       `json:"line,omitempty"`
	OriginalLine        *int       `json:"original_line,omitempty"`
	OriginalStartLine   *int       `json:"original_start_line,omitempty"`
	Side                *string    `json:"side,omitempty"`
	StartSide           *string    `json:"start_side,omitempty"`
	CommitID            *string    `json:"commit_id,omitempty"`
	OriginalCommitID    *string    `json:"original_commit_id,omitempty"`
	User                *User      `json:"user,omitempty"`
	Reactions           *Reactions `json:"reactions,omitempty"`
	CreatedAt           *time.Time `json:"created_at,omitempty"`
	UpdatedAt           *time.Time `json:"updated_at,omitempty"`
	// AuthorAssociation is the comment author's relationship to the pull request's repository.
	// Possible values are "COLLABORATOR", "CONTRIBUTOR", "FIRST_TIMER", "FIRST_TIME_CONTRIBUTOR", "MEMBER", "OWNER", or "NONE".
	AuthorAssociation *string `json:"author_association,omitempty"`
	URL               *string `json:"url,omitempty"`
	HTMLURL           *string `json:"html_url,omitempty"`
	PullRequestURL    *string `json:"pull_request_url,omitempty"`
}

func (p PullRequestComment) String() string {
	return Stringify(p)
}

// PullRequestListCommentsOptions specifies the optional parameters to the
// PullRequestsService.ListComments method.
type PullRequestListCommentsOptions struct {
	// Sort specifies how to sort comments. Possible values are: created, updated.
	Sort string `url:"sort,omitempty"`

	// Direction in which to sort comments. Possible values are: asc, desc.
	Direction string `url:"direction,omitempty"`

	// Since filters comments by time.
	Since time.Time `url:"since,omitempty"`

	ListOptions
}

// ListComments lists all comments on the specified pull request. Specifying a
// pull request number of 0 will return all comments on all pull requests for
// the repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#list-review-comments-on-a-pull-request
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#list-review-comments-in-a-repository
func (s *PullRequestsService) ListComments(ctx context.Context, owner string, repo string, number int, opts *PullRequestListCommentsOptions) ([]*PullRequestComment, *Response, error) {
	var u string
	if number == 0 {
		u = fmt.Sprintf("repos/%v/%v/pulls/comments", owner, repo)
	} else {
		u = fmt.Sprintf("repos/%v/%v/pulls/%d/comments", owner, repo, number)
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
	acceptHeaders := []string{mediaTypeReactionsPreview, mediaTypeMultiLineCommentsPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	var comments []*PullRequestComment
	resp, err := s.client.Do(ctx, req, &comments)
	if err != nil {
		return nil, resp, err
	}

	return comments, resp, nil
}

// GetComment fetches the specified pull request comment.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#get-a-review-comment-for-a-pull-request
func (s *PullRequestsService) GetComment(ctx context.Context, owner string, repo string, commentID int64) (*PullRequestComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/comments/%d", owner, repo, commentID)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	acceptHeaders := []string{mediaTypeReactionsPreview, mediaTypeMultiLineCommentsPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	comment := new(PullRequestComment)
	resp, err := s.client.Do(ctx, req, comment)
	if err != nil {
		return nil, resp, err
	}

	return comment, resp, nil
}

// CreateComment creates a new comment on the specified pull request.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#create-a-review-comment-for-a-pull-request
func (s *PullRequestsService) CreateComment(ctx context.Context, owner string, repo string, number int, comment *PullRequestComment) (*PullRequestComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/comments", owner, repo, number)
	req, err := s.client.NewRequest("POST", u, comment)
	if err != nil {
		return nil, nil, err
	}
	// TODO: remove custom Accept headers when their respective API fully launches.
	acceptHeaders := []string{mediaTypeReactionsPreview, mediaTypeMultiLineCommentsPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	c := new(PullRequestComment)
	resp, err := s.client.Do(ctx, req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, nil
}

// CreateCommentInReplyTo creates a new comment as a reply to an existing pull request comment.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#create-a-review-comment-for-a-pull-request
func (s *PullRequestsService) CreateCommentInReplyTo(ctx context.Context, owner string, repo string, number int, body string, commentID int64) (*PullRequestComment, *Response, error) {
	comment := &struct {
		Body      string `json:"body,omitempty"`
		InReplyTo int64  `json:"in_reply_to,omitempty"`
	}{
		Body:      body,
		InReplyTo: commentID,
	}
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/comments", owner, repo, number)
	req, err := s.client.NewRequest("POST", u, comment)
	if err != nil {
		return nil, nil, err
	}

	c := new(PullRequestComment)
	resp, err := s.client.Do(ctx, req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, nil
}

// EditComment updates a pull request comment.
// A non-nil comment.Body must be provided. Other comment fields should be left nil.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#update-a-review-comment-for-a-pull-request
func (s *PullRequestsService) EditComment(ctx context.Context, owner string, repo string, commentID int64, comment *PullRequestComment) (*PullRequestComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/comments/%d", owner, repo, commentID)
	req, err := s.client.NewRequest("PATCH", u, comment)
	if err != nil {
		return nil, nil, err
	}

	c := new(PullRequestComment)
	resp, err := s.client.Do(ctx, req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, nil
}

// DeleteComment deletes a pull request comment.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#delete-a-review-comment-for-a-pull-request
func (s *PullRequestsService) DeleteComment(ctx context.Context, owner string, repo string, commentID int64) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/comments/%d", owner, repo, commentID)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}
