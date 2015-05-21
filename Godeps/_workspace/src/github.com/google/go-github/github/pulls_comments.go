// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// PullRequestComment represents a comment left on a pull request.
type PullRequestComment struct {
	ID        *int       `json:"id,omitempty"`
	Body      *string    `json:"body,omitempty"`
	Path      *string    `json:"path,omitempty"`
	Position  *int       `json:"position,omitempty"`
	CommitID  *string    `json:"commit_id,omitempty"`
	User      *User      `json:"user,omitempty"`
	CreatedAt *time.Time `json:"created_at,omitempty"`
	UpdatedAt *time.Time `json:"updated_at,omitempty"`
}

func (p PullRequestComment) String() string {
	return Stringify(p)
}

// PullRequestListCommentsOptions specifies the optional parameters to the
// PullRequestsService.ListComments method.
type PullRequestListCommentsOptions struct {
	// Sort specifies how to sort comments.  Possible values are: created, updated.
	Sort string `url:"sort,omitempty"`

	// Direction in which to sort comments.  Possible values are: asc, desc.
	Direction string `url:"direction,omitempty"`

	// Since filters comments by time.
	Since time.Time `url:"since,omitempty"`

	ListOptions
}

// ListComments lists all comments on the specified pull request.  Specifying a
// pull request number of 0 will return all comments on all pull requests for
// the repository.
//
// GitHub API docs: https://developer.github.com/v3/pulls/comments/#list-comments-on-a-pull-request
func (s *PullRequestsService) ListComments(owner string, repo string, number int, opt *PullRequestListCommentsOptions) ([]PullRequestComment, *Response, error) {
	var u string
	if number == 0 {
		u = fmt.Sprintf("repos/%v/%v/pulls/comments", owner, repo)
	} else {
		u = fmt.Sprintf("repos/%v/%v/pulls/%d/comments", owner, repo, number)
	}
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	comments := new([]PullRequestComment)
	resp, err := s.client.Do(req, comments)
	if err != nil {
		return nil, resp, err
	}

	return *comments, resp, err
}

// GetComment fetches the specified pull request comment.
//
// GitHub API docs: https://developer.github.com/v3/pulls/comments/#get-a-single-comment
func (s *PullRequestsService) GetComment(owner string, repo string, number int) (*PullRequestComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/comments/%d", owner, repo, number)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	comment := new(PullRequestComment)
	resp, err := s.client.Do(req, comment)
	if err != nil {
		return nil, resp, err
	}

	return comment, resp, err
}

// CreateComment creates a new comment on the specified pull request.
//
// GitHub API docs: https://developer.github.com/v3/pulls/comments/#get-a-single-comment
func (s *PullRequestsService) CreateComment(owner string, repo string, number int, comment *PullRequestComment) (*PullRequestComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/comments", owner, repo, number)
	req, err := s.client.NewRequest("POST", u, comment)
	if err != nil {
		return nil, nil, err
	}

	c := new(PullRequestComment)
	resp, err := s.client.Do(req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// EditComment updates a pull request comment.
//
// GitHub API docs: https://developer.github.com/v3/pulls/comments/#edit-a-comment
func (s *PullRequestsService) EditComment(owner string, repo string, number int, comment *PullRequestComment) (*PullRequestComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/comments/%d", owner, repo, number)
	req, err := s.client.NewRequest("PATCH", u, comment)
	if err != nil {
		return nil, nil, err
	}

	c := new(PullRequestComment)
	resp, err := s.client.Do(req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// DeleteComment deletes a pull request comment.
//
// GitHub API docs: https://developer.github.com/v3/pulls/comments/#delete-a-comment
func (s *PullRequestsService) DeleteComment(owner string, repo string, number int) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/comments/%d", owner, repo, number)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}
