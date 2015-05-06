// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// IssueComment represents a comment left on an issue.
type IssueComment struct {
	ID        *int       `json:"id,omitempty"`
	Body      *string    `json:"body,omitempty"`
	User      *User      `json:"user,omitempty"`
	CreatedAt *time.Time `json:"created_at,omitempty"`
	UpdatedAt *time.Time `json:"updated_at,omitempty"`
	URL       *string    `json:"url,omitempty"`
	HTMLURL   *string    `json:"html_url,omitempty"`
	IssueURL  *string    `json:"issue_url,omitempty"`
}

func (i IssueComment) String() string {
	return Stringify(i)
}

// IssueListCommentsOptions specifies the optional parameters to the
// IssuesService.ListComments method.
type IssueListCommentsOptions struct {
	// Sort specifies how to sort comments.  Possible values are: created, updated.
	Sort string `url:"sort,omitempty"`

	// Direction in which to sort comments.  Possible values are: asc, desc.
	Direction string `url:"direction,omitempty"`

	// Since filters comments by time.
	Since time.Time `url:"since,omitempty"`

	ListOptions
}

// ListComments lists all comments on the specified issue.  Specifying an issue
// number of 0 will return all comments on all issues for the repository.
//
// GitHub API docs: http://developer.github.com/v3/issues/comments/#list-comments-on-an-issue
func (s *IssuesService) ListComments(owner string, repo string, number int, opt *IssueListCommentsOptions) ([]IssueComment, *Response, error) {
	var u string
	if number == 0 {
		u = fmt.Sprintf("repos/%v/%v/issues/comments", owner, repo)
	} else {
		u = fmt.Sprintf("repos/%v/%v/issues/%d/comments", owner, repo, number)
	}
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}
	comments := new([]IssueComment)
	resp, err := s.client.Do(req, comments)
	if err != nil {
		return nil, resp, err
	}

	return *comments, resp, err
}

// GetComment fetches the specified issue comment.
//
// GitHub API docs: http://developer.github.com/v3/issues/comments/#get-a-single-comment
func (s *IssuesService) GetComment(owner string, repo string, id int) (*IssueComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/comments/%d", owner, repo, id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}
	comment := new(IssueComment)
	resp, err := s.client.Do(req, comment)
	if err != nil {
		return nil, resp, err
	}

	return comment, resp, err
}

// CreateComment creates a new comment on the specified issue.
//
// GitHub API docs: http://developer.github.com/v3/issues/comments/#create-a-comment
func (s *IssuesService) CreateComment(owner string, repo string, number int, comment *IssueComment) (*IssueComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d/comments", owner, repo, number)
	req, err := s.client.NewRequest("POST", u, comment)
	if err != nil {
		return nil, nil, err
	}
	c := new(IssueComment)
	resp, err := s.client.Do(req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// EditComment updates an issue comment.
//
// GitHub API docs: http://developer.github.com/v3/issues/comments/#edit-a-comment
func (s *IssuesService) EditComment(owner string, repo string, id int, comment *IssueComment) (*IssueComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/comments/%d", owner, repo, id)
	req, err := s.client.NewRequest("PATCH", u, comment)
	if err != nil {
		return nil, nil, err
	}
	c := new(IssueComment)
	resp, err := s.client.Do(req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// DeleteComment deletes an issue comment.
//
// GitHub API docs: http://developer.github.com/v3/issues/comments/#delete-a-comment
func (s *IssuesService) DeleteComment(owner string, repo string, id int) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/comments/%d", owner, repo, id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}
