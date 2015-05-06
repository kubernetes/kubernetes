// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// GistComment represents a Gist comment.
type GistComment struct {
	ID        *int       `json:"id,omitempty"`
	URL       *string    `json:"url,omitempty"`
	Body      *string    `json:"body,omitempty"`
	User      *User      `json:"user,omitempty"`
	CreatedAt *time.Time `json:"created_at,omitempty"`
}

func (g GistComment) String() string {
	return Stringify(g)
}

// ListComments lists all comments for a gist.
//
// GitHub API docs: http://developer.github.com/v3/gists/comments/#list-comments-on-a-gist
func (s *GistsService) ListComments(gistID string, opt *ListOptions) ([]GistComment, *Response, error) {
	u := fmt.Sprintf("gists/%v/comments", gistID)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	comments := new([]GistComment)
	resp, err := s.client.Do(req, comments)
	if err != nil {
		return nil, resp, err
	}

	return *comments, resp, err
}

// GetComment retrieves a single comment from a gist.
//
// GitHub API docs: http://developer.github.com/v3/gists/comments/#get-a-single-comment
func (s *GistsService) GetComment(gistID string, commentID int) (*GistComment, *Response, error) {
	u := fmt.Sprintf("gists/%v/comments/%v", gistID, commentID)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	c := new(GistComment)
	resp, err := s.client.Do(req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// CreateComment creates a comment for a gist.
//
// GitHub API docs: http://developer.github.com/v3/gists/comments/#create-a-comment
func (s *GistsService) CreateComment(gistID string, comment *GistComment) (*GistComment, *Response, error) {
	u := fmt.Sprintf("gists/%v/comments", gistID)
	req, err := s.client.NewRequest("POST", u, comment)
	if err != nil {
		return nil, nil, err
	}

	c := new(GistComment)
	resp, err := s.client.Do(req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// EditComment edits an existing gist comment.
//
// GitHub API docs: http://developer.github.com/v3/gists/comments/#edit-a-comment
func (s *GistsService) EditComment(gistID string, commentID int, comment *GistComment) (*GistComment, *Response, error) {
	u := fmt.Sprintf("gists/%v/comments/%v", gistID, commentID)
	req, err := s.client.NewRequest("PATCH", u, comment)
	if err != nil {
		return nil, nil, err
	}

	c := new(GistComment)
	resp, err := s.client.Do(req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// DeleteComment deletes a gist comment.
//
// GitHub API docs: http://developer.github.com/v3/gists/comments/#delete-a-comment
func (s *GistsService) DeleteComment(gistID string, commentID int) (*Response, error) {
	u := fmt.Sprintf("gists/%v/comments/%v", gistID, commentID)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
