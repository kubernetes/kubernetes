// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// RepositoryComment represents a comment for a commit, file, or line in a repository.
type RepositoryComment struct {
	HTMLURL   *string    `json:"html_url,omitempty"`
	URL       *string    `json:"url,omitempty"`
	ID        *int       `json:"id,omitempty"`
	CommitID  *string    `json:"commit_id,omitempty"`
	User      *User      `json:"user,omitempty"`
	CreatedAt *time.Time `json:"created_at,omitempty"`
	UpdatedAt *time.Time `json:"updated_at,omitempty"`

	// User-mutable fields
	Body *string `json:"body"`
	// User-initialized fields
	Path     *string `json:"path,omitempty"`
	Position *int    `json:"position,omitempty"`
}

func (r RepositoryComment) String() string {
	return Stringify(r)
}

// ListComments lists all the comments for the repository.
//
// GitHub API docs: http://developer.github.com/v3/repos/comments/#list-commit-comments-for-a-repository
func (s *RepositoriesService) ListComments(owner, repo string, opt *ListOptions) ([]RepositoryComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/comments", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	comments := new([]RepositoryComment)
	resp, err := s.client.Do(req, comments)
	if err != nil {
		return nil, resp, err
	}

	return *comments, resp, err
}

// ListCommitComments lists all the comments for a given commit SHA.
//
// GitHub API docs: http://developer.github.com/v3/repos/comments/#list-comments-for-a-single-commit
func (s *RepositoriesService) ListCommitComments(owner, repo, sha string, opt *ListOptions) ([]RepositoryComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/commits/%v/comments", owner, repo, sha)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	comments := new([]RepositoryComment)
	resp, err := s.client.Do(req, comments)
	if err != nil {
		return nil, resp, err
	}

	return *comments, resp, err
}

// CreateComment creates a comment for the given commit.
// Note: GitHub allows for comments to be created for non-existing files and positions.
//
// GitHub API docs: http://developer.github.com/v3/repos/comments/#create-a-commit-comment
func (s *RepositoriesService) CreateComment(owner, repo, sha string, comment *RepositoryComment) (*RepositoryComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/commits/%v/comments", owner, repo, sha)
	req, err := s.client.NewRequest("POST", u, comment)
	if err != nil {
		return nil, nil, err
	}

	c := new(RepositoryComment)
	resp, err := s.client.Do(req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// GetComment gets a single comment from a repository.
//
// GitHub API docs: http://developer.github.com/v3/repos/comments/#get-a-single-commit-comment
func (s *RepositoriesService) GetComment(owner, repo string, id int) (*RepositoryComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/comments/%v", owner, repo, id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	c := new(RepositoryComment)
	resp, err := s.client.Do(req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// UpdateComment updates the body of a single comment.
//
// GitHub API docs: http://developer.github.com/v3/repos/comments/#update-a-commit-comment
func (s *RepositoriesService) UpdateComment(owner, repo string, id int, comment *RepositoryComment) (*RepositoryComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/comments/%v", owner, repo, id)
	req, err := s.client.NewRequest("PATCH", u, comment)
	if err != nil {
		return nil, nil, err
	}

	c := new(RepositoryComment)
	resp, err := s.client.Do(req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// DeleteComment deletes a single comment from a repository.
//
// GitHub API docs: http://developer.github.com/v3/repos/comments/#delete-a-commit-comment
func (s *RepositoriesService) DeleteComment(owner, repo string, id int) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/comments/%v", owner, repo, id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}
