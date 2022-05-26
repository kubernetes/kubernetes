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

// RepositoryComment represents a comment for a commit, file, or line in a repository.
type RepositoryComment struct {
	HTMLURL   *string    `json:"html_url,omitempty"`
	URL       *string    `json:"url,omitempty"`
	ID        *int64     `json:"id,omitempty"`
	NodeID    *string    `json:"node_id,omitempty"`
	CommitID  *string    `json:"commit_id,omitempty"`
	User      *User      `json:"user,omitempty"`
	Reactions *Reactions `json:"reactions,omitempty"`
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
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#list-commit-comments-for-a-repository
func (s *RepositoriesService) ListComments(ctx context.Context, owner, repo string, opts *ListOptions) ([]*RepositoryComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/comments", owner, repo)
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

	var comments []*RepositoryComment
	resp, err := s.client.Do(ctx, req, &comments)
	if err != nil {
		return nil, resp, err
	}

	return comments, resp, nil
}

// ListCommitComments lists all the comments for a given commit SHA.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#list-commit-comments
func (s *RepositoriesService) ListCommitComments(ctx context.Context, owner, repo, sha string, opts *ListOptions) ([]*RepositoryComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/commits/%v/comments", owner, repo, sha)
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

	var comments []*RepositoryComment
	resp, err := s.client.Do(ctx, req, &comments)
	if err != nil {
		return nil, resp, err
	}

	return comments, resp, nil
}

// CreateComment creates a comment for the given commit.
// Note: GitHub allows for comments to be created for non-existing files and positions.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#create-a-commit-comment
func (s *RepositoriesService) CreateComment(ctx context.Context, owner, repo, sha string, comment *RepositoryComment) (*RepositoryComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/commits/%v/comments", owner, repo, sha)
	req, err := s.client.NewRequest("POST", u, comment)
	if err != nil {
		return nil, nil, err
	}

	c := new(RepositoryComment)
	resp, err := s.client.Do(ctx, req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, nil
}

// GetComment gets a single comment from a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-a-commit-comment
func (s *RepositoriesService) GetComment(ctx context.Context, owner, repo string, id int64) (*RepositoryComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/comments/%v", owner, repo, id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeReactionsPreview)

	c := new(RepositoryComment)
	resp, err := s.client.Do(ctx, req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, nil
}

// UpdateComment updates the body of a single comment.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#update-a-commit-comment
func (s *RepositoriesService) UpdateComment(ctx context.Context, owner, repo string, id int64, comment *RepositoryComment) (*RepositoryComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/comments/%v", owner, repo, id)
	req, err := s.client.NewRequest("PATCH", u, comment)
	if err != nil {
		return nil, nil, err
	}

	c := new(RepositoryComment)
	resp, err := s.client.Do(ctx, req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, nil
}

// DeleteComment deletes a single comment from a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#delete-a-commit-comment
func (s *RepositoriesService) DeleteComment(ctx context.Context, owner, repo string, id int64) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/comments/%v", owner, repo, id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}
