// Copyright 2014 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// RepositoryMergeRequest represents a request to merge a branch in a
// repository.
type RepositoryMergeRequest struct {
	Base          *string `json:"base,omitempty"`
	Head          *string `json:"head,omitempty"`
	CommitMessage *string `json:"commit_message,omitempty"`
}

// Merge a branch in the specified repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#merge-a-branch
func (s *RepositoriesService) Merge(ctx context.Context, owner, repo string, request *RepositoryMergeRequest) (*RepositoryCommit, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/merges", owner, repo)
	req, err := s.client.NewRequest("POST", u, request)
	if err != nil {
		return nil, nil, err
	}

	commit := new(RepositoryCommit)
	resp, err := s.client.Do(ctx, req, commit)
	if err != nil {
		return nil, resp, err
	}

	return commit, resp, nil
}
