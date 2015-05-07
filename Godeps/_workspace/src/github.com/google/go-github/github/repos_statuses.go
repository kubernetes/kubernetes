// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// RepoStatus represents the status of a repository at a particular reference.
type RepoStatus struct {
	ID  *int    `json:"id,omitempty"`
	URL *string `json:"url,omitempty"`

	// State is the current state of the repository.  Possible values are:
	// pending, success, error, or failure.
	State *string `json:"state,omitempty"`

	// TargetURL is the URL of the page representing this status.  It will be
	// linked from the GitHub UI to allow users to see the source of the status.
	TargetURL *string `json:"target_url,omitempty"`

	// Description is a short high level summary of the status.
	Description *string `json:"description,omitempty"`

	// A string label to differentiate this status from the statuses of other systems.
	Context *string `json:"context,omitempty"`

	Creator   *User      `json:"creator,omitempty"`
	CreatedAt *time.Time `json:"created_at,omitempty"`
	UpdatedAt *time.Time `json:"updated_at,omitempty"`
}

func (r RepoStatus) String() string {
	return Stringify(r)
}

// ListStatuses lists the statuses of a repository at the specified
// reference.  ref can be a SHA, a branch name, or a tag name.
//
// GitHub API docs: http://developer.github.com/v3/repos/statuses/#list-statuses-for-a-specific-ref
func (s *RepositoriesService) ListStatuses(owner, repo, ref string, opt *ListOptions) ([]RepoStatus, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/commits/%v/statuses", owner, repo, ref)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	statuses := new([]RepoStatus)
	resp, err := s.client.Do(req, statuses)
	if err != nil {
		return nil, resp, err
	}

	return *statuses, resp, err
}

// CreateStatus creates a new status for a repository at the specified
// reference.  Ref can be a SHA, a branch name, or a tag name.
//
// GitHub API docs: http://developer.github.com/v3/repos/statuses/#create-a-status
func (s *RepositoriesService) CreateStatus(owner, repo, ref string, status *RepoStatus) (*RepoStatus, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/statuses/%v", owner, repo, ref)
	req, err := s.client.NewRequest("POST", u, status)
	if err != nil {
		return nil, nil, err
	}

	repoStatus := new(RepoStatus)
	resp, err := s.client.Do(req, repoStatus)
	if err != nil {
		return nil, resp, err
	}

	return repoStatus, resp, err
}

// CombinedStatus represents the combined status of a repository at a particular reference.
type CombinedStatus struct {
	// State is the combined state of the repository.  Possible values are:
	// failture, pending, or success.
	State *string `json:"state,omitempty"`

	Name       *string      `json:"name,omitempty"`
	SHA        *string      `json:"sha,omitempty"`
	TotalCount *int         `json:"total_count,omitempty"`
	Statuses   []RepoStatus `json:"statuses,omitempty"`

	CommitURL     *string `json:"commit_url,omitempty"`
	RepositoryURL *string `json:"repository_url,omitempty"`
}

func (s CombinedStatus) String() string {
	return Stringify(s)
}

// GetCombinedStatus returns the combined status of a repository at the specified
// reference.  ref can be a SHA, a branch name, or a tag name.
//
// GitHub API docs: https://developer.github.com/v3/repos/statuses/#get-the-combined-status-for-a-specific-ref
func (s *RepositoriesService) GetCombinedStatus(owner, repo, ref string, opt *ListOptions) (*CombinedStatus, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/commits/%v/status", owner, repo, ref)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	status := new(CombinedStatus)
	resp, err := s.client.Do(req, status)
	if err != nil {
		return nil, resp, err
	}

	return status, resp, err
}
