// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// PullRequestsService handles communication with the pull request related
// methods of the GitHub API.
//
// GitHub API docs: http://developer.github.com/v3/pulls/
type PullRequestsService struct {
	client *Client
}

// PullRequest represents a GitHub pull request on a repository.
type PullRequest struct {
	Number       *int       `json:"number,omitempty"`
	State        *string    `json:"state,omitempty"`
	Title        *string    `json:"title,omitempty"`
	Body         *string    `json:"body,omitempty"`
	CreatedAt    *time.Time `json:"created_at,omitempty"`
	UpdatedAt    *time.Time `json:"updated_at,omitempty"`
	ClosedAt     *time.Time `json:"closed_at,omitempty"`
	MergedAt     *time.Time `json:"merged_at,omitempty"`
	User         *User      `json:"user,omitempty"`
	Merged       *bool      `json:"merged,omitempty"`
	Mergeable    *bool      `json:"mergeable,omitempty"`
	MergedBy     *User      `json:"merged_by,omitempty"`
	Comments     *int       `json:"comments,omitempty"`
	Commits      *int       `json:"commits,omitempty"`
	Additions    *int       `json:"additions,omitempty"`
	Deletions    *int       `json:"deletions,omitempty"`
	ChangedFiles *int       `json:"changed_files,omitempty"`
	URL          *string    `json:"url,omitempty"`
	HTMLURL      *string    `json:"html_url,omitempty"`
	IssueURL     *string    `json:"issue_url,omitempty"`
	StatusesURL  *string    `json:"statuses_url,omitempty"`
	DiffURL      *string    `json:"diff_url,omitempty"`
	PatchURL     *string    `json:"patch_url,omitempty"`

	Head *PullRequestBranch `json:"head,omitempty"`
	Base *PullRequestBranch `json:"base,omitempty"`
}

func (p PullRequest) String() string {
	return Stringify(p)
}

// PullRequestBranch represents a base or head branch in a GitHub pull request.
type PullRequestBranch struct {
	Label *string     `json:"label,omitempty"`
	Ref   *string     `json:"ref,omitempty"`
	SHA   *string     `json:"sha,omitempty"`
	Repo  *Repository `json:"repo,omitempty"`
	User  *User       `json:"user,omitempty"`
}

// PullRequestListOptions specifies the optional parameters to the
// PullRequestsService.List method.
type PullRequestListOptions struct {
	// State filters pull requests based on their state.  Possible values are:
	// open, closed.  Default is "open".
	State string `url:"state,omitempty"`

	// Head filters pull requests by head user and branch name in the format of:
	// "user:ref-name".
	Head string `url:"head,omitempty"`

	// Base filters pull requests by base branch name.
	Base string `url:"base,omitempty"`

	// Sort specifies how to sort pull requests. Possible values are: created,
	// updated, popularity, long-running. Default is "created".
	Sort string `url:"sort,omitempty"`

	// Direction in which to sort pull requests. Possible values are: asc, desc.
	// If Sort is "created" or not specified, Default is "desc", otherwise Default
	// is "asc"
	Direction string `url:"direction,omitempty"`

	ListOptions
}

// List the pull requests for the specified repository.
//
// GitHub API docs: http://developer.github.com/v3/pulls/#list-pull-requests
func (s *PullRequestsService) List(owner string, repo string, opt *PullRequestListOptions) ([]PullRequest, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	pulls := new([]PullRequest)
	resp, err := s.client.Do(req, pulls)
	if err != nil {
		return nil, resp, err
	}

	return *pulls, resp, err
}

// Get a single pull request.
//
// GitHub API docs: https://developer.github.com/v3/pulls/#get-a-single-pull-request
func (s *PullRequestsService) Get(owner string, repo string, number int) (*PullRequest, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d", owner, repo, number)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	pull := new(PullRequest)
	resp, err := s.client.Do(req, pull)
	if err != nil {
		return nil, resp, err
	}

	return pull, resp, err
}

// NewPullRequest represents a new pull request to be created.
type NewPullRequest struct {
	Title *string `json:"title,omitempty"`
	Head  *string `json:"head,omitempty"`
	Base  *string `json:"base,omitempty"`
	Body  *string `json:"body,omitempty"`
	Issue *int    `json:"issue,omitempty"`
}

// Create a new pull request on the specified repository.
//
// GitHub API docs: https://developer.github.com/v3/pulls/#create-a-pull-request
func (s *PullRequestsService) Create(owner string, repo string, pull *NewPullRequest) (*PullRequest, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls", owner, repo)
	req, err := s.client.NewRequest("POST", u, pull)
	if err != nil {
		return nil, nil, err
	}

	p := new(PullRequest)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// Edit a pull request.
//
// GitHub API docs: https://developer.github.com/v3/pulls/#update-a-pull-request
func (s *PullRequestsService) Edit(owner string, repo string, number int, pull *PullRequest) (*PullRequest, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d", owner, repo, number)
	req, err := s.client.NewRequest("PATCH", u, pull)
	if err != nil {
		return nil, nil, err
	}

	p := new(PullRequest)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// ListCommits lists the commits in a pull request.
//
// GitHub API docs: https://developer.github.com/v3/pulls/#list-commits-on-a-pull-request
func (s *PullRequestsService) ListCommits(owner string, repo string, number int, opt *ListOptions) ([]RepositoryCommit, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/commits", owner, repo, number)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	commits := new([]RepositoryCommit)
	resp, err := s.client.Do(req, commits)
	if err != nil {
		return nil, resp, err
	}

	return *commits, resp, err
}

// ListFiles lists the files in a pull request.
//
// GitHub API docs: https://developer.github.com/v3/pulls/#list-pull-requests-files
func (s *PullRequestsService) ListFiles(owner string, repo string, number int, opt *ListOptions) ([]CommitFile, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/files", owner, repo, number)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	commitFiles := new([]CommitFile)
	resp, err := s.client.Do(req, commitFiles)
	if err != nil {
		return nil, resp, err
	}

	return *commitFiles, resp, err
}

// IsMerged checks if a pull request has been merged.
//
// GitHub API docs: https://developer.github.com/v3/pulls/#get-if-a-pull-request-has-been-merged
func (s *PullRequestsService) IsMerged(owner string, repo string, number int) (bool, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/merge", owner, repo, number)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return false, nil, err
	}

	resp, err := s.client.Do(req, nil)
	merged, err := parseBoolResponse(err)
	return merged, resp, err
}

// PullRequestMergeResult represents the result of merging a pull request.
type PullRequestMergeResult struct {
	SHA     *string `json:"sha,omitempty"`
	Merged  *bool   `json:"merged,omitempty"`
	Message *string `json:"message,omitempty"`
}

type pullRequestMergeRequest struct {
	CommitMessage *string `json:"commit_message"`
}

// Merge a pull request (Merge Button™).
//
// GitHub API docs: https://developer.github.com/v3/pulls/#merge-a-pull-request-merge-buttontrade
func (s *PullRequestsService) Merge(owner string, repo string, number int, commitMessage string) (*PullRequestMergeResult, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/merge", owner, repo, number)

	req, err := s.client.NewRequest("PUT", u, &pullRequestMergeRequest{
		CommitMessage: &commitMessage,
	})

	if err != nil {
		return nil, nil, err
	}

	mergeResult := new(PullRequestMergeResult)
	resp, err := s.client.Do(req, mergeResult)
	if err != nil {
		return nil, resp, err
	}

	return mergeResult, resp, err
}
