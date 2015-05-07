// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// RepositoryCommit represents a commit in a repo.
// Note that it's wrapping a Commit, so author/committer information is in two places,
// but contain different details about them: in RepositoryCommit "github details", in Commit - "git details".
type RepositoryCommit struct {
	SHA       *string  `json:"sha,omitempty"`
	Commit    *Commit  `json:"commit,omitempty"`
	Author    *User    `json:"author,omitempty"`
	Committer *User    `json:"committer,omitempty"`
	Parents   []Commit `json:"parents,omitempty"`
	Message   *string  `json:"message,omitempty"`
	HTMLURL   *string  `json:"html_url,omitempty"`

	// Details about how many changes were made in this commit. Only filled in during GetCommit!
	Stats *CommitStats `json:"stats,omitempty"`
	// Details about which files, and how this commit touched. Only filled in during GetCommit!
	Files []CommitFile `json:"files,omitempty"`
}

func (r RepositoryCommit) String() string {
	return Stringify(r)
}

// CommitStats represents the number of additions / deletions from a file in a given RepositoryCommit.
type CommitStats struct {
	Additions *int `json:"additions,omitempty"`
	Deletions *int `json:"deletions,omitempty"`
	Total     *int `json:"total,omitempty"`
}

func (c CommitStats) String() string {
	return Stringify(c)
}

// CommitFile represents a file modified in a commit.
type CommitFile struct {
	SHA       *string `json:"sha,omitempty"`
	Filename  *string `json:"filename,omitempty"`
	Additions *int    `json:"additions,omitempty"`
	Deletions *int    `json:"deletions,omitempty"`
	Changes   *int    `json:"changes,omitempty"`
	Status    *string `json:"status,omitempty"`
	Patch     *string `json:"patch,omitempty"`
}

func (c CommitFile) String() string {
	return Stringify(c)
}

// CommitsComparison is the result of comparing two commits.
// See CompareCommits() for details.
type CommitsComparison struct {
	BaseCommit      *RepositoryCommit `json:"base_commit,omitempty"`
	MergeBaseCommit *RepositoryCommit `json:"merge_base_commit,omitempty"`

	// Head can be 'behind' or 'ahead'
	Status       *string `json:"status,omitempty"`
	AheadBy      *int    `json:"ahead_by,omitempty"`
	BehindBy     *int    `json:"behind_by,omitempty"`
	TotalCommits *int    `json:"total_commits,omitempty"`

	Commits []RepositoryCommit `json:"commits,omitempty"`

	Files []CommitFile `json:"files,omitempty"`
}

func (c CommitsComparison) String() string {
	return Stringify(c)
}

// CommitsListOptions specifies the optional parameters to the
// RepositoriesService.ListCommits method.
type CommitsListOptions struct {
	// SHA or branch to start listing Commits from.
	SHA string `url:"sha,omitempty"`

	// Path that should be touched by the returned Commits.
	Path string `url:"path,omitempty"`

	// Author of by which to filter Commits.
	Author string `url:"author,omitempty"`

	// Since when should Commits be included in the response.
	Since time.Time `url:"since,omitempty"`

	// Until when should Commits be included in the response.
	Until time.Time `url:"until,omitempty"`

	ListOptions
}

// ListCommits lists the commits of a repository.
//
// GitHub API docs: http://developer.github.com/v3/repos/commits/#list
func (s *RepositoriesService) ListCommits(owner, repo string, opt *CommitsListOptions) ([]RepositoryCommit, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/commits", owner, repo)
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

// GetCommit fetches the specified commit, including all details about it.
// todo: support media formats - https://github.com/google/go-github/issues/6
//
// GitHub API docs: http://developer.github.com/v3/repos/commits/#get-a-single-commit
// See also: http://developer.github.com//v3/git/commits/#get-a-single-commit provides the same functionality
func (s *RepositoriesService) GetCommit(owner, repo, sha string) (*RepositoryCommit, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/commits/%v", owner, repo, sha)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	commit := new(RepositoryCommit)
	resp, err := s.client.Do(req, commit)
	if err != nil {
		return nil, resp, err
	}

	return commit, resp, err
}

// CompareCommits compares a range of commits with each other.
// todo: support media formats - https://github.com/google/go-github/issues/6
//
// GitHub API docs: http://developer.github.com/v3/repos/commits/index.html#compare-two-commits
func (s *RepositoriesService) CompareCommits(owner, repo string, base, head string) (*CommitsComparison, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/compare/%v...%v", owner, repo, base, head)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	comp := new(CommitsComparison)
	resp, err := s.client.Do(req, comp)
	if err != nil {
		return nil, resp, err
	}

	return comp, resp, err
}
