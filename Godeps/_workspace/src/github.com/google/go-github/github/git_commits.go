// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// Commit represents a GitHub commit.
type Commit struct {
	SHA       *string       `json:"sha,omitempty"`
	Author    *CommitAuthor `json:"author,omitempty"`
	Committer *CommitAuthor `json:"committer,omitempty"`
	Message   *string       `json:"message,omitempty"`
	Tree      *Tree         `json:"tree,omitempty"`
	Parents   []Commit      `json:"parents,omitempty"`
	Stats     *CommitStats  `json:"stats,omitempty"`
	URL       *string       `json:"url,omitempty"`

	// CommentCount is the number of GitHub comments on the commit.  This
	// is only populated for requests that fetch GitHub data like
	// Pulls.ListCommits, Repositories.ListCommits, etc.
	CommentCount *int `json:"comment_count,omitempty"`
}

func (c Commit) String() string {
	return Stringify(c)
}

// CommitAuthor represents the author or committer of a commit.  The commit
// author may not correspond to a GitHub User.
type CommitAuthor struct {
	Date  *time.Time `json:"date,omitempty"`
	Name  *string    `json:"name,omitempty"`
	Email *string    `json:"email,omitempty"`
}

func (c CommitAuthor) String() string {
	return Stringify(c)
}

// GetCommit fetchs the Commit object for a given SHA.
//
// GitHub API docs: http://developer.github.com/v3/git/commits/#get-a-commit
func (s *GitService) GetCommit(owner string, repo string, sha string) (*Commit, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/git/commits/%v", owner, repo, sha)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	c := new(Commit)
	resp, err := s.client.Do(req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// createCommit represents the body of a CreateCommit request.
type createCommit struct {
	Author    *CommitAuthor `json:"author,omitempty"`
	Committer *CommitAuthor `json:"committer,omitempty"`
	Message   *string       `json:"message,omitempty"`
	Tree      *string       `json:"tree,omitempty"`
	Parents   []string      `json:"parents,omitempty"`
}

// CreateCommit creates a new commit in a repository.
//
// The commit.Committer is optional and will be filled with the commit.Author
// data if omitted. If the commit.Author is omitted, it will be filled in with
// the authenticated user’s information and the current date.
//
// GitHub API docs: http://developer.github.com/v3/git/commits/#create-a-commit
func (s *GitService) CreateCommit(owner string, repo string, commit *Commit) (*Commit, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/git/commits", owner, repo)

	body := &createCommit{}
	if commit != nil {
		parents := make([]string, len(commit.Parents))
		for i, parent := range commit.Parents {
			parents[i] = *parent.SHA
		}

		body = &createCommit{
			Author:    commit.Author,
			Committer: commit.Committer,
			Message:   commit.Message,
			Tree:      commit.Tree.SHA,
			Parents:   parents,
		}
	}

	req, err := s.client.NewRequest("POST", u, body)
	if err != nil {
		return nil, nil, err
	}

	c := new(Commit)
	resp, err := s.client.Do(req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}
