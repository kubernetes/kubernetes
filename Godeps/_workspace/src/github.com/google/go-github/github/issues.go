// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// IssuesService handles communication with the issue related
// methods of the GitHub API.
//
// GitHub API docs: http://developer.github.com/v3/issues/
type IssuesService struct {
	client *Client
}

// Issue represents a GitHub issue on a repository.
type Issue struct {
	Number           *int              `json:"number,omitempty"`
	State            *string           `json:"state,omitempty"`
	Title            *string           `json:"title,omitempty"`
	Body             *string           `json:"body,omitempty"`
	User             *User             `json:"user,omitempty"`
	Labels           []Label           `json:"labels,omitempty"`
	Assignee         *User             `json:"assignee,omitempty"`
	Comments         *int              `json:"comments,omitempty"`
	ClosedAt         *time.Time        `json:"closed_at,omitempty"`
	CreatedAt        *time.Time        `json:"created_at,omitempty"`
	UpdatedAt        *time.Time        `json:"updated_at,omitempty"`
	URL              *string           `json:"url,omitempty"`
	HTMLURL          *string           `json:"html_url,omitempty"`
	Milestone        *Milestone        `json:"milestone,omitempty"`
	PullRequestLinks *PullRequestLinks `json:"pull_request,omitempty"`

	// TextMatches is only populated from search results that request text matches
	// See: search.go and https://developer.github.com/v3/search/#text-match-metadata
	TextMatches []TextMatch `json:"text_matches,omitempty"`
}

func (i Issue) String() string {
	return Stringify(i)
}

// IssueRequest represents a request to create/edit an issue.
// It is separate from Issue above because otherwise Labels
// and Assignee fail to serialize to the correct JSON.
type IssueRequest struct {
	Title     *string   `json:"title,omitempty"`
	Body      *string   `json:"body,omitempty"`
	Labels    *[]string `json:"labels,omitempty"`
	Assignee  *string   `json:"assignee,omitempty"`
	State     *string   `json:"state,omitempty"`
	Milestone *int      `json:"milestone,omitempty"`
}

// IssueListOptions specifies the optional parameters to the IssuesService.List
// and IssuesService.ListByOrg methods.
type IssueListOptions struct {
	// Filter specifies which issues to list.  Possible values are: assigned,
	// created, mentioned, subscribed, all.  Default is "assigned".
	Filter string `url:"filter,omitempty"`

	// State filters issues based on their state.  Possible values are: open,
	// closed.  Default is "open".
	State string `url:"state,omitempty"`

	// Labels filters issues based on their label.
	Labels []string `url:"labels,comma,omitempty"`

	// Sort specifies how to sort issues.  Possible values are: created, updated,
	// and comments.  Default value is "assigned".
	Sort string `url:"sort,omitempty"`

	// Direction in which to sort issues.  Possible values are: asc, desc.
	// Default is "asc".
	Direction string `url:"direction,omitempty"`

	// Since filters issues by time.
	Since time.Time `url:"since,omitempty"`

	ListOptions
}

// PullRequestLinks object is added to the Issue object when it's an issue included
// in the IssueCommentEvent webhook payload, if the webhooks is fired by a comment on a PR
type PullRequestLinks struct {
	URL      *string `json:"url,omitempty"`
	HTMLURL  *string `json:"html_url,omitempty"`
	DiffURL  *string `json:"diff_url,omitempty"`
	PatchURL *string `json:"patch_url,omitempty"`
}

// List the issues for the authenticated user.  If all is true, list issues
// across all the user's visible repositories including owned, member, and
// organization repositories; if false, list only owned and member
// repositories.
//
// GitHub API docs: http://developer.github.com/v3/issues/#list-issues
func (s *IssuesService) List(all bool, opt *IssueListOptions) ([]Issue, *Response, error) {
	var u string
	if all {
		u = "issues"
	} else {
		u = "user/issues"
	}
	return s.listIssues(u, opt)
}

// ListByOrg fetches the issues in the specified organization for the
// authenticated user.
//
// GitHub API docs: http://developer.github.com/v3/issues/#list-issues
func (s *IssuesService) ListByOrg(org string, opt *IssueListOptions) ([]Issue, *Response, error) {
	u := fmt.Sprintf("orgs/%v/issues", org)
	return s.listIssues(u, opt)
}

func (s *IssuesService) listIssues(u string, opt *IssueListOptions) ([]Issue, *Response, error) {
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	issues := new([]Issue)
	resp, err := s.client.Do(req, issues)
	if err != nil {
		return nil, resp, err
	}

	return *issues, resp, err
}

// IssueListByRepoOptions specifies the optional parameters to the
// IssuesService.ListByRepo method.
type IssueListByRepoOptions struct {
	// Milestone limits issues for the specified milestone.  Possible values are
	// a milestone number, "none" for issues with no milestone, "*" for issues
	// with any milestone.
	Milestone string `url:"milestone,omitempty"`

	// State filters issues based on their state.  Possible values are: open,
	// closed.  Default is "open".
	State string `url:"state,omitempty"`

	// Assignee filters issues based on their assignee.  Possible values are a
	// user name, "none" for issues that are not assigned, "*" for issues with
	// any assigned user.
	Assignee string `url:"assignee,omitempty"`

	// Assignee filters issues based on their creator.
	Creator string `url:"creator,omitempty"`

	// Assignee filters issues to those mentioned a specific user.
	Mentioned string `url:"mentioned,omitempty"`

	// Labels filters issues based on their label.
	Labels []string `url:"labels,omitempty,comma"`

	// Sort specifies how to sort issues.  Possible values are: created, updated,
	// and comments.  Default value is "assigned".
	Sort string `url:"sort,omitempty"`

	// Direction in which to sort issues.  Possible values are: asc, desc.
	// Default is "asc".
	Direction string `url:"direction,omitempty"`

	// Since filters issues by time.
	Since time.Time `url:"since,omitempty"`

	ListOptions
}

// ListByRepo lists the issues for the specified repository.
//
// GitHub API docs: http://developer.github.com/v3/issues/#list-issues-for-a-repository
func (s *IssuesService) ListByRepo(owner string, repo string, opt *IssueListByRepoOptions) ([]Issue, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	issues := new([]Issue)
	resp, err := s.client.Do(req, issues)
	if err != nil {
		return nil, resp, err
	}

	return *issues, resp, err
}

// Get a single issue.
//
// GitHub API docs: http://developer.github.com/v3/issues/#get-a-single-issue
func (s *IssuesService) Get(owner string, repo string, number int) (*Issue, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d", owner, repo, number)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	issue := new(Issue)
	resp, err := s.client.Do(req, issue)
	if err != nil {
		return nil, resp, err
	}

	return issue, resp, err
}

// Create a new issue on the specified repository.
//
// GitHub API docs: http://developer.github.com/v3/issues/#create-an-issue
func (s *IssuesService) Create(owner string, repo string, issue *IssueRequest) (*Issue, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues", owner, repo)
	req, err := s.client.NewRequest("POST", u, issue)
	if err != nil {
		return nil, nil, err
	}

	i := new(Issue)
	resp, err := s.client.Do(req, i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, err
}

// Edit an issue.
//
// GitHub API docs: http://developer.github.com/v3/issues/#edit-an-issue
func (s *IssuesService) Edit(owner string, repo string, number int, issue *IssueRequest) (*Issue, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d", owner, repo, number)
	req, err := s.client.NewRequest("PATCH", u, issue)
	if err != nil {
		return nil, nil, err
	}

	i := new(Issue)
	resp, err := s.client.Do(req, i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, err
}
