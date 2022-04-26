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

// IssuesService handles communication with the issue related
// methods of the GitHub API.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/
type IssuesService service

// Issue represents a GitHub issue on a repository.
//
// Note: As far as the GitHub API is concerned, every pull request is an issue,
// but not every issue is a pull request. Some endpoints, events, and webhooks
// may also return pull requests via this struct. If PullRequestLinks is nil,
// this is an issue, and if PullRequestLinks is not nil, this is a pull request.
// The IsPullRequest helper method can be used to check that.
type Issue struct {
	ID                *int64            `json:"id,omitempty"`
	Number            *int              `json:"number,omitempty"`
	State             *string           `json:"state,omitempty"`
	Locked            *bool             `json:"locked,omitempty"`
	Title             *string           `json:"title,omitempty"`
	Body              *string           `json:"body,omitempty"`
	AuthorAssociation *string           `json:"author_association,omitempty"`
	User              *User             `json:"user,omitempty"`
	Labels            []*Label          `json:"labels,omitempty"`
	Assignee          *User             `json:"assignee,omitempty"`
	Comments          *int              `json:"comments,omitempty"`
	ClosedAt          *time.Time        `json:"closed_at,omitempty"`
	CreatedAt         *time.Time        `json:"created_at,omitempty"`
	UpdatedAt         *time.Time        `json:"updated_at,omitempty"`
	ClosedBy          *User             `json:"closed_by,omitempty"`
	URL               *string           `json:"url,omitempty"`
	HTMLURL           *string           `json:"html_url,omitempty"`
	CommentsURL       *string           `json:"comments_url,omitempty"`
	EventsURL         *string           `json:"events_url,omitempty"`
	LabelsURL         *string           `json:"labels_url,omitempty"`
	RepositoryURL     *string           `json:"repository_url,omitempty"`
	Milestone         *Milestone        `json:"milestone,omitempty"`
	PullRequestLinks  *PullRequestLinks `json:"pull_request,omitempty"`
	Repository        *Repository       `json:"repository,omitempty"`
	Reactions         *Reactions        `json:"reactions,omitempty"`
	Assignees         []*User           `json:"assignees,omitempty"`
	NodeID            *string           `json:"node_id,omitempty"`

	// TextMatches is only populated from search results that request text matches
	// See: search.go and https://docs.github.com/en/free-pro-team@latest/rest/reference/search/#text-match-metadata
	TextMatches []*TextMatch `json:"text_matches,omitempty"`

	// ActiveLockReason is populated only when LockReason is provided while locking the issue.
	// Possible values are: "off-topic", "too heated", "resolved", and "spam".
	ActiveLockReason *string `json:"active_lock_reason,omitempty"`
}

func (i Issue) String() string {
	return Stringify(i)
}

// IsPullRequest reports whether the issue is also a pull request. It uses the
// method recommended by GitHub's API documentation, which is to check whether
// PullRequestLinks is non-nil.
func (i Issue) IsPullRequest() bool {
	return i.PullRequestLinks != nil
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
	Assignees *[]string `json:"assignees,omitempty"`
}

// IssueListOptions specifies the optional parameters to the IssuesService.List
// and IssuesService.ListByOrg methods.
type IssueListOptions struct {
	// Filter specifies which issues to list. Possible values are: assigned,
	// created, mentioned, subscribed, all. Default is "assigned".
	Filter string `url:"filter,omitempty"`

	// State filters issues based on their state. Possible values are: open,
	// closed, all. Default is "open".
	State string `url:"state,omitempty"`

	// Labels filters issues based on their label.
	Labels []string `url:"labels,comma,omitempty"`

	// Sort specifies how to sort issues. Possible values are: created, updated,
	// and comments. Default value is "created".
	Sort string `url:"sort,omitempty"`

	// Direction in which to sort issues. Possible values are: asc, desc.
	// Default is "desc".
	Direction string `url:"direction,omitempty"`

	// Since filters issues by time.
	Since time.Time `url:"since,omitempty"`

	ListOptions
}

// PullRequestLinks object is added to the Issue object when it's an issue included
// in the IssueCommentEvent webhook payload, if the webhook is fired by a comment on a PR.
type PullRequestLinks struct {
	URL      *string `json:"url,omitempty"`
	HTMLURL  *string `json:"html_url,omitempty"`
	DiffURL  *string `json:"diff_url,omitempty"`
	PatchURL *string `json:"patch_url,omitempty"`
}

// List the issues for the authenticated user. If all is true, list issues
// across all the user's visible repositories including owned, member, and
// organization repositories; if false, list only owned and member
// repositories.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#list-user-account-issues-assigned-to-the-authenticated-user
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#list-issues-assigned-to-the-authenticated-user
func (s *IssuesService) List(ctx context.Context, all bool, opts *IssueListOptions) ([]*Issue, *Response, error) {
	var u string
	if all {
		u = "issues"
	} else {
		u = "user/issues"
	}
	return s.listIssues(ctx, u, opts)
}

// ListByOrg fetches the issues in the specified organization for the
// authenticated user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#list-organization-issues-assigned-to-the-authenticated-user
func (s *IssuesService) ListByOrg(ctx context.Context, org string, opts *IssueListOptions) ([]*Issue, *Response, error) {
	u := fmt.Sprintf("orgs/%v/issues", org)
	return s.listIssues(ctx, u, opts)
}

func (s *IssuesService) listIssues(ctx context.Context, u string, opts *IssueListOptions) ([]*Issue, *Response, error) {
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launch.
	req.Header.Set("Accept", mediaTypeReactionsPreview)

	var issues []*Issue
	resp, err := s.client.Do(ctx, req, &issues)
	if err != nil {
		return nil, resp, err
	}

	return issues, resp, nil
}

// IssueListByRepoOptions specifies the optional parameters to the
// IssuesService.ListByRepo method.
type IssueListByRepoOptions struct {
	// Milestone limits issues for the specified milestone. Possible values are
	// a milestone number, "none" for issues with no milestone, "*" for issues
	// with any milestone.
	Milestone string `url:"milestone,omitempty"`

	// State filters issues based on their state. Possible values are: open,
	// closed, all. Default is "open".
	State string `url:"state,omitempty"`

	// Assignee filters issues based on their assignee. Possible values are a
	// user name, "none" for issues that are not assigned, "*" for issues with
	// any assigned user.
	Assignee string `url:"assignee,omitempty"`

	// Creator filters issues based on their creator.
	Creator string `url:"creator,omitempty"`

	// Mentioned filters issues to those mentioned a specific user.
	Mentioned string `url:"mentioned,omitempty"`

	// Labels filters issues based on their label.
	Labels []string `url:"labels,omitempty,comma"`

	// Sort specifies how to sort issues. Possible values are: created, updated,
	// and comments. Default value is "created".
	Sort string `url:"sort,omitempty"`

	// Direction in which to sort issues. Possible values are: asc, desc.
	// Default is "desc".
	Direction string `url:"direction,omitempty"`

	// Since filters issues by time.
	Since time.Time `url:"since,omitempty"`

	ListOptions
}

// ListByRepo lists the issues for the specified repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#list-repository-issues
func (s *IssuesService) ListByRepo(ctx context.Context, owner string, repo string, opts *IssueListByRepoOptions) ([]*Issue, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues", owner, repo)
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

	var issues []*Issue
	resp, err := s.client.Do(ctx, req, &issues)
	if err != nil {
		return nil, resp, err
	}

	return issues, resp, nil
}

// Get a single issue.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#get-an-issue
func (s *IssuesService) Get(ctx context.Context, owner string, repo string, number int) (*Issue, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d", owner, repo, number)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launch.
	req.Header.Set("Accept", mediaTypeReactionsPreview)

	issue := new(Issue)
	resp, err := s.client.Do(ctx, req, issue)
	if err != nil {
		return nil, resp, err
	}

	return issue, resp, nil
}

// Create a new issue on the specified repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#create-an-issue
func (s *IssuesService) Create(ctx context.Context, owner string, repo string, issue *IssueRequest) (*Issue, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues", owner, repo)
	req, err := s.client.NewRequest("POST", u, issue)
	if err != nil {
		return nil, nil, err
	}

	i := new(Issue)
	resp, err := s.client.Do(ctx, req, i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, nil
}

// Edit an issue.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#update-an-issue
func (s *IssuesService) Edit(ctx context.Context, owner string, repo string, number int, issue *IssueRequest) (*Issue, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d", owner, repo, number)
	req, err := s.client.NewRequest("PATCH", u, issue)
	if err != nil {
		return nil, nil, err
	}

	i := new(Issue)
	resp, err := s.client.Do(ctx, req, i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, nil
}

// LockIssueOptions specifies the optional parameters to the
// IssuesService.Lock method.
type LockIssueOptions struct {
	// LockReason specifies the reason to lock this issue.
	// Providing a lock reason can help make it clearer to contributors why an issue
	// was locked. Possible values are: "off-topic", "too heated", "resolved", and "spam".
	LockReason string `json:"lock_reason,omitempty"`
}

// Lock an issue's conversation.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#lock-an-issue
func (s *IssuesService) Lock(ctx context.Context, owner string, repo string, number int, opts *LockIssueOptions) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d/lock", owner, repo, number)
	req, err := s.client.NewRequest("PUT", u, opts)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// Unlock an issue's conversation.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#unlock-an-issue
func (s *IssuesService) Unlock(ctx context.Context, owner string, repo string, number int) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d/lock", owner, repo, number)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
