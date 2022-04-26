// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"bytes"
	"context"
	"fmt"
	"time"
)

// PullRequestsService handles communication with the pull request related
// methods of the GitHub API.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/
type PullRequestsService service

// PullRequest represents a GitHub pull request on a repository.
type PullRequest struct {
	ID                  *int64     `json:"id,omitempty"`
	Number              *int       `json:"number,omitempty"`
	State               *string    `json:"state,omitempty"`
	Locked              *bool      `json:"locked,omitempty"`
	Title               *string    `json:"title,omitempty"`
	Body                *string    `json:"body,omitempty"`
	CreatedAt           *time.Time `json:"created_at,omitempty"`
	UpdatedAt           *time.Time `json:"updated_at,omitempty"`
	ClosedAt            *time.Time `json:"closed_at,omitempty"`
	MergedAt            *time.Time `json:"merged_at,omitempty"`
	Labels              []*Label   `json:"labels,omitempty"`
	User                *User      `json:"user,omitempty"`
	Draft               *bool      `json:"draft,omitempty"`
	Merged              *bool      `json:"merged,omitempty"`
	Mergeable           *bool      `json:"mergeable,omitempty"`
	MergeableState      *string    `json:"mergeable_state,omitempty"`
	MergedBy            *User      `json:"merged_by,omitempty"`
	MergeCommitSHA      *string    `json:"merge_commit_sha,omitempty"`
	Rebaseable          *bool      `json:"rebaseable,omitempty"`
	Comments            *int       `json:"comments,omitempty"`
	Commits             *int       `json:"commits,omitempty"`
	Additions           *int       `json:"additions,omitempty"`
	Deletions           *int       `json:"deletions,omitempty"`
	ChangedFiles        *int       `json:"changed_files,omitempty"`
	URL                 *string    `json:"url,omitempty"`
	HTMLURL             *string    `json:"html_url,omitempty"`
	IssueURL            *string    `json:"issue_url,omitempty"`
	StatusesURL         *string    `json:"statuses_url,omitempty"`
	DiffURL             *string    `json:"diff_url,omitempty"`
	PatchURL            *string    `json:"patch_url,omitempty"`
	CommitsURL          *string    `json:"commits_url,omitempty"`
	CommentsURL         *string    `json:"comments_url,omitempty"`
	ReviewCommentsURL   *string    `json:"review_comments_url,omitempty"`
	ReviewCommentURL    *string    `json:"review_comment_url,omitempty"`
	ReviewComments      *int       `json:"review_comments,omitempty"`
	Assignee            *User      `json:"assignee,omitempty"`
	Assignees           []*User    `json:"assignees,omitempty"`
	Milestone           *Milestone `json:"milestone,omitempty"`
	MaintainerCanModify *bool      `json:"maintainer_can_modify,omitempty"`
	AuthorAssociation   *string    `json:"author_association,omitempty"`
	NodeID              *string    `json:"node_id,omitempty"`
	RequestedReviewers  []*User    `json:"requested_reviewers,omitempty"`

	// RequestedTeams is populated as part of the PullRequestEvent.
	// See, https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/events/types/#pullrequestevent for an example.
	RequestedTeams []*Team `json:"requested_teams,omitempty"`

	Links *PRLinks           `json:"_links,omitempty"`
	Head  *PullRequestBranch `json:"head,omitempty"`
	Base  *PullRequestBranch `json:"base,omitempty"`

	// ActiveLockReason is populated only when LockReason is provided while locking the pull request.
	// Possible values are: "off-topic", "too heated", "resolved", and "spam".
	ActiveLockReason *string `json:"active_lock_reason,omitempty"`
}

func (p PullRequest) String() string {
	return Stringify(p)
}

// PRLink represents a single link object from GitHub pull request _links.
type PRLink struct {
	HRef *string `json:"href,omitempty"`
}

// PRLinks represents the "_links" object in a GitHub pull request.
type PRLinks struct {
	Self           *PRLink `json:"self,omitempty"`
	HTML           *PRLink `json:"html,omitempty"`
	Issue          *PRLink `json:"issue,omitempty"`
	Comments       *PRLink `json:"comments,omitempty"`
	ReviewComments *PRLink `json:"review_comments,omitempty"`
	ReviewComment  *PRLink `json:"review_comment,omitempty"`
	Commits        *PRLink `json:"commits,omitempty"`
	Statuses       *PRLink `json:"statuses,omitempty"`
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
	// State filters pull requests based on their state. Possible values are:
	// open, closed, all. Default is "open".
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
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#list-pull-requests
func (s *PullRequestsService) List(ctx context.Context, owner string, repo string, opts *PullRequestListOptions) ([]*PullRequest, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var pulls []*PullRequest
	resp, err := s.client.Do(ctx, req, &pulls)
	if err != nil {
		return nil, resp, err
	}

	return pulls, resp, nil
}

// ListPullRequestsWithCommit returns pull requests associated with a commit SHA.
//
// The results will include open and closed pull requests.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#list-pull-requests-associated-with-a-commit
func (s *PullRequestsService) ListPullRequestsWithCommit(ctx context.Context, owner, repo, sha string, opts *PullRequestListOptions) ([]*PullRequest, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/commits/%v/pulls", owner, repo, sha)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeListPullsOrBranchesForCommitPreview)
	var pulls []*PullRequest
	resp, err := s.client.Do(ctx, req, &pulls)
	if err != nil {
		return nil, resp, err
	}

	return pulls, resp, nil
}

// Get a single pull request.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#get-a-pull-request
func (s *PullRequestsService) Get(ctx context.Context, owner string, repo string, number int) (*PullRequest, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d", owner, repo, number)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	pull := new(PullRequest)
	resp, err := s.client.Do(ctx, req, pull)
	if err != nil {
		return nil, resp, err
	}

	return pull, resp, nil
}

// GetRaw gets a single pull request in raw (diff or patch) format.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#get-a-pull-request
func (s *PullRequestsService) GetRaw(ctx context.Context, owner string, repo string, number int, opts RawOptions) (string, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d", owner, repo, number)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return "", nil, err
	}

	switch opts.Type {
	case Diff:
		req.Header.Set("Accept", mediaTypeV3Diff)
	case Patch:
		req.Header.Set("Accept", mediaTypeV3Patch)
	default:
		return "", nil, fmt.Errorf("unsupported raw type %d", opts.Type)
	}

	var buf bytes.Buffer
	resp, err := s.client.Do(ctx, req, &buf)
	if err != nil {
		return "", resp, err
	}

	return buf.String(), resp, nil
}

// NewPullRequest represents a new pull request to be created.
type NewPullRequest struct {
	Title               *string `json:"title,omitempty"`
	Head                *string `json:"head,omitempty"`
	Base                *string `json:"base,omitempty"`
	Body                *string `json:"body,omitempty"`
	Issue               *int    `json:"issue,omitempty"`
	MaintainerCanModify *bool   `json:"maintainer_can_modify,omitempty"`
	Draft               *bool   `json:"draft,omitempty"`
}

// Create a new pull request on the specified repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#create-a-pull-request
func (s *PullRequestsService) Create(ctx context.Context, owner string, repo string, pull *NewPullRequest) (*PullRequest, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls", owner, repo)
	req, err := s.client.NewRequest("POST", u, pull)
	if err != nil {
		return nil, nil, err
	}

	p := new(PullRequest)
	resp, err := s.client.Do(ctx, req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, nil
}

// PullRequestBranchUpdateOptions specifies the optional parameters to the
// PullRequestsService.UpdateBranch method.
type PullRequestBranchUpdateOptions struct {
	// ExpectedHeadSHA specifies the most recent commit on the pull request's branch.
	// Default value is the SHA of the pull request's current HEAD ref.
	ExpectedHeadSHA *string `json:"expected_head_sha,omitempty"`
}

// PullRequestBranchUpdateResponse specifies the response of pull request branch update.
type PullRequestBranchUpdateResponse struct {
	Message *string `json:"message,omitempty"`
	URL     *string `json:"url,omitempty"`
}

// UpdateBranch updates the pull request branch with latest upstream changes.
//
// This method might return an AcceptedError and a status code of
// 202. This is because this is the status that GitHub returns to signify that
// it has now scheduled the update of the pull request branch in a background task.
// A follow up request, after a delay of a second or so, should result
// in a successful request.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#update-a-pull-request-branch
func (s *PullRequestsService) UpdateBranch(ctx context.Context, owner, repo string, number int, opts *PullRequestBranchUpdateOptions) (*PullRequestBranchUpdateResponse, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/update-branch", owner, repo, number)

	req, err := s.client.NewRequest("PUT", u, opts)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeUpdatePullRequestBranchPreview)

	p := new(PullRequestBranchUpdateResponse)
	resp, err := s.client.Do(ctx, req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, nil
}

type pullRequestUpdate struct {
	Title               *string `json:"title,omitempty"`
	Body                *string `json:"body,omitempty"`
	State               *string `json:"state,omitempty"`
	Base                *string `json:"base,omitempty"`
	MaintainerCanModify *bool   `json:"maintainer_can_modify,omitempty"`
}

// Edit a pull request.
// pull must not be nil.
//
// The following fields are editable: Title, Body, State, Base.Ref and MaintainerCanModify.
// Base.Ref updates the base branch of the pull request.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#update-a-pull-request
func (s *PullRequestsService) Edit(ctx context.Context, owner string, repo string, number int, pull *PullRequest) (*PullRequest, *Response, error) {
	if pull == nil {
		return nil, nil, fmt.Errorf("pull must be provided")
	}

	u := fmt.Sprintf("repos/%v/%v/pulls/%d", owner, repo, number)

	update := &pullRequestUpdate{
		Title:               pull.Title,
		Body:                pull.Body,
		State:               pull.State,
		MaintainerCanModify: pull.MaintainerCanModify,
	}
	// avoid updating the base branch when closing the Pull Request
	// - otherwise the GitHub API server returns a "Validation Failed" error:
	// "Cannot change base branch of closed pull request".
	if pull.Base != nil && pull.GetState() != "closed" {
		update.Base = pull.Base.Ref
	}

	req, err := s.client.NewRequest("PATCH", u, update)
	if err != nil {
		return nil, nil, err
	}

	p := new(PullRequest)
	resp, err := s.client.Do(ctx, req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, nil
}

// ListCommits lists the commits in a pull request.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#list-commits-on-a-pull-request
func (s *PullRequestsService) ListCommits(ctx context.Context, owner string, repo string, number int, opts *ListOptions) ([]*RepositoryCommit, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/commits", owner, repo, number)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var commits []*RepositoryCommit
	resp, err := s.client.Do(ctx, req, &commits)
	if err != nil {
		return nil, resp, err
	}

	return commits, resp, nil
}

// ListFiles lists the files in a pull request.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#list-pull-requests-files
func (s *PullRequestsService) ListFiles(ctx context.Context, owner string, repo string, number int, opts *ListOptions) ([]*CommitFile, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/files", owner, repo, number)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var commitFiles []*CommitFile
	resp, err := s.client.Do(ctx, req, &commitFiles)
	if err != nil {
		return nil, resp, err
	}

	return commitFiles, resp, nil
}

// IsMerged checks if a pull request has been merged.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#check-if-a-pull-request-has-been-merged
func (s *PullRequestsService) IsMerged(ctx context.Context, owner string, repo string, number int) (bool, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/merge", owner, repo, number)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return false, nil, err
	}

	resp, err := s.client.Do(ctx, req, nil)
	merged, err := parseBoolResponse(err)
	return merged, resp, err
}

// PullRequestMergeResult represents the result of merging a pull request.
type PullRequestMergeResult struct {
	SHA     *string `json:"sha,omitempty"`
	Merged  *bool   `json:"merged,omitempty"`
	Message *string `json:"message,omitempty"`
}

// PullRequestOptions lets you define how a pull request will be merged.
type PullRequestOptions struct {
	CommitTitle string // Title for the automatic commit message. (Optional.)
	SHA         string // SHA that pull request head must match to allow merge. (Optional.)

	// The merge method to use. Possible values include: "merge", "squash", and "rebase" with the default being merge. (Optional.)
	MergeMethod string
}

type pullRequestMergeRequest struct {
	CommitMessage string `json:"commit_message,omitempty"`
	CommitTitle   string `json:"commit_title,omitempty"`
	MergeMethod   string `json:"merge_method,omitempty"`
	SHA           string `json:"sha,omitempty"`
}

// Merge a pull request.
// commitMessage is an extra detail to append to automatic commit message.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#merge-a-pull-request
func (s *PullRequestsService) Merge(ctx context.Context, owner string, repo string, number int, commitMessage string, options *PullRequestOptions) (*PullRequestMergeResult, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/merge", owner, repo, number)

	pullRequestBody := &pullRequestMergeRequest{CommitMessage: commitMessage}
	if options != nil {
		pullRequestBody.CommitTitle = options.CommitTitle
		pullRequestBody.MergeMethod = options.MergeMethod
		pullRequestBody.SHA = options.SHA
	}
	req, err := s.client.NewRequest("PUT", u, pullRequestBody)
	if err != nil {
		return nil, nil, err
	}

	mergeResult := new(PullRequestMergeResult)
	resp, err := s.client.Do(ctx, req, mergeResult)
	if err != nil {
		return nil, resp, err
	}

	return mergeResult, resp, nil
}
