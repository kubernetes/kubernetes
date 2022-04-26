//
// Copyright 2021, Sander van Harmelen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package gitlab

import (
	"fmt"
	"net/url"
	"time"
)

// CommitsService handles communication with the commit related methods
// of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html
type CommitsService struct {
	client *Client
}

// Commit represents a GitLab commit.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html
type Commit struct {
	ID             string           `json:"id"`
	ShortID        string           `json:"short_id"`
	Title          string           `json:"title"`
	AuthorName     string           `json:"author_name"`
	AuthorEmail    string           `json:"author_email"`
	AuthoredDate   *time.Time       `json:"authored_date"`
	CommitterName  string           `json:"committer_name"`
	CommitterEmail string           `json:"committer_email"`
	CommittedDate  *time.Time       `json:"committed_date"`
	CreatedAt      *time.Time       `json:"created_at"`
	Message        string           `json:"message"`
	ParentIDs      []string         `json:"parent_ids"`
	Stats          *CommitStats     `json:"stats"`
	Status         *BuildStateValue `json:"status"`
	LastPipeline   *PipelineInfo    `json:"last_pipeline"`
	ProjectID      int              `json:"project_id"`
	WebURL         string           `json:"web_url"`
}

// CommitStats represents the number of added and deleted files in a commit.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html
type CommitStats struct {
	Additions int `json:"additions"`
	Deletions int `json:"deletions"`
	Total     int `json:"total"`
}

func (c Commit) String() string {
	return Stringify(c)
}

// ListCommitsOptions represents the available ListCommits() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html#list-repository-commits
type ListCommitsOptions struct {
	ListOptions
	RefName     *string    `url:"ref_name,omitempty" json:"ref_name,omitempty"`
	Since       *time.Time `url:"since,omitempty" json:"since,omitempty"`
	Until       *time.Time `url:"until,omitempty" json:"until,omitempty"`
	Path        *string    `url:"path,omitempty" json:"path,omitempty"`
	All         *bool      `url:"all,omitempty" json:"all,omitempty"`
	WithStats   *bool      `url:"with_stats,omitempty" json:"with_stats,omitempty"`
	FirstParent *bool      `url:"first_parent,omitempty" json:"first_parent,omitempty"`
}

// ListCommits gets a list of repository commits in a project.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html#list-commits
func (s *CommitsService) ListCommits(pid interface{}, opt *ListCommitsOptions, options ...RequestOptionFunc) ([]*Commit, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/repository/commits", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var c []*Commit
	resp, err := s.client.Do(req, &c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// CommitRef represents the reference of branches/tags in a commit.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/commits.html#get-references-a-commit-is-pushed-to
type CommitRef struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

// GetCommitRefsOptions represents the available GetCommitRefs() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/commits.html#get-references-a-commit-is-pushed-to
type GetCommitRefsOptions struct {
	ListOptions
	Type *string `url:"type,omitempty" json:"type,omitempty"`
}

// GetCommitRefs gets all references (from branches or tags) a commit is pushed to
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/commits.html#get-references-a-commit-is-pushed-to
func (s *CommitsService) GetCommitRefs(pid interface{}, sha string, opt *GetCommitRefsOptions, options ...RequestOptionFunc) ([]*CommitRef, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/repository/commits/%s/refs", pathEscape(project), url.PathEscape(sha))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var cs []*CommitRef
	resp, err := s.client.Do(req, &cs)
	if err != nil {
		return nil, resp, err
	}

	return cs, resp, err
}

// GetCommit gets a specific commit identified by the commit hash or name of a
// branch or tag.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html#get-a-single-commit
func (s *CommitsService) GetCommit(pid interface{}, sha string, options ...RequestOptionFunc) (*Commit, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	if sha == "" {
		return nil, nil, fmt.Errorf("SHA must be a non-empty string")
	}
	u := fmt.Sprintf("projects/%s/repository/commits/%s", pathEscape(project), url.PathEscape(sha))

	req, err := s.client.NewRequest("GET", u, nil, options)
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

// CreateCommitOptions represents the available options for a new commit.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html#create-a-commit-with-multiple-files-and-actions
type CreateCommitOptions struct {
	Branch        *string                `url:"branch,omitempty" json:"branch,omitempty"`
	CommitMessage *string                `url:"commit_message,omitempty" json:"commit_message,omitempty"`
	StartBranch   *string                `url:"start_branch,omitempty" json:"start_branch,omitempty"`
	StartSHA      *string                `url:"start_sha,omitempty" json:"start_sha,omitempty"`
	StartProject  *string                `url:"start_project,omitempty" json:"start_project,omitempty"`
	Actions       []*CommitActionOptions `url:"actions,omitempty" json:"actions,omitempty"`
	AuthorEmail   *string                `url:"author_email,omitempty" json:"author_email,omitempty"`
	AuthorName    *string                `url:"author_name,omitempty" json:"author_name,omitempty"`
	Stats         *bool                  `url:"stats,omitempty" json:"stats,omitempty"`
	Force         *bool                  `url:"force,omitempty" json:"force,omitempty"`
}

// CommitActionOptions represents the available options for a new single
// file action.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html#create-a-commit-with-multiple-files-and-actions
type CommitActionOptions struct {
	Action          *FileAction `url:"action,omitempty" json:"action,omitempty"`
	FilePath        *string     `url:"file_path,omitempty" json:"file_path,omitempty"`
	PreviousPath    *string     `url:"previous_path,omitempty" json:"previous_path,omitempty"`
	Content         *string     `url:"content,omitempty" json:"content,omitempty"`
	Encoding        *string     `url:"encoding,omitempty" json:"encoding,omitempty"`
	LastCommitID    *string     `url:"last_commit_id,omitempty" json:"last_commit_id,omitempty"`
	ExecuteFilemode *bool       `url:"execute_filemode,omitempty" json:"execute_filemode,omitempty"`
}

// CreateCommit creates a commit with multiple files and actions.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html#create-a-commit-with-multiple-files-and-actions
func (s *CommitsService) CreateCommit(pid interface{}, opt *CreateCommitOptions, options ...RequestOptionFunc) (*Commit, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/repository/commits", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	c := new(Commit)
	resp, err := s.client.Do(req, &c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// Diff represents a GitLab diff.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html
type Diff struct {
	Diff        string `json:"diff"`
	NewPath     string `json:"new_path"`
	OldPath     string `json:"old_path"`
	AMode       string `json:"a_mode"`
	BMode       string `json:"b_mode"`
	NewFile     bool   `json:"new_file"`
	RenamedFile bool   `json:"renamed_file"`
	DeletedFile bool   `json:"deleted_file"`
}

func (d Diff) String() string {
	return Stringify(d)
}

// GetCommitDiffOptions represents the available GetCommitDiff() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/commits.html#get-the-diff-of-a-commit
type GetCommitDiffOptions ListOptions

// GetCommitDiff gets the diff of a commit in a project..
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/commits.html#get-the-diff-of-a-commit
func (s *CommitsService) GetCommitDiff(pid interface{}, sha string, opt *GetCommitDiffOptions, options ...RequestOptionFunc) ([]*Diff, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/repository/commits/%s/diff", pathEscape(project), url.PathEscape(sha))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var d []*Diff
	resp, err := s.client.Do(req, &d)
	if err != nil {
		return nil, resp, err
	}

	return d, resp, err
}

// CommitComment represents a GitLab commit comment.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html
type CommitComment struct {
	Note     string `json:"note"`
	Path     string `json:"path"`
	Line     int    `json:"line"`
	LineType string `json:"line_type"`
	Author   Author `json:"author"`
}

// Author represents a GitLab commit author
type Author struct {
	ID        int        `json:"id"`
	Username  string     `json:"username"`
	Email     string     `json:"email"`
	Name      string     `json:"name"`
	State     string     `json:"state"`
	Blocked   bool       `json:"blocked"`
	CreatedAt *time.Time `json:"created_at"`
}

func (c CommitComment) String() string {
	return Stringify(c)
}

// GetCommitCommentsOptions represents the available GetCommitComments() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/commits.html#get-the-comments-of-a-commit
type GetCommitCommentsOptions ListOptions

// GetCommitComments gets the comments of a commit in a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/commits.html#get-the-comments-of-a-commit
func (s *CommitsService) GetCommitComments(pid interface{}, sha string, opt *GetCommitCommentsOptions, options ...RequestOptionFunc) ([]*CommitComment, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/repository/commits/%s/comments", pathEscape(project), url.PathEscape(sha))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var c []*CommitComment
	resp, err := s.client.Do(req, &c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// PostCommitCommentOptions represents the available PostCommitComment()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/commits.html#post-comment-to-commit
type PostCommitCommentOptions struct {
	Note     *string `url:"note,omitempty" json:"note,omitempty"`
	Path     *string `url:"path" json:"path"`
	Line     *int    `url:"line" json:"line"`
	LineType *string `url:"line_type" json:"line_type"`
}

// PostCommitComment adds a comment to a commit. Optionally you can post
// comments on a specific line of a commit. Therefor both path, line_new and
// line_old are required.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/commits.html#post-comment-to-commit
func (s *CommitsService) PostCommitComment(pid interface{}, sha string, opt *PostCommitCommentOptions, options ...RequestOptionFunc) (*CommitComment, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/repository/commits/%s/comments", pathEscape(project), url.PathEscape(sha))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	c := new(CommitComment)
	resp, err := s.client.Do(req, c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// GetCommitStatusesOptions represents the available GetCommitStatuses() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html#get-the-status-of-a-commit
type GetCommitStatusesOptions struct {
	ListOptions
	Ref   *string `url:"ref,omitempty" json:"ref,omitempty"`
	Stage *string `url:"stage,omitempty" json:"stage,omitempty"`
	Name  *string `url:"name,omitempty" json:"name,omitempty"`
	All   *bool   `url:"all,omitempty" json:"all,omitempty"`
}

// CommitStatus represents a GitLab commit status.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html#get-the-status-of-a-commit
type CommitStatus struct {
	ID           int        `json:"id"`
	SHA          string     `json:"sha"`
	Ref          string     `json:"ref"`
	Status       string     `json:"status"`
	CreatedAt    *time.Time `json:"created_at"`
	StartedAt    *time.Time `json:"started_at"`
	FinishedAt   *time.Time `json:"finished_at"`
	Name         string     `json:"name"`
	AllowFailure bool       `json:"allow_failure"`
	Author       Author     `json:"author"`
	Description  string     `json:"description"`
	TargetURL    string     `json:"target_url"`
}

// GetCommitStatuses gets the statuses of a commit in a project.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html#get-the-status-of-a-commit
func (s *CommitsService) GetCommitStatuses(pid interface{}, sha string, opt *GetCommitStatusesOptions, options ...RequestOptionFunc) ([]*CommitStatus, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/repository/commits/%s/statuses", pathEscape(project), url.PathEscape(sha))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var cs []*CommitStatus
	resp, err := s.client.Do(req, &cs)
	if err != nil {
		return nil, resp, err
	}

	return cs, resp, err
}

// SetCommitStatusOptions represents the available SetCommitStatus() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html#post-the-status-to-commit
type SetCommitStatusOptions struct {
	State       BuildStateValue `url:"state" json:"state"`
	Ref         *string         `url:"ref,omitempty" json:"ref,omitempty"`
	Name        *string         `url:"name,omitempty" json:"name,omitempty"`
	Context     *string         `url:"context,omitempty" json:"context,omitempty"`
	TargetURL   *string         `url:"target_url,omitempty" json:"target_url,omitempty"`
	Description *string         `url:"description,omitempty" json:"description,omitempty"`
	Coverage    *float64        `url:"coverage,omitempty" json:"coverage,omitempty"`
	PipelineID  *int            `url:"pipeline_id,omitempty" json:"pipeline_id,omitempty"`
}

// SetCommitStatus sets the status of a commit in a project.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html#post-the-status-to-commit
func (s *CommitsService) SetCommitStatus(pid interface{}, sha string, opt *SetCommitStatusOptions, options ...RequestOptionFunc) (*CommitStatus, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/statuses/%s", pathEscape(project), url.PathEscape(sha))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	cs := new(CommitStatus)
	resp, err := s.client.Do(req, &cs)
	if err != nil {
		return nil, resp, err
	}

	return cs, resp, err
}

// GetMergeRequestsByCommit gets merge request associated with a commit.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/commits.html#list-merge-requests-associated-with-a-commit
func (s *CommitsService) GetMergeRequestsByCommit(pid interface{}, sha string, options ...RequestOptionFunc) ([]*MergeRequest, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/repository/commits/%s/merge_requests", pathEscape(project), url.PathEscape(sha))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var mrs []*MergeRequest
	resp, err := s.client.Do(req, &mrs)
	if err != nil {
		return nil, resp, err
	}

	return mrs, resp, err
}

// CherryPickCommitOptions represents the available CherryPickCommit() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html#cherry-pick-a-commit
type CherryPickCommitOptions struct {
	Branch *string `url:"branch,omitempty" json:"branch,omitempty"`
}

// CherryPickCommit cherry picks a commit to a given branch.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/commits.html#cherry-pick-a-commit
func (s *CommitsService) CherryPickCommit(pid interface{}, sha string, opt *CherryPickCommitOptions, options ...RequestOptionFunc) (*Commit, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/repository/commits/%s/cherry_pick", pathEscape(project), url.PathEscape(sha))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	c := new(Commit)
	resp, err := s.client.Do(req, &c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// RevertCommitOptions represents the available RevertCommit() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/commits.html#revert-a-commit
type RevertCommitOptions struct {
	Branch *string `url:"branch,omitempty" json:"branch,omitempty"`
}

// RevertCommit reverts a commit in a given branch.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/commits.html#revert-a-commit
func (s *CommitsService) RevertCommit(pid interface{}, sha string, opt *RevertCommitOptions, options ...RequestOptionFunc) (*Commit, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/repository/commits/%s/revert", pathEscape(project), url.PathEscape(sha))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	c := new(Commit)
	resp, err := s.client.Do(req, &c)
	if err != nil {
		return nil, resp, err
	}

	return c, resp, err
}

// GPGSignature represents a Gitlab commit's GPG Signature.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/commits.html#get-gpg-signature-of-a-commit
type GPGSignature struct {
	KeyID              int    `json:"gpg_key_id"`
	KeyPrimaryKeyID    string `json:"gpg_key_primary_keyid"`
	KeyUserName        string `json:"gpg_key_user_name"`
	KeyUserEmail       string `json:"gpg_key_user_email"`
	VerificationStatus string `json:"verification_status"`
	KeySubkeyID        int    `json:"gpg_key_subkey_id"`
}

// GetGPGSiganature gets a GPG signature of a commit.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/commits.html#get-gpg-signature-of-a-commit
func (s *CommitsService) GetGPGSiganature(pid interface{}, sha string, options ...RequestOptionFunc) (*GPGSignature, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/repository/commits/%s/signature", pathEscape(project), url.PathEscape(sha))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	sig := new(GPGSignature)
	resp, err := s.client.Do(req, &sig)
	if err != nil {
		return nil, resp, err
	}

	return sig, resp, err
}
