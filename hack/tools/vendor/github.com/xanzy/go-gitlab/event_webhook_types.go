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
	"encoding/json"
	"fmt"
	"strconv"
	"time"
)

// PushEvent represents a push event.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/user/project/integrations/webhooks.html#push-events
type PushEvent struct {
	ObjectKind   string `json:"object_kind"`
	Before       string `json:"before"`
	After        string `json:"after"`
	Ref          string `json:"ref"`
	CheckoutSHA  string `json:"checkout_sha"`
	UserID       int    `json:"user_id"`
	UserName     string `json:"user_name"`
	UserUsername string `json:"user_username"`
	UserEmail    string `json:"user_email"`
	UserAvatar   string `json:"user_avatar"`
	ProjectID    int    `json:"project_id"`
	Project      struct {
		Name              string          `json:"name"`
		Description       string          `json:"description"`
		AvatarURL         string          `json:"avatar_url"`
		GitSSHURL         string          `json:"git_ssh_url"`
		GitHTTPURL        string          `json:"git_http_url"`
		Namespace         string          `json:"namespace"`
		PathWithNamespace string          `json:"path_with_namespace"`
		DefaultBranch     string          `json:"default_branch"`
		Homepage          string          `json:"homepage"`
		URL               string          `json:"url"`
		SSHURL            string          `json:"ssh_url"`
		HTTPURL           string          `json:"http_url"`
		WebURL            string          `json:"web_url"`
		Visibility        VisibilityValue `json:"visibility"`
	} `json:"project"`
	Repository *Repository `json:"repository"`
	Commits    []*struct {
		ID        string     `json:"id"`
		Message   string     `json:"message"`
		Timestamp *time.Time `json:"timestamp"`
		URL       string     `json:"url"`
		Author    struct {
			Name  string `json:"name"`
			Email string `json:"email"`
		} `json:"author"`
		Added    []string `json:"added"`
		Modified []string `json:"modified"`
		Removed  []string `json:"removed"`
	} `json:"commits"`
	TotalCommitsCount int `json:"total_commits_count"`
}

// TagEvent represents a tag event.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/user/project/integrations/webhooks.html#tag-events
type TagEvent struct {
	ObjectKind  string `json:"object_kind"`
	Before      string `json:"before"`
	After       string `json:"after"`
	Ref         string `json:"ref"`
	CheckoutSHA string `json:"checkout_sha"`
	UserID      int    `json:"user_id"`
	UserName    string `json:"user_name"`
	UserAvatar  string `json:"user_avatar"`
	UserEmail   string `json:"user_email"`
	ProjectID   int    `json:"project_id"`
	Message     string `json:"message"`
	Project     struct {
		Name              string          `json:"name"`
		Description       string          `json:"description"`
		AvatarURL         string          `json:"avatar_url"`
		GitSSHURL         string          `json:"git_ssh_url"`
		GitHTTPURL        string          `json:"git_http_url"`
		Namespace         string          `json:"namespace"`
		PathWithNamespace string          `json:"path_with_namespace"`
		DefaultBranch     string          `json:"default_branch"`
		Homepage          string          `json:"homepage"`
		URL               string          `json:"url"`
		SSHURL            string          `json:"ssh_url"`
		HTTPURL           string          `json:"http_url"`
		WebURL            string          `json:"web_url"`
		Visibility        VisibilityValue `json:"visibility"`
	} `json:"project"`
	Repository *Repository `json:"repository"`
	Commits    []*struct {
		ID        string     `json:"id"`
		Message   string     `json:"message"`
		Timestamp *time.Time `json:"timestamp"`
		URL       string     `json:"url"`
		Author    struct {
			Name  string `json:"name"`
			Email string `json:"email"`
		} `json:"author"`
		Added    []string `json:"added"`
		Modified []string `json:"modified"`
		Removed  []string `json:"removed"`
	} `json:"commits"`
	TotalCommitsCount int `json:"total_commits_count"`
}

// IssueEvent represents a issue event.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/user/project/integrations/webhooks.html#issues-events
type IssueEvent struct {
	ObjectKind string `json:"object_kind"`
	User       *User  `json:"user"`
	Project    struct {
		ID                int             `json:"id"`
		Name              string          `json:"name"`
		Description       string          `json:"description"`
		AvatarURL         string          `json:"avatar_url"`
		GitSSHURL         string          `json:"git_ssh_url"`
		GitHTTPURL        string          `json:"git_http_url"`
		Namespace         string          `json:"namespace"`
		PathWithNamespace string          `json:"path_with_namespace"`
		DefaultBranch     string          `json:"default_branch"`
		Homepage          string          `json:"homepage"`
		URL               string          `json:"url"`
		SSHURL            string          `json:"ssh_url"`
		HTTPURL           string          `json:"http_url"`
		WebURL            string          `json:"web_url"`
		Visibility        VisibilityValue `json:"visibility"`
	} `json:"project"`
	Repository       *Repository `json:"repository"`
	ObjectAttributes struct {
		ID          int    `json:"id"`
		Title       string `json:"title"`
		AssigneeID  int    `json:"assignee_id"`
		AuthorID    int    `json:"author_id"`
		ProjectID   int    `json:"project_id"`
		CreatedAt   string `json:"created_at"` // Should be *time.Time (see Gitlab issue #21468)
		UpdatedAt   string `json:"updated_at"` // Should be *time.Time (see Gitlab issue #21468)
		Position    int    `json:"position"`
		BranchName  string `json:"branch_name"`
		Description string `json:"description"`
		MilestoneID int    `json:"milestone_id"`
		State       string `json:"state"`
		IID         int    `json:"iid"`
		URL         string `json:"url"`
		Action      string `json:"action"`
	} `json:"object_attributes"`
	Assignee struct {
		Name      string `json:"name"`
		Username  string `json:"username"`
		AvatarURL string `json:"avatar_url"`
	} `json:"assignee"`
	Assignees []struct {
		Name      string `json:"name"`
		Username  string `json:"username"`
		AvatarURL string `json:"avatar_url"`
	} `json:"assignees"`
	Labels  []Label `json:"labels"`
	Changes struct {
		Description struct {
			Previous string `json:"previous"`
			Current  string `json:"current"`
		} `json:"description"`
		Labels struct {
			Previous []Label `json:"previous"`
			Current  []Label `json:"current"`
		} `json:"labels"`
		Title struct {
			Previous string `json:"previous"`
			Current  string `json:"current"`
		} `json:"title"`
		UpdatedByID struct {
			Previous int `json:"previous"`
			Current  int `json:"current"`
		} `json:"updated_by_id"`
	} `json:"changes"`
}

// JobEvent represents a job event.
//
// GitLab API docs:
// TODO: link to docs instead of src once they are published.
// https://gitlab.com/gitlab-org/gitlab-ce/blob/master/lib/gitlab/data_builder/build.rb
type JobEvent struct {
	ObjectKind         string  `json:"object_kind"`
	Ref                string  `json:"ref"`
	Tag                bool    `json:"tag"`
	BeforeSHA          string  `json:"before_sha"`
	SHA                string  `json:"sha"`
	BuildID            int     `json:"build_id"`
	BuildName          string  `json:"build_name"`
	BuildStage         string  `json:"build_stage"`
	BuildStatus        string  `json:"build_status"`
	BuildStartedAt     string  `json:"build_started_at"`
	BuildFinishedAt    string  `json:"build_finished_at"`
	BuildDuration      float64 `json:"build_duration"`
	BuildAllowFailure  bool    `json:"build_allow_failure"`
	BuildFailureReason string  `json:"build_failure_reason"`
	PipelineID         int     `json:"pipeline_id"`
	ProjectID          int     `json:"project_id"`
	ProjectName        string  `json:"project_name"`
	User               struct {
		ID    int    `json:"id"`
		Name  string `json:"name"`
		Email string `json:"email"`
	} `json:"user"`
	Commit struct {
		ID          int    `json:"id"`
		SHA         string `json:"sha"`
		Message     string `json:"message"`
		AuthorName  string `json:"author_name"`
		AuthorEmail string `json:"author_email"`
		AuthorURL   string `json:"author_url"`
		Status      string `json:"status"`
		Duration    int    `json:"duration"`
		StartedAt   string `json:"started_at"`
		FinishedAt  string `json:"finished_at"`
	} `json:"commit"`
	Repository *Repository `json:"repository"`
	Runner     struct {
		ID          int    `json:"id"`
		Active      bool   `json:"active"`
		Shared      bool   `json:"is_shared"`
		Description string `json:"description"`
	} `json:"runner"`
}

// CommitCommentEvent represents a comment on a commit event.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/user/project/integrations/webhooks.html#comment-on-commit
type CommitCommentEvent struct {
	ObjectKind string `json:"object_kind"`
	User       *User  `json:"user"`
	ProjectID  int    `json:"project_id"`
	Project    struct {
		Name              string          `json:"name"`
		Description       string          `json:"description"`
		AvatarURL         string          `json:"avatar_url"`
		GitSSHURL         string          `json:"git_ssh_url"`
		GitHTTPURL        string          `json:"git_http_url"`
		Namespace         string          `json:"namespace"`
		PathWithNamespace string          `json:"path_with_namespace"`
		DefaultBranch     string          `json:"default_branch"`
		Homepage          string          `json:"homepage"`
		URL               string          `json:"url"`
		SSHURL            string          `json:"ssh_url"`
		HTTPURL           string          `json:"http_url"`
		WebURL            string          `json:"web_url"`
		Visibility        VisibilityValue `json:"visibility"`
	} `json:"project"`
	Repository       *Repository `json:"repository"`
	ObjectAttributes struct {
		ID           int    `json:"id"`
		Note         string `json:"note"`
		NoteableType string `json:"noteable_type"`
		AuthorID     int    `json:"author_id"`
		CreatedAt    string `json:"created_at"`
		UpdatedAt    string `json:"updated_at"`
		ProjectID    int    `json:"project_id"`
		Attachment   string `json:"attachment"`
		LineCode     string `json:"line_code"`
		CommitID     string `json:"commit_id"`
		NoteableID   int    `json:"noteable_id"`
		System       bool   `json:"system"`
		StDiff       struct {
			Diff        string `json:"diff"`
			NewPath     string `json:"new_path"`
			OldPath     string `json:"old_path"`
			AMode       string `json:"a_mode"`
			BMode       string `json:"b_mode"`
			NewFile     bool   `json:"new_file"`
			RenamedFile bool   `json:"renamed_file"`
			DeletedFile bool   `json:"deleted_file"`
		} `json:"st_diff"`
	} `json:"object_attributes"`
	Commit *struct {
		ID        string     `json:"id"`
		Message   string     `json:"message"`
		Timestamp *time.Time `json:"timestamp"`
		URL       string     `json:"url"`
		Author    struct {
			Name  string `json:"name"`
			Email string `json:"email"`
		} `json:"author"`
	} `json:"commit"`
}

// MergeCommentEvent represents a comment on a merge event.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/user/project/integrations/webhooks.html#comment-on-merge-request
type MergeCommentEvent struct {
	ObjectKind string `json:"object_kind"`
	User       *User  `json:"user"`
	ProjectID  int    `json:"project_id"`
	Project    struct {
		Name              string          `json:"name"`
		Description       string          `json:"description"`
		AvatarURL         string          `json:"avatar_url"`
		GitSSHURL         string          `json:"git_ssh_url"`
		GitHTTPURL        string          `json:"git_http_url"`
		Namespace         string          `json:"namespace"`
		PathWithNamespace string          `json:"path_with_namespace"`
		DefaultBranch     string          `json:"default_branch"`
		Homepage          string          `json:"homepage"`
		URL               string          `json:"url"`
		SSHURL            string          `json:"ssh_url"`
		HTTPURL           string          `json:"http_url"`
		WebURL            string          `json:"web_url"`
		Visibility        VisibilityValue `json:"visibility"`
	} `json:"project"`
	ObjectAttributes struct {
		Attachment       string        `json:"attachment"`
		AuthorID         int           `json:"author_id"`
		ChangePosition   *NotePosition `json:"change_position"`
		CommitID         string        `json:"commit_id"`
		CreatedAt        string        `json:"created_at"`
		DiscussionID     string        `json:"discussion_id"`
		ID               int           `json:"id"`
		LineCode         string        `json:"line_code"`
		Note             string        `json:"note"`
		NoteableID       int           `json:"noteable_id"`
		NoteableType     string        `json:"noteable_type"`
		OriginalPosition *NotePosition `json:"original_position"`
		Position         *NotePosition `json:"position"`
		ProjectID        int           `json:"project_id"`
		ResolvedAt       string        `json:"resolved_at"`
		ResolvedByID     string        `json:"resolved_by_id"`
		ResolvedByPush   string        `json:"resolved_by_push"`
		StDiff           *Diff         `json:"st_diff"`
		System           bool          `json:"system"`
		Type             string        `json:"type"`
		UpdatedAt        string        `json:"updated_at"`
		UpdatedByID      string        `json:"updated_by_id"`
		Description      string        `json:"description"`
		URL              string        `json:"url"`
	} `json:"object_attributes"`
	Repository   *Repository `json:"repository"`
	MergeRequest struct {
		ID                        int          `json:"id"`
		TargetBranch              string       `json:"target_branch"`
		SourceBranch              string       `json:"source_branch"`
		SourceProjectID           int          `json:"source_project_id"`
		AuthorID                  int          `json:"author_id"`
		AssigneeID                int          `json:"assignee_id"`
		AssigneeIDs               []int        `json:"assignee_ids"`
		Title                     string       `json:"title"`
		CreatedAt                 string       `json:"created_at"`
		UpdatedAt                 string       `json:"updated_at"`
		MilestoneID               int          `json:"milestone_id"`
		State                     string       `json:"state"`
		MergeStatus               string       `json:"merge_status"`
		TargetProjectID           int          `json:"target_project_id"`
		IID                       int          `json:"iid"`
		Description               string       `json:"description"`
		Position                  int          `json:"position"`
		LockedAt                  string       `json:"locked_at"`
		UpdatedByID               int          `json:"updated_by_id"`
		MergeError                string       `json:"merge_error"`
		MergeParams               *MergeParams `json:"merge_params"`
		MergeWhenPipelineSucceeds bool         `json:"merge_when_pipeline_succeeds"`
		MergeUserID               int          `json:"merge_user_id"`
		MergeCommitSHA            string       `json:"merge_commit_sha"`
		DeletedAt                 string       `json:"deleted_at"`
		InProgressMergeCommitSHA  string       `json:"in_progress_merge_commit_sha"`
		LockVersion               int          `json:"lock_version"`
		ApprovalsBeforeMerge      string       `json:"approvals_before_merge"`
		RebaseCommitSHA           string       `json:"rebase_commit_sha"`
		TimeEstimate              int          `json:"time_estimate"`
		Squash                    bool         `json:"squash"`
		LastEditedAt              string       `json:"last_edited_at"`
		LastEditedByID            int          `json:"last_edited_by_id"`
		Source                    *Repository  `json:"source"`
		Target                    *Repository  `json:"target"`
		LastCommit                struct {
			ID        string     `json:"id"`
			Message   string     `json:"message"`
			Timestamp *time.Time `json:"timestamp"`
			URL       string     `json:"url"`
			Author    struct {
				Name  string `json:"name"`
				Email string `json:"email"`
			} `json:"author"`
		} `json:"last_commit"`
		WorkInProgress bool `json:"work_in_progress"`
		TotalTimeSpent int  `json:"total_time_spent"`
		HeadPipelineID int  `json:"head_pipeline_id"`
	} `json:"merge_request"`
}

// IssueCommentEvent represents a comment on an issue event.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/user/project/integrations/webhooks.html#comment-on-issue
type IssueCommentEvent struct {
	ObjectKind string `json:"object_kind"`
	User       *User  `json:"user"`
	ProjectID  int    `json:"project_id"`
	Project    struct {
		Name              string          `json:"name"`
		Description       string          `json:"description"`
		AvatarURL         string          `json:"avatar_url"`
		GitSSHURL         string          `json:"git_ssh_url"`
		GitHTTPURL        string          `json:"git_http_url"`
		Namespace         string          `json:"namespace"`
		PathWithNamespace string          `json:"path_with_namespace"`
		DefaultBranch     string          `json:"default_branch"`
		Homepage          string          `json:"homepage"`
		URL               string          `json:"url"`
		SSHURL            string          `json:"ssh_url"`
		HTTPURL           string          `json:"http_url"`
		WebURL            string          `json:"web_url"`
		Visibility        VisibilityValue `json:"visibility"`
	} `json:"project"`
	Repository       *Repository `json:"repository"`
	ObjectAttributes struct {
		ID           int     `json:"id"`
		Note         string  `json:"note"`
		NoteableType string  `json:"noteable_type"`
		AuthorID     int     `json:"author_id"`
		CreatedAt    string  `json:"created_at"`
		UpdatedAt    string  `json:"updated_at"`
		ProjectID    int     `json:"project_id"`
		Attachment   string  `json:"attachment"`
		LineCode     string  `json:"line_code"`
		CommitID     string  `json:"commit_id"`
		NoteableID   int     `json:"noteable_id"`
		System       bool    `json:"system"`
		StDiff       []*Diff `json:"st_diff"`
		URL          string  `json:"url"`
	} `json:"object_attributes"`
	Issue struct {
		ID                  int      `json:"id"`
		IID                 int      `json:"iid"`
		ProjectID           int      `json:"project_id"`
		MilestoneID         int      `json:"milestone_id"`
		AuthorID            int      `json:"author_id"`
		Description         string   `json:"description"`
		State               string   `json:"state"`
		Title               string   `json:"title"`
		LastEditedAt        string   `json:"last_edit_at"`
		LastEditedByID      int      `json:"last_edited_by_id"`
		UpdatedAt           string   `json:"updated_at"`
		UpdatedByID         int      `json:"updated_by_id"`
		CreatedAt           string   `json:"created_at"`
		ClosedAt            string   `json:"closed_at"`
		DueDate             *ISOTime `json:"due_date"`
		URL                 string   `json:"url"`
		TimeEstimate        int      `json:"time_estimate"`
		Confidential        bool     `json:"confidential"`
		TotalTimeSpent      int      `json:"total_time_spent"`
		HumanTotalTimeSpent string   `json:"human_total_time_spent"`
		HumanTimeEstimate   string   `json:"human_time_estimate"`
		AssigneeIDs         []int    `json:"assignee_ids"`
		AssigneeID          int      `json:"assignee_id"`
	} `json:"issue"`
}

// SnippetCommentEvent represents a comment on a snippet event.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/user/project/integrations/webhooks.html#comment-on-code-snippet
type SnippetCommentEvent struct {
	ObjectKind string `json:"object_kind"`
	User       *User  `json:"user"`
	ProjectID  int    `json:"project_id"`
	Project    struct {
		Name              string          `json:"name"`
		Description       string          `json:"description"`
		AvatarURL         string          `json:"avatar_url"`
		GitSSHURL         string          `json:"git_ssh_url"`
		GitHTTPURL        string          `json:"git_http_url"`
		Namespace         string          `json:"namespace"`
		PathWithNamespace string          `json:"path_with_namespace"`
		DefaultBranch     string          `json:"default_branch"`
		Homepage          string          `json:"homepage"`
		URL               string          `json:"url"`
		SSHURL            string          `json:"ssh_url"`
		HTTPURL           string          `json:"http_url"`
		WebURL            string          `json:"web_url"`
		Visibility        VisibilityValue `json:"visibility"`
	} `json:"project"`
	Repository       *Repository `json:"repository"`
	ObjectAttributes struct {
		ID           int    `json:"id"`
		Note         string `json:"note"`
		NoteableType string `json:"noteable_type"`
		AuthorID     int    `json:"author_id"`
		CreatedAt    string `json:"created_at"`
		UpdatedAt    string `json:"updated_at"`
		ProjectID    int    `json:"project_id"`
		Attachment   string `json:"attachment"`
		LineCode     string `json:"line_code"`
		CommitID     string `json:"commit_id"`
		NoteableID   int    `json:"noteable_id"`
		System       bool   `json:"system"`
		StDiff       *Diff  `json:"st_diff"`
		URL          string `json:"url"`
	} `json:"object_attributes"`
	Snippet *Snippet `json:"snippet"`
}

// MergeEvent represents a merge event.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/user/project/integrations/webhooks.html#merge-request-events
type MergeEvent struct {
	ObjectKind string `json:"object_kind"`
	User       *User  `json:"user"`
	Project    struct {
		ID                int             `json:"id"`
		Name              string          `json:"name"`
		Description       string          `json:"description"`
		AvatarURL         string          `json:"avatar_url"`
		GitSSHURL         string          `json:"git_ssh_url"`
		GitHTTPURL        string          `json:"git_http_url"`
		Namespace         string          `json:"namespace"`
		PathWithNamespace string          `json:"path_with_namespace"`
		DefaultBranch     string          `json:"default_branch"`
		Homepage          string          `json:"homepage"`
		URL               string          `json:"url"`
		SSHURL            string          `json:"ssh_url"`
		HTTPURL           string          `json:"http_url"`
		WebURL            string          `json:"web_url"`
		Visibility        VisibilityValue `json:"visibility"`
	} `json:"project"`
	ObjectAttributes struct {
		ID                       int          `json:"id"`
		TargetBranch             string       `json:"target_branch"`
		SourceBranch             string       `json:"source_branch"`
		SourceProjectID          int          `json:"source_project_id"`
		AuthorID                 int          `json:"author_id"`
		AssigneeID               int          `json:"assignee_id"`
		AssigneeIDs              []int        `json:"assignee_ids"`
		Title                    string       `json:"title"`
		CreatedAt                string       `json:"created_at"` // Should be *time.Time (see Gitlab issue #21468)
		UpdatedAt                string       `json:"updated_at"` // Should be *time.Time (see Gitlab issue #21468)
		StCommits                []*Commit    `json:"st_commits"`
		StDiffs                  []*Diff      `json:"st_diffs"`
		MilestoneID              int          `json:"milestone_id"`
		State                    string       `json:"state"`
		MergeStatus              string       `json:"merge_status"`
		TargetProjectID          int          `json:"target_project_id"`
		IID                      int          `json:"iid"`
		Description              string       `json:"description"`
		Position                 int          `json:"position"`
		LockedAt                 string       `json:"locked_at"`
		UpdatedByID              int          `json:"updated_by_id"`
		MergeError               string       `json:"merge_error"`
		MergeParams              *MergeParams `json:"merge_params"`
		MergeWhenBuildSucceeds   bool         `json:"merge_when_build_succeeds"`
		MergeUserID              int          `json:"merge_user_id"`
		MergeCommitSHA           string       `json:"merge_commit_sha"`
		DeletedAt                string       `json:"deleted_at"`
		ApprovalsBeforeMerge     string       `json:"approvals_before_merge"`
		RebaseCommitSHA          string       `json:"rebase_commit_sha"`
		InProgressMergeCommitSHA string       `json:"in_progress_merge_commit_sha"`
		LockVersion              int          `json:"lock_version"`
		TimeEstimate             int          `json:"time_estimate"`
		Source                   *Repository  `json:"source"`
		Target                   *Repository  `json:"target"`
		LastCommit               struct {
			ID        string     `json:"id"`
			Message   string     `json:"message"`
			Timestamp *time.Time `json:"timestamp"`
			URL       string     `json:"url"`
			Author    struct {
				Name  string `json:"name"`
				Email string `json:"email"`
			} `json:"author"`
		} `json:"last_commit"`
		WorkInProgress bool          `json:"work_in_progress"`
		URL            string        `json:"url"`
		Action         string        `json:"action"`
		OldRev         string        `json:"oldrev"`
		Assignee       MergeAssignee `json:"assignee"`
	} `json:"object_attributes"`
	Repository *Repository   `json:"repository"`
	Assignee   MergeAssignee `json:"assignee"`
	Labels     []Label       `json:"labels"`
	Changes    struct {
		Assignees struct {
			Previous []MergeAssignee `json:"previous"`
			Current  []MergeAssignee `json:"current"`
		} `json:"assignees"`
		Description struct {
			Previous string `json:"previous"`
			Current  string `json:"current"`
		} `json:"description"`
		Labels struct {
			Previous []Label `json:"previous"`
			Current  []Label `json:"current"`
		} `json:"labels"`
		SourceBranch struct {
			Previous string `json:"previous"`
			Current  string `json:"current"`
		} `json:"source_branch"`
		SourceProjectID struct {
			Previous int `json:"previous"`
			Current  int `json:"current"`
		} `json:"source_project_id"`
		StateID struct {
			Previous int `json:"previous"`
			Current  int `json:"current"`
		} `json:"state_id"`
		TargetBranch struct {
			Previous string `json:"previous"`
			Current  string `json:"current"`
		} `json:"target_branch"`
		TargetProjectID struct {
			Previous int `json:"previous"`
			Current  int `json:"current"`
		} `json:"target_project_id"`
		Title struct {
			Previous string `json:"previous"`
			Current  string `json:"current"`
		} `json:"title"`
		UpdatedByID struct {
			Previous int `json:"previous"`
			Current  int `json:"current"`
		} `json:"updated_by_id"`
		MilestoneID struct {
			Previous int `json:"previous"`
			Current  int `json:"current"`
		} `json:"milestone_id"`
	} `json:"changes"`
}

// MergeAssignee represents a merge assignee.
type MergeAssignee struct {
	Name      string `json:"name"`
	Username  string `json:"username"`
	AvatarURL string `json:"avatar_url"`
}

// MergeParams represents the merge params.
type MergeParams struct {
	ForceRemoveSourceBranch bool `json:"force_remove_source_branch"`
}

// UnmarshalJSON decodes the merge parameters
//
// This allows support of ForceRemoveSourceBranch for both type bool (>11.9) and string (<11.9)
func (p *MergeParams) UnmarshalJSON(b []byte) error {
	type Alias MergeParams
	raw := struct {
		*Alias
		ForceRemoveSourceBranch interface{} `json:"force_remove_source_branch"`
	}{
		Alias: (*Alias)(p),
	}

	err := json.Unmarshal(b, &raw)
	if err != nil {
		return err
	}

	switch v := raw.ForceRemoveSourceBranch.(type) {
	case nil:
		// No action needed.
	case bool:
		p.ForceRemoveSourceBranch = v
	case string:
		p.ForceRemoveSourceBranch, err = strconv.ParseBool(v)
		if err != nil {
			return err
		}
	default:
		return fmt.Errorf("failed to unmarshal ForceRemoveSourceBranch of type: %T", v)
	}

	return nil
}

// WikiPageEvent represents a wiki page event.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/user/project/integrations/webhooks.html#wiki-page-events
type WikiPageEvent struct {
	ObjectKind string `json:"object_kind"`
	User       *User  `json:"user"`
	Project    struct {
		Name              string          `json:"name"`
		Description       string          `json:"description"`
		AvatarURL         string          `json:"avatar_url"`
		GitSSHURL         string          `json:"git_ssh_url"`
		GitHTTPURL        string          `json:"git_http_url"`
		Namespace         string          `json:"namespace"`
		PathWithNamespace string          `json:"path_with_namespace"`
		DefaultBranch     string          `json:"default_branch"`
		Homepage          string          `json:"homepage"`
		URL               string          `json:"url"`
		SSHURL            string          `json:"ssh_url"`
		HTTPURL           string          `json:"http_url"`
		WebURL            string          `json:"web_url"`
		Visibility        VisibilityValue `json:"visibility"`
	} `json:"project"`
	Wiki struct {
		WebURL            string `json:"web_url"`
		GitSSHURL         string `json:"git_ssh_url"`
		GitHTTPURL        string `json:"git_http_url"`
		PathWithNamespace string `json:"path_with_namespace"`
		DefaultBranch     string `json:"default_branch"`
	} `json:"wiki"`
	ObjectAttributes struct {
		Title   string `json:"title"`
		Content string `json:"content"`
		Format  string `json:"format"`
		Message string `json:"message"`
		Slug    string `json:"slug"`
		URL     string `json:"url"`
		Action  string `json:"action"`
	} `json:"object_attributes"`
}

// PipelineEvent represents a pipeline event.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/user/project/integrations/webhooks.html#pipeline-events
type PipelineEvent struct {
	ObjectKind       string `json:"object_kind"`
	ObjectAttributes struct {
		ID         int      `json:"id"`
		Ref        string   `json:"ref"`
		Tag        bool     `json:"tag"`
		SHA        string   `json:"sha"`
		BeforeSHA  string   `json:"before_sha"`
		Source     string   `json:"source"`
		Status     string   `json:"status"`
		Stages     []string `json:"stages"`
		CreatedAt  string   `json:"created_at"`
		FinishedAt string   `json:"finished_at"`
		Duration   int      `json:"duration"`
	} `json:"object_attributes"`
	MergeRequest struct {
		ID                 int    `json:"id"`
		IID                int    `json:"iid"`
		Title              string `json:"title"`
		SourceBranch       string `json:"source_branch"`
		SourceProjectID    int    `json:"source_project_id"`
		TargetBranch       string `json:"target_branch"`
		TargetProjectID    int    `json:"target_project_id"`
		State              string `json:"state"`
		MergeRequestStatus string `json:"merge_status"`
		URL                string `json:"url"`
	} `json:"merge_request"`
	User struct {
		Name      string `json:"name"`
		Username  string `json:"username"`
		AvatarURL string `json:"avatar_url"`
	} `json:"user"`
	Project struct {
		ID                int             `json:"id"`
		Name              string          `json:"name"`
		Description       string          `json:"description"`
		AvatarURL         string          `json:"avatar_url"`
		GitSSHURL         string          `json:"git_ssh_url"`
		GitHTTPURL        string          `json:"git_http_url"`
		Namespace         string          `json:"namespace"`
		PathWithNamespace string          `json:"path_with_namespace"`
		DefaultBranch     string          `json:"default_branch"`
		Homepage          string          `json:"homepage"`
		URL               string          `json:"url"`
		SSHURL            string          `json:"ssh_url"`
		HTTPURL           string          `json:"http_url"`
		WebURL            string          `json:"web_url"`
		Visibility        VisibilityValue `json:"visibility"`
	} `json:"project"`
	Commit struct {
		ID        string     `json:"id"`
		Message   string     `json:"message"`
		Timestamp *time.Time `json:"timestamp"`
		URL       string     `json:"url"`
		Author    struct {
			Name  string `json:"name"`
			Email string `json:"email"`
		} `json:"author"`
	} `json:"commit"`
	Builds []struct {
		ID         int    `json:"id"`
		Stage      string `json:"stage"`
		Name       string `json:"name"`
		Status     string `json:"status"`
		CreatedAt  string `json:"created_at"`
		StartedAt  string `json:"started_at"`
		FinishedAt string `json:"finished_at"`
		When       string `json:"when"`
		Manual     bool   `json:"manual"`
		User       struct {
			Name      string `json:"name"`
			Username  string `json:"username"`
			AvatarURL string `json:"avatar_url"`
		} `json:"user"`
		Runner struct {
			ID          int    `json:"id"`
			Description string `json:"description"`
			Active      bool   `json:"active"`
			IsShared    bool   `json:"is_shared"`
		} `json:"runner"`
		ArtifactsFile struct {
			Filename string `json:"filename"`
			Size     int    `json:"size"`
		} `json:"artifacts_file"`
	} `json:"builds"`
}

//BuildEvent represents a build event
//
// GitLab API docs:
// https://docs.gitlab.com/ce/user/project/integrations/webhooks.html#build-events
type BuildEvent struct {
	ObjectKind        string  `json:"object_kind"`
	Ref               string  `json:"ref"`
	Tag               bool    `json:"tag"`
	BeforeSHA         string  `json:"before_sha"`
	SHA               string  `json:"sha"`
	BuildID           int     `json:"build_id"`
	BuildName         string  `json:"build_name"`
	BuildStage        string  `json:"build_stage"`
	BuildStatus       string  `json:"build_status"`
	BuildStartedAt    string  `json:"build_started_at"`
	BuildFinishedAt   string  `json:"build_finished_at"`
	BuildDuration     float64 `json:"build_duration"`
	BuildAllowFailure bool    `json:"build_allow_failure"`
	ProjectID         int     `json:"project_id"`
	ProjectName       string  `json:"project_name"`
	User              struct {
		ID    int    `json:"id"`
		Name  string `json:"name"`
		Email string `json:"email"`
	} `json:"user"`
	Commit struct {
		ID          int    `json:"id"`
		SHA         string `json:"sha"`
		Message     string `json:"message"`
		AuthorName  string `json:"author_name"`
		AuthorEmail string `json:"author_email"`
		Status      string `json:"status"`
		Duration    int    `json:"duration"`
		StartedAt   string `json:"started_at"`
		FinishedAt  string `json:"finished_at"`
	} `json:"commit"`
	Repository *Repository `json:"repository"`
}

// DeploymentEvent represents a deployment event
//
// GitLab API docs:
// https://docs.gitlab.com/ce/user/project/integrations/webhooks.html#deployment-events
type DeploymentEvent struct {
	ObjectKind    string `json:"object_kind"`
	Status        string `json:"status"`
	DeployableID  int    `json:"deployable_id"`
	DeployableURL string `json:"deployable_url"`
	Environment   string `json:"environment"`
	Project       struct {
		ID                int     `json:"id"`
		Name              string  `json:"name"`
		Description       string  `json:"description"`
		WebURL            string  `json:"web_url"`
		AvatarURL         *string `json:"avatar_url"`
		GitSSHURL         string  `json:"git_ssh_url"`
		GitHTTPURL        string  `json:"git_http_url"`
		Namespace         string  `json:"namespace"`
		VisibilityLevel   int     `json:"visibility_level"`
		PathWithNamespace string  `json:"path_with_namespace"`
		DefaultBranch     string  `json:"default_branch"`
		CIConfigPath      string  `json:"ci_config_path"`
		Homepage          string  `json:"homepage"`
		URL               string  `json:"url"`
		SSHURL            string  `json:"ssh_url"`
		HTTPURL           string  `json:"http_url"`
	} `json:"project"`
	ShortSHA string `json:"short_sha"`
	User     struct {
		Name      string `json:"name"`
		Username  string `json:"username"`
		AvatarURL string `json:"avatar_url"`
		Email     string `json:"email"`
	} `json:"user"`
	UserURL     string `json:"user_url"`
	CommitURL   string `json:"commit_url"`
	CommitTitle string `json:"commit_title"`
}
