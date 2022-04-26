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
	"time"
)

// TodosService handles communication with the todos related methods of
// the Gitlab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/todos.html
type TodosService struct {
	client *Client
}

// TodoAction represents the available actions that can be performed on a todo.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/todos.html
type TodoAction string

// The available todo actions.
const (
	TodoAssigned          TodoAction = "assigned"
	TodoMentioned         TodoAction = "mentioned"
	TodoBuildFailed       TodoAction = "build_failed"
	TodoMarked            TodoAction = "marked"
	TodoApprovalRequired  TodoAction = "approval_required"
	TodoDirectlyAddressed TodoAction = "directly_addressed"
)

// TodoTarget represents a todo target of type Issue or MergeRequest
type TodoTarget struct {
	// TODO: replace both Assignee and Author structs with v4 User struct
	Assignee struct {
		Name      string `json:"name"`
		Username  string `json:"username"`
		ID        int    `json:"id"`
		State     string `json:"state"`
		AvatarURL string `json:"avatar_url"`
		WebURL    string `json:"web_url"`
	} `json:"assignee"`
	Author struct {
		Name      string `json:"name"`
		Username  string `json:"username"`
		ID        int    `json:"id"`
		State     string `json:"state"`
		AvatarURL string `json:"avatar_url"`
		WebURL    string `json:"web_url"`
	} `json:"author"`
	CreatedAt      *time.Time `json:"created_at"`
	Description    string     `json:"description"`
	Downvotes      int        `json:"downvotes"`
	ID             int        `json:"id"`
	IID            int        `json:"iid"`
	Labels         []string   `json:"labels"`
	Milestone      Milestone  `json:"milestone"`
	ProjectID      int        `json:"project_id"`
	State          string     `json:"state"`
	Subscribed     bool       `json:"subscribed"`
	Title          string     `json:"title"`
	UpdatedAt      *time.Time `json:"updated_at"`
	Upvotes        int        `json:"upvotes"`
	UserNotesCount int        `json:"user_notes_count"`
	WebURL         string     `json:"web_url"`

	// Only available for type Issue
	Confidential bool   `json:"confidential"`
	DueDate      string `json:"due_date"`
	Weight       int    `json:"weight"`

	// Only available for type MergeRequest
	ApprovalsBeforeMerge      int    `json:"approvals_before_merge"`
	ForceRemoveSourceBranch   bool   `json:"force_remove_source_branch"`
	MergeCommitSHA            string `json:"merge_commit_sha"`
	MergeWhenPipelineSucceeds bool   `json:"merge_when_pipeline_succeeds"`
	MergeStatus               string `json:"merge_status"`
	SHA                       string `json:"sha"`
	ShouldRemoveSourceBranch  bool   `json:"should_remove_source_branch"`
	SourceBranch              string `json:"source_branch"`
	SourceProjectID           int    `json:"source_project_id"`
	Squash                    bool   `json:"squash"`
	TargetBranch              string `json:"target_branch"`
	TargetProjectID           int    `json:"target_project_id"`
	WorkInProgress            bool   `json:"work_in_progress"`
}

// Todo represents a GitLab todo.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/todos.html
type Todo struct {
	ID      int `json:"id"`
	Project struct {
		ID                int    `json:"id"`
		HTTPURLToRepo     string `json:"http_url_to_repo"`
		WebURL            string `json:"web_url"`
		Name              string `json:"name"`
		NameWithNamespace string `json:"name_with_namespace"`
		Path              string `json:"path"`
		PathWithNamespace string `json:"path_with_namespace"`
	} `json:"project"`
	Author struct {
		ID        int    `json:"id"`
		Name      string `json:"name"`
		Username  string `json:"username"`
		State     string `json:"state"`
		AvatarURL string `json:"avatar_url"`
		WebURL    string `json:"web_url"`
	} `json:"author"`
	ActionName TodoAction `json:"action_name"`
	TargetType string     `json:"target_type"`
	Target     TodoTarget `json:"target"`
	TargetURL  string     `json:"target_url"`
	Body       string     `json:"body"`
	State      string     `json:"state"`
	CreatedAt  *time.Time `json:"created_at"`
}

func (t Todo) String() string {
	return Stringify(t)
}

// ListTodosOptions represents the available ListTodos() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/todos.html#get-a-list-of-todos
type ListTodosOptions struct {
	ListOptions
	Action    *TodoAction `url:"action,omitempty" json:"action,omitempty"`
	AuthorID  *int        `url:"author_id,omitempty" json:"author_id,omitempty"`
	ProjectID *int        `url:"project_id,omitempty" json:"project_id,omitempty"`
	State     *string     `url:"state,omitempty" json:"state,omitempty"`
	Type      *string     `url:"type,omitempty" json:"type,omitempty"`
}

// ListTodos lists all todos created by authenticated user.
// When no filter is applied, it returns all pending todos for the current user.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/todos.html#get-a-list-of-todos
func (s *TodosService) ListTodos(opt *ListTodosOptions, options ...RequestOptionFunc) ([]*Todo, *Response, error) {
	req, err := s.client.NewRequest("GET", "todos", opt, options)
	if err != nil {
		return nil, nil, err
	}

	var t []*Todo
	resp, err := s.client.Do(req, &t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

// MarkTodoAsDone marks a single pending todo given by its ID for the current user as done.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/todos.html#mark-a-todo-as-done
func (s *TodosService) MarkTodoAsDone(id int, options ...RequestOptionFunc) (*Response, error) {
	u := fmt.Sprintf("todos/%d/mark_as_done", id)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// MarkAllTodosAsDone marks all pending todos for the current user as done.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/todos.html#mark-all-todos-as-done
func (s *TodosService) MarkAllTodosAsDone(options ...RequestOptionFunc) (*Response, error) {
	req, err := s.client.NewRequest("POST", "todos/mark_as_done", nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
