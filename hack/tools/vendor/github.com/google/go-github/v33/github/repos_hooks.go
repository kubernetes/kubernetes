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

// WebHookPayload represents the data that is received from GitHub when a push
// event hook is triggered. The format of these payloads pre-date most of the
// GitHub v3 API, so there are lots of minor incompatibilities with the types
// defined in the rest of the API. Therefore, several types are duplicated
// here to account for these differences.
//
// GitHub API docs: https://help.github.com/articles/post-receive-hooks
type WebHookPayload struct {
	After      *string          `json:"after,omitempty"`
	Before     *string          `json:"before,omitempty"`
	Commits    []*WebHookCommit `json:"commits,omitempty"`
	Compare    *string          `json:"compare,omitempty"`
	Created    *bool            `json:"created,omitempty"`
	Deleted    *bool            `json:"deleted,omitempty"`
	Forced     *bool            `json:"forced,omitempty"`
	HeadCommit *WebHookCommit   `json:"head_commit,omitempty"`
	Pusher     *User            `json:"pusher,omitempty"`
	Ref        *string          `json:"ref,omitempty"`
	Repo       *Repository      `json:"repository,omitempty"`
	Sender     *User            `json:"sender,omitempty"`
}

func (w WebHookPayload) String() string {
	return Stringify(w)
}

// WebHookCommit represents the commit variant we receive from GitHub in a
// WebHookPayload.
type WebHookCommit struct {
	Added     []string       `json:"added,omitempty"`
	Author    *WebHookAuthor `json:"author,omitempty"`
	Committer *WebHookAuthor `json:"committer,omitempty"`
	Distinct  *bool          `json:"distinct,omitempty"`
	ID        *string        `json:"id,omitempty"`
	Message   *string        `json:"message,omitempty"`
	Modified  []string       `json:"modified,omitempty"`
	Removed   []string       `json:"removed,omitempty"`
	Timestamp *time.Time     `json:"timestamp,omitempty"`
}

func (w WebHookCommit) String() string {
	return Stringify(w)
}

// WebHookAuthor represents the author or committer of a commit, as specified
// in a WebHookCommit. The commit author may not correspond to a GitHub User.
type WebHookAuthor struct {
	Email    *string `json:"email,omitempty"`
	Name     *string `json:"name,omitempty"`
	Username *string `json:"username,omitempty"`
}

func (w WebHookAuthor) String() string {
	return Stringify(w)
}

// Hook represents a GitHub (web and service) hook for a repository.
type Hook struct {
	CreatedAt *time.Time `json:"created_at,omitempty"`
	UpdatedAt *time.Time `json:"updated_at,omitempty"`
	URL       *string    `json:"url,omitempty"`
	ID        *int64     `json:"id,omitempty"`

	// Only the following fields are used when creating a hook.
	// Config is required.
	Config map[string]interface{} `json:"config,omitempty"`
	Events []string               `json:"events,omitempty"`
	Active *bool                  `json:"active,omitempty"`
}

func (h Hook) String() string {
	return Stringify(h)
}

// createHookRequest is a subset of Hook and is used internally
// by CreateHook to pass only the known fields for the endpoint.
//
// See https://github.com/google/go-github/issues/1015 for more
// information.
type createHookRequest struct {
	// Config is required.
	Name   string                 `json:"name"`
	Config map[string]interface{} `json:"config,omitempty"`
	Events []string               `json:"events,omitempty"`
	Active *bool                  `json:"active,omitempty"`
}

// CreateHook creates a Hook for the specified repository.
// Config is a required field.
//
// Note that only a subset of the hook fields are used and hook must
// not be nil.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#create-a-repository-webhook
func (s *RepositoriesService) CreateHook(ctx context.Context, owner, repo string, hook *Hook) (*Hook, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/hooks", owner, repo)

	hookReq := &createHookRequest{
		Name:   "web",
		Events: hook.Events,
		Active: hook.Active,
		Config: hook.Config,
	}

	req, err := s.client.NewRequest("POST", u, hookReq)
	if err != nil {
		return nil, nil, err
	}

	h := new(Hook)
	resp, err := s.client.Do(ctx, req, h)
	if err != nil {
		return nil, resp, err
	}

	return h, resp, nil
}

// ListHooks lists all Hooks for the specified repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#list-repository-webhooks
func (s *RepositoriesService) ListHooks(ctx context.Context, owner, repo string, opts *ListOptions) ([]*Hook, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/hooks", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var hooks []*Hook
	resp, err := s.client.Do(ctx, req, &hooks)
	if err != nil {
		return nil, resp, err
	}

	return hooks, resp, nil
}

// GetHook returns a single specified Hook.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-a-repository-webhook
func (s *RepositoriesService) GetHook(ctx context.Context, owner, repo string, id int64) (*Hook, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/hooks/%d", owner, repo, id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}
	h := new(Hook)
	resp, err := s.client.Do(ctx, req, h)
	if err != nil {
		return nil, resp, err
	}

	return h, resp, nil
}

// EditHook updates a specified Hook.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#update-a-repository-webhook
func (s *RepositoriesService) EditHook(ctx context.Context, owner, repo string, id int64, hook *Hook) (*Hook, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/hooks/%d", owner, repo, id)
	req, err := s.client.NewRequest("PATCH", u, hook)
	if err != nil {
		return nil, nil, err
	}
	h := new(Hook)
	resp, err := s.client.Do(ctx, req, h)
	if err != nil {
		return nil, resp, err
	}

	return h, resp, nil
}

// DeleteHook deletes a specified Hook.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#delete-a-repository-webhook
func (s *RepositoriesService) DeleteHook(ctx context.Context, owner, repo string, id int64) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/hooks/%d", owner, repo, id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}

// PingHook triggers a 'ping' event to be sent to the Hook.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#ping-a-repository-webhook
func (s *RepositoriesService) PingHook(ctx context.Context, owner, repo string, id int64) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/hooks/%d/pings", owner, repo, id)
	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}

// TestHook triggers a test Hook by github.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#test-the-push-repository-webhook
func (s *RepositoriesService) TestHook(ctx context.Context, owner, repo string, id int64) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/hooks/%d/tests", owner, repo, id)
	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}
