// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// WebHookPayload represents the data that is received from GitHub when a push
// event hook is triggered.  The format of these payloads pre-date most of the
// GitHub v3 API, so there are lots of minor incompatibilities with the types
// defined in the rest of the API.  Therefore, several types are duplicated
// here to account for these differences.
//
// GitHub API docs: https://help.github.com/articles/post-receive-hooks
type WebHookPayload struct {
	After      *string         `json:"after,omitempty"`
	Before     *string         `json:"before,omitempty"`
	Commits    []WebHookCommit `json:"commits,omitempty"`
	Compare    *string         `json:"compare,omitempty"`
	Created    *bool           `json:"created,omitempty"`
	Deleted    *bool           `json:"deleted,omitempty"`
	Forced     *bool           `json:"forced,omitempty"`
	HeadCommit *WebHookCommit  `json:"head_commit,omitempty"`
	Pusher     *User           `json:"pusher,omitempty"`
	Ref        *string         `json:"ref,omitempty"`
	Repo       *Repository     `json:"repository,omitempty"`
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
// in a WebHookCommit.  The commit author may not correspond to a GitHub User.
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
	CreatedAt *time.Time             `json:"created_at,omitempty"`
	UpdatedAt *time.Time             `json:"updated_at,omitempty"`
	Name      *string                `json:"name,omitempty"`
	Events    []string               `json:"events,omitempty"`
	Active    *bool                  `json:"active,omitempty"`
	Config    map[string]interface{} `json:"config,omitempty"`
	ID        *int                   `json:"id,omitempty"`
}

func (h Hook) String() string {
	return Stringify(h)
}

// CreateHook creates a Hook for the specified repository.
// Name and Config are required fields.
//
// GitHub API docs: http://developer.github.com/v3/repos/hooks/#create-a-hook
func (s *RepositoriesService) CreateHook(owner, repo string, hook *Hook) (*Hook, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/hooks", owner, repo)
	req, err := s.client.NewRequest("POST", u, hook)
	if err != nil {
		return nil, nil, err
	}

	h := new(Hook)
	resp, err := s.client.Do(req, h)
	if err != nil {
		return nil, resp, err
	}

	return h, resp, err
}

// ListHooks lists all Hooks for the specified repository.
//
// GitHub API docs: http://developer.github.com/v3/repos/hooks/#list
func (s *RepositoriesService) ListHooks(owner, repo string, opt *ListOptions) ([]Hook, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/hooks", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	hooks := new([]Hook)
	resp, err := s.client.Do(req, hooks)
	if err != nil {
		return nil, resp, err
	}

	return *hooks, resp, err
}

// GetHook returns a single specified Hook.
//
// GitHub API docs: http://developer.github.com/v3/repos/hooks/#get-single-hook
func (s *RepositoriesService) GetHook(owner, repo string, id int) (*Hook, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/hooks/%d", owner, repo, id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}
	hook := new(Hook)
	resp, err := s.client.Do(req, hook)
	return hook, resp, err
}

// EditHook updates a specified Hook.
//
// GitHub API docs: http://developer.github.com/v3/repos/hooks/#edit-a-hook
func (s *RepositoriesService) EditHook(owner, repo string, id int, hook *Hook) (*Hook, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/hooks/%d", owner, repo, id)
	req, err := s.client.NewRequest("PATCH", u, hook)
	if err != nil {
		return nil, nil, err
	}
	h := new(Hook)
	resp, err := s.client.Do(req, h)
	return h, resp, err
}

// DeleteHook deletes a specified Hook.
//
// GitHub API docs: http://developer.github.com/v3/repos/hooks/#delete-a-hook
func (s *RepositoriesService) DeleteHook(owner, repo string, id int) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/hooks/%d", owner, repo, id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// PingHook triggers a 'ping' event to be sent to the Hook.
//
// GitHub API docs: https://developer.github.com/v3/repos/hooks/#ping-a-hook
func (s *RepositoriesService) PingHook(owner, repo string, id int) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/hooks/%d/pings", owner, repo, id)
	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// TestHook triggers a test Hook by github.
//
// GitHub API docs: http://developer.github.com/v3/repos/hooks/#test-a-push-hook
func (s *RepositoriesService) TestHook(owner, repo string, id int) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/hooks/%d/tests", owner, repo, id)
	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// ListServiceHooks is deprecated.  Use Client.ListServiceHooks instead.
func (s *RepositoriesService) ListServiceHooks() ([]ServiceHook, *Response, error) {
	return s.client.ListServiceHooks()
}
