// Copyright 2018 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// PreReceiveHook represents a GitHub pre-receive hook for a repository.
type PreReceiveHook struct {
	ID          *int64  `json:"id,omitempty"`
	Name        *string `json:"name,omitempty"`
	Enforcement *string `json:"enforcement,omitempty"`
	ConfigURL   *string `json:"configuration_url,omitempty"`
}

func (p PreReceiveHook) String() string {
	return Stringify(p)
}

// ListPreReceiveHooks lists all pre-receive hooks for the specified repository.
//
// GitHub API docs: https://developer.github.com/enterprise/2.13/v3/repos/pre_receive_hooks/#list-pre-receive-hooks
func (s *RepositoriesService) ListPreReceiveHooks(ctx context.Context, owner, repo string, opts *ListOptions) ([]*PreReceiveHook, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pre-receive-hooks", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypePreReceiveHooksPreview)

	var hooks []*PreReceiveHook
	resp, err := s.client.Do(ctx, req, &hooks)
	if err != nil {
		return nil, resp, err
	}

	return hooks, resp, nil
}

// GetPreReceiveHook returns a single specified pre-receive hook.
//
// GitHub API docs: https://developer.github.com/enterprise/2.13/v3/repos/pre_receive_hooks/#get-a-single-pre-receive-hook
func (s *RepositoriesService) GetPreReceiveHook(ctx context.Context, owner, repo string, id int64) (*PreReceiveHook, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pre-receive-hooks/%d", owner, repo, id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypePreReceiveHooksPreview)

	h := new(PreReceiveHook)
	resp, err := s.client.Do(ctx, req, h)
	if err != nil {
		return nil, resp, err
	}

	return h, resp, nil
}

// UpdatePreReceiveHook updates a specified pre-receive hook.
//
// GitHub API docs: https://developer.github.com/enterprise/2.13/v3/repos/pre_receive_hooks/#update-pre-receive-hook-enforcement
func (s *RepositoriesService) UpdatePreReceiveHook(ctx context.Context, owner, repo string, id int64, hook *PreReceiveHook) (*PreReceiveHook, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pre-receive-hooks/%d", owner, repo, id)
	req, err := s.client.NewRequest("PATCH", u, hook)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypePreReceiveHooksPreview)

	h := new(PreReceiveHook)
	resp, err := s.client.Do(ctx, req, h)
	if err != nil {
		return nil, resp, err
	}

	return h, resp, nil
}

// DeletePreReceiveHook deletes a specified pre-receive hook.
//
// GitHub API docs: https://developer.github.com/enterprise/2.13/v3/repos/pre_receive_hooks/#remove-enforcement-overrides-for-a-pre-receive-hook
func (s *RepositoriesService) DeletePreReceiveHook(ctx context.Context, owner, repo string, id int64) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pre-receive-hooks/%d", owner, repo, id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypePreReceiveHooksPreview)

	return s.client.Do(ctx, req, nil)
}
