// Copyright 2015 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// ListHooks lists all Hooks for the specified organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#list-organization-webhooks
func (s *OrganizationsService) ListHooks(ctx context.Context, org string, opts *ListOptions) ([]*Hook, *Response, error) {
	u := fmt.Sprintf("orgs/%v/hooks", org)
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
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#get-an-organization-webhook
func (s *OrganizationsService) GetHook(ctx context.Context, org string, id int64) (*Hook, *Response, error) {
	u := fmt.Sprintf("orgs/%v/hooks/%d", org, id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}
	hook := new(Hook)
	resp, err := s.client.Do(ctx, req, hook)
	return hook, resp, err
}

// CreateHook creates a Hook for the specified org.
// Config is a required field.
//
// Note that only a subset of the hook fields are used and hook must
// not be nil.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#create-an-organization-webhook
func (s *OrganizationsService) CreateHook(ctx context.Context, org string, hook *Hook) (*Hook, *Response, error) {
	u := fmt.Sprintf("orgs/%v/hooks", org)

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

// EditHook updates a specified Hook.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#update-an-organization-webhook
func (s *OrganizationsService) EditHook(ctx context.Context, org string, id int64, hook *Hook) (*Hook, *Response, error) {
	u := fmt.Sprintf("orgs/%v/hooks/%d", org, id)
	req, err := s.client.NewRequest("PATCH", u, hook)
	if err != nil {
		return nil, nil, err
	}
	h := new(Hook)
	resp, err := s.client.Do(ctx, req, h)
	return h, resp, err
}

// PingHook triggers a 'ping' event to be sent to the Hook.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#ping-an-organization-webhook
func (s *OrganizationsService) PingHook(ctx context.Context, org string, id int64) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/hooks/%d/pings", org, id)
	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}

// DeleteHook deletes a specified Hook.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/orgs/#delete-an-organization-webhook
func (s *OrganizationsService) DeleteHook(ctx context.Context, org string, id int64) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/hooks/%d", org, id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}
