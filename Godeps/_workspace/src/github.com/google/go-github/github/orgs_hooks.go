// Copyright 2015 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "fmt"

// ListHooks lists all Hooks for the specified organization.
//
// GitHub API docs: https://developer.github.com/v3/orgs/hooks/#list-hooks
func (s *OrganizationsService) ListHooks(org string, opt *ListOptions) ([]Hook, *Response, error) {
	u := fmt.Sprintf("orgs/%v/hooks", org)
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
// GitHub API docs: https://developer.github.com/v3/orgs/hooks/#get-single-hook
func (s *OrganizationsService) GetHook(org string, id int) (*Hook, *Response, error) {
	u := fmt.Sprintf("orgs/%v/hooks/%d", org, id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}
	hook := new(Hook)
	resp, err := s.client.Do(req, hook)
	return hook, resp, err
}

// CreateHook creates a Hook for the specified org.
// Name and Config are required fields.
//
// GitHub API docs: https://developer.github.com/v3/orgs/hooks/#create-a-hook
func (s *OrganizationsService) CreateHook(org string, hook *Hook) (*Hook, *Response, error) {
	u := fmt.Sprintf("orgs/%v/hooks", org)
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

// EditHook updates a specified Hook.
//
// GitHub API docs: https://developer.github.com/v3/orgs/hooks/#edit-a-hook
func (s *OrganizationsService) EditHook(org string, id int, hook *Hook) (*Hook, *Response, error) {
	u := fmt.Sprintf("orgs/%v/hooks/%d", org, id)
	req, err := s.client.NewRequest("PATCH", u, hook)
	if err != nil {
		return nil, nil, err
	}
	h := new(Hook)
	resp, err := s.client.Do(req, h)
	return h, resp, err
}

// PingHook triggers a 'ping' event to be sent to the Hook.
//
// GitHub API docs: https://developer.github.com/v3/orgs/hooks/#ping-a-hook
func (s *OrganizationsService) PingHook(org string, id int) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/hooks/%d/pings", org, id)
	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// DeleteHook deletes a specified Hook.
//
// GitHub API docs: https://developer.github.com/v3/orgs/hooks/#delete-a-hook
func (s *OrganizationsService) DeleteHook(org string, id int) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/hooks/%d", org, id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}
