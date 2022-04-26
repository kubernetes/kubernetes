// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// ListEvents drinks from the firehose of all public events across GitHub.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#list-public-events
func (s *ActivityService) ListEvents(ctx context.Context, opts *ListOptions) ([]*Event, *Response, error) {
	u, err := addOptions("events", opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var events []*Event
	resp, err := s.client.Do(ctx, req, &events)
	if err != nil {
		return nil, resp, err
	}

	return events, resp, nil
}

// ListRepositoryEvents lists events for a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#list-repository-events
func (s *ActivityService) ListRepositoryEvents(ctx context.Context, owner, repo string, opts *ListOptions) ([]*Event, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/events", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var events []*Event
	resp, err := s.client.Do(ctx, req, &events)
	if err != nil {
		return nil, resp, err
	}

	return events, resp, nil
}

// ListIssueEventsForRepository lists issue events for a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#list-issue-events-for-a-repository
func (s *ActivityService) ListIssueEventsForRepository(ctx context.Context, owner, repo string, opts *ListOptions) ([]*IssueEvent, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/events", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var events []*IssueEvent
	resp, err := s.client.Do(ctx, req, &events)
	if err != nil {
		return nil, resp, err
	}

	return events, resp, nil
}

// ListEventsForRepoNetwork lists public events for a network of repositories.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#list-public-events-for-a-network-of-repositories
func (s *ActivityService) ListEventsForRepoNetwork(ctx context.Context, owner, repo string, opts *ListOptions) ([]*Event, *Response, error) {
	u := fmt.Sprintf("networks/%v/%v/events", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var events []*Event
	resp, err := s.client.Do(ctx, req, &events)
	if err != nil {
		return nil, resp, err
	}

	return events, resp, nil
}

// ListEventsForOrganization lists public events for an organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#list-public-organization-events
func (s *ActivityService) ListEventsForOrganization(ctx context.Context, org string, opts *ListOptions) ([]*Event, *Response, error) {
	u := fmt.Sprintf("orgs/%v/events", org)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var events []*Event
	resp, err := s.client.Do(ctx, req, &events)
	if err != nil {
		return nil, resp, err
	}

	return events, resp, nil
}

// ListEventsPerformedByUser lists the events performed by a user. If publicOnly is
// true, only public events will be returned.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#list-events-for-the-authenticated-user
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#list-public-events-for-a-user
func (s *ActivityService) ListEventsPerformedByUser(ctx context.Context, user string, publicOnly bool, opts *ListOptions) ([]*Event, *Response, error) {
	var u string
	if publicOnly {
		u = fmt.Sprintf("users/%v/events/public", user)
	} else {
		u = fmt.Sprintf("users/%v/events", user)
	}
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var events []*Event
	resp, err := s.client.Do(ctx, req, &events)
	if err != nil {
		return nil, resp, err
	}

	return events, resp, nil
}

// ListEventsReceivedByUser lists the events received by a user. If publicOnly is
// true, only public events will be returned.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#list-events-received-by-the-authenticated-user
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#list-public-events-received-by-a-user
func (s *ActivityService) ListEventsReceivedByUser(ctx context.Context, user string, publicOnly bool, opts *ListOptions) ([]*Event, *Response, error) {
	var u string
	if publicOnly {
		u = fmt.Sprintf("users/%v/received_events/public", user)
	} else {
		u = fmt.Sprintf("users/%v/received_events", user)
	}
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var events []*Event
	resp, err := s.client.Do(ctx, req, &events)
	if err != nil {
		return nil, resp, err
	}

	return events, resp, nil
}

// ListUserEventsForOrganization provides the user’s organization dashboard. You
// must be authenticated as the user to view this.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/#list-organization-events-for-the-authenticated-user
func (s *ActivityService) ListUserEventsForOrganization(ctx context.Context, org, user string, opts *ListOptions) ([]*Event, *Response, error) {
	u := fmt.Sprintf("users/%v/events/orgs/%v", user, org)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var events []*Event
	resp, err := s.client.Do(ctx, req, &events)
	if err != nil {
		return nil, resp, err
	}

	return events, resp, nil
}
