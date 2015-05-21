// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"encoding/json"
	"fmt"
	"time"
)

// Event represents a GitHub event.
type Event struct {
	Type       *string          `json:"type,omitempty"`
	Public     *bool            `json:"public"`
	RawPayload *json.RawMessage `json:"payload,omitempty"`
	Repo       *Repository      `json:"repo,omitempty"`
	Actor      *User            `json:"actor,omitempty"`
	Org        *Organization    `json:"org,omitempty"`
	CreatedAt  *time.Time       `json:"created_at,omitempty"`
	ID         *string          `json:"id,omitempty"`
}

func (e Event) String() string {
	return Stringify(e)
}

// Payload returns the parsed event payload. For recognized event types
// (PushEvent), a value of the corresponding struct type will be returned.
func (e *Event) Payload() (payload interface{}) {
	switch *e.Type {
	case "PushEvent":
		payload = &PushEvent{}
	}
	if err := json.Unmarshal(*e.RawPayload, &payload); err != nil {
		panic(err.Error())
	}
	return payload
}

// PushEvent represents a git push to a GitHub repository.
//
// GitHub API docs: http://developer.github.com/v3/activity/events/types/#pushevent
type PushEvent struct {
	PushID  *int              `json:"push_id,omitempty"`
	Head    *string           `json:"head,omitempty"`
	Ref     *string           `json:"ref,omitempty"`
	Size    *int              `json:"size,omitempty"`
	Commits []PushEventCommit `json:"commits,omitempty"`
	Repo    *Repository       `json:"repository,omitempty"`
}

func (p PushEvent) String() string {
	return Stringify(p)
}

// PushEventCommit represents a git commit in a GitHub PushEvent.
type PushEventCommit struct {
	SHA      *string       `json:"sha,omitempty"`
	Message  *string       `json:"message,omitempty"`
	Author   *CommitAuthor `json:"author,omitempty"`
	URL      *string       `json:"url,omitempty"`
	Distinct *bool         `json:"distinct,omitempty"`
	Added    []string      `json:"added,omitempty"`
	Removed  []string      `json:"removed,omitempty"`
	Modified []string      `json:"modified,omitempty"`
}

func (p PushEventCommit) String() string {
	return Stringify(p)
}

//PullRequestEvent represents the payload delivered by PullRequestEvent webhook
type PullRequestEvent struct {
	Action      *string      `json:"action,omitempty"`
	Number      *int         `json:"number,omitempty"`
	PullRequest *PullRequest `json:"pull_request,omitempty"`
	Repo        *Repository  `json:"repository,omitempty"`
	Sender      *User        `json:"sender,omitempty"`
}

// IssueActivityEvent represents the payload delivered by Issue webhook
type IssueActivityEvent struct {
	Action *string     `json:"action,omitempty"`
	Issue  *Issue      `json:"issue,omitempty"`
	Repo   *Repository `json:"repository,omitempty"`
	Sender *User       `json:"sender,omitempty"`
}

// IssueCommentEvent represents the payload delivered by IssueComment webhook
//
// This webhook also gets fired for comments on pull requests
type IssueCommentEvent struct {
	Action  *string       `json:"action,omitempty"`
	Issue   *Issue        `json:"issue,omitempty"`
	Comment *IssueComment `json:"comment,omitempty"`
	Repo    *Repository   `json:"repository,omitempty"`
	Sender  *User         `json:"sender,omitempty"`
}

// ListEvents drinks from the firehose of all public events across GitHub.
//
// GitHub API docs: http://developer.github.com/v3/activity/events/#list-public-events
func (s *ActivityService) ListEvents(opt *ListOptions) ([]Event, *Response, error) {
	u, err := addOptions("events", opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	events := new([]Event)
	resp, err := s.client.Do(req, events)
	if err != nil {
		return nil, resp, err
	}

	return *events, resp, err
}

// ListRepositoryEvents lists events for a repository.
//
// GitHub API docs: http://developer.github.com/v3/activity/events/#list-repository-events
func (s *ActivityService) ListRepositoryEvents(owner, repo string, opt *ListOptions) ([]Event, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/events", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	events := new([]Event)
	resp, err := s.client.Do(req, events)
	if err != nil {
		return nil, resp, err
	}

	return *events, resp, err
}

// ListIssueEventsForRepository lists issue events for a repository.
//
// GitHub API docs: http://developer.github.com/v3/activity/events/#list-issue-events-for-a-repository
func (s *ActivityService) ListIssueEventsForRepository(owner, repo string, opt *ListOptions) ([]Event, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/events", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	events := new([]Event)
	resp, err := s.client.Do(req, events)
	if err != nil {
		return nil, resp, err
	}

	return *events, resp, err
}

// ListEventsForRepoNetwork lists public events for a network of repositories.
//
// GitHub API docs: http://developer.github.com/v3/activity/events/#list-public-events-for-a-network-of-repositories
func (s *ActivityService) ListEventsForRepoNetwork(owner, repo string, opt *ListOptions) ([]Event, *Response, error) {
	u := fmt.Sprintf("networks/%v/%v/events", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	events := new([]Event)
	resp, err := s.client.Do(req, events)
	if err != nil {
		return nil, resp, err
	}

	return *events, resp, err
}

// ListEventsForOrganization lists public events for an organization.
//
// GitHub API docs: http://developer.github.com/v3/activity/events/#list-public-events-for-an-organization
func (s *ActivityService) ListEventsForOrganization(org string, opt *ListOptions) ([]Event, *Response, error) {
	u := fmt.Sprintf("orgs/%v/events", org)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	events := new([]Event)
	resp, err := s.client.Do(req, events)
	if err != nil {
		return nil, resp, err
	}

	return *events, resp, err
}

// ListEventsPerformedByUser lists the events performed by a user. If publicOnly is
// true, only public events will be returned.
//
// GitHub API docs: http://developer.github.com/v3/activity/events/#list-events-performed-by-a-user
func (s *ActivityService) ListEventsPerformedByUser(user string, publicOnly bool, opt *ListOptions) ([]Event, *Response, error) {
	var u string
	if publicOnly {
		u = fmt.Sprintf("users/%v/events/public", user)
	} else {
		u = fmt.Sprintf("users/%v/events", user)
	}
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	events := new([]Event)
	resp, err := s.client.Do(req, events)
	if err != nil {
		return nil, resp, err
	}

	return *events, resp, err
}

// ListEventsRecievedByUser lists the events recieved by a user. If publicOnly is
// true, only public events will be returned.
//
// GitHub API docs: http://developer.github.com/v3/activity/events/#list-events-that-a-user-has-received
func (s *ActivityService) ListEventsRecievedByUser(user string, publicOnly bool, opt *ListOptions) ([]Event, *Response, error) {
	var u string
	if publicOnly {
		u = fmt.Sprintf("users/%v/received_events/public", user)
	} else {
		u = fmt.Sprintf("users/%v/received_events", user)
	}
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	events := new([]Event)
	resp, err := s.client.Do(req, events)
	if err != nil {
		return nil, resp, err
	}

	return *events, resp, err
}

// ListUserEventsForOrganization provides the user’s organization dashboard. You
// must be authenticated as the user to view this.
//
// GitHub API docs: http://developer.github.com/v3/activity/events/#list-events-for-an-organization
func (s *ActivityService) ListUserEventsForOrganization(org, user string, opt *ListOptions) ([]Event, *Response, error) {
	u := fmt.Sprintf("users/%v/events/orgs/%v", user, org)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	events := new([]Event)
	resp, err := s.client.Do(req, events)
	if err != nil {
		return nil, resp, err
	}

	return *events, resp, err
}
