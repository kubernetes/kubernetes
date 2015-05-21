// Copyright 2014 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// IssueEvent represents an event that occurred around an Issue or Pull Request.
type IssueEvent struct {
	ID  *int    `json:"id,omitempty"`
	URL *string `json:"url,omitempty"`

	// The User that generated this event.
	Actor *User `json:"actor,omitempty"`

	// Event identifies the actual type of Event that occurred.  Possible
	// values are:
	//
	//     closed
	//       The issue was closed by the actor. When the commit_id is
	//       present, it identifies the commit that closed the issue using
	//       “closes / fixes #NN” syntax.
	//
	//     reopened
	//       The issue was reopened by the actor.
	//
	//     subscribed
	//       The actor subscribed to receive notifications for an issue.
	//
	//     merged
	//       The issue was merged by the actor. The commit_id attribute is the SHA1 of the HEAD commit that was merged.
	//
	//     referenced
	//       The issue was referenced from a commit message. The commit_id attribute is the commit SHA1 of where that happened.
	//
	//     mentioned
	//       The actor was @mentioned in an issue body.
	//
	//     assigned
	//       The issue was assigned to the actor.
	//
	//     head_ref_deleted
	//       The pull request’s branch was deleted.
	//
	//     head_ref_restored
	//       The pull request’s branch was restored.
	Event *string `json:"event,omitempty"`

	// The SHA of the commit that referenced this commit, if applicable.
	CommitID *string `json:"commit_id,omitempty"`

	CreatedAt *time.Time `json:"created_at,omitempty"`
	Issue     *Issue     `json:"issue,omitempty"`
}

// ListIssueEvents lists events for the specified issue.
//
// GitHub API docs: https://developer.github.com/v3/issues/events/#list-events-for-an-issue
func (s *IssuesService) ListIssueEvents(owner, repo string, number int, opt *ListOptions) ([]IssueEvent, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%v/events", owner, repo, number)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var events []IssueEvent
	resp, err := s.client.Do(req, &events)
	if err != nil {
		return nil, resp, err
	}

	return events, resp, err
}

// ListRepositoryEvents lists events for the specified repository.
//
// GitHub API docs: https://developer.github.com/v3/issues/events/#list-events-for-a-repository
func (s *IssuesService) ListRepositoryEvents(owner, repo string, opt *ListOptions) ([]IssueEvent, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/events", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var events []IssueEvent
	resp, err := s.client.Do(req, &events)
	if err != nil {
		return nil, resp, err
	}

	return events, resp, err
}

// GetEvent returns the specified issue event.
//
// GitHub API docs: https://developer.github.com/v3/issues/events/#get-a-single-event
func (s *IssuesService) GetEvent(owner, repo string, id int) (*IssueEvent, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/events/%v", owner, repo, id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	event := new(IssueEvent)
	resp, err := s.client.Do(req, event)
	if err != nil {
		return nil, resp, err
	}

	return event, resp, err
}
