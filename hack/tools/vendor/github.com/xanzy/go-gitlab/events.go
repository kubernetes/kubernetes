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

// EventsService handles communication with the event related methods of
// the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/events.html
type EventsService struct {
	client *Client
}

// ContributionEvent represents a user's contribution
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/events.html#get-user-contribution-events
type ContributionEvent struct {
	ID          int        `json:"id"`
	Title       string     `json:"title"`
	ProjectID   int        `json:"project_id"`
	ActionName  string     `json:"action_name"`
	TargetID    int        `json:"target_id"`
	TargetIID   int        `json:"target_iid"`
	TargetType  string     `json:"target_type"`
	AuthorID    int        `json:"author_id"`
	TargetTitle string     `json:"target_title"`
	CreatedAt   *time.Time `json:"created_at"`
	PushData    struct {
		CommitCount int    `json:"commit_count"`
		Action      string `json:"action"`
		RefType     string `json:"ref_type"`
		CommitFrom  string `json:"commit_from"`
		CommitTo    string `json:"commit_to"`
		Ref         string `json:"ref"`
		CommitTitle string `json:"commit_title"`
	} `json:"push_data"`
	Note   *Note `json:"note"`
	Author struct {
		Name      string `json:"name"`
		Username  string `json:"username"`
		ID        int    `json:"id"`
		State     string `json:"state"`
		AvatarURL string `json:"avatar_url"`
		WebURL    string `json:"web_url"`
	} `json:"author"`
	AuthorUsername string `json:"author_username"`
}

// ListContributionEventsOptions represents the options for GetUserContributionEvents
//
// GitLap API docs:
// https://docs.gitlab.com/ce/api/events.html#get-user-contribution-events
type ListContributionEventsOptions struct {
	ListOptions
	Action     *EventTypeValue       `url:"action,omitempty" json:"action,omitempty"`
	TargetType *EventTargetTypeValue `url:"target_type,omitempty" json:"target_type,omitempty"`
	Before     *ISOTime              `url:"before,omitempty" json:"before,omitempty"`
	After      *ISOTime              `url:"after,omitempty" json:"after,omitempty"`
	Sort       *string               `url:"sort,omitempty" json:"sort,omitempty"`
}

// ListUserContributionEvents retrieves user contribution events
// for the specified user, sorted from newest to oldest.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/events.html#get-user-contribution-events
func (s *UsersService) ListUserContributionEvents(uid interface{}, opt *ListContributionEventsOptions, options ...RequestOptionFunc) ([]*ContributionEvent, *Response, error) {
	user, err := parseID(uid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("users/%s/events", user)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var cs []*ContributionEvent
	resp, err := s.client.Do(req, &cs)
	if err != nil {
		return nil, resp, err
	}

	return cs, resp, err
}

// ListCurrentUserContributionEvents gets a list currently authenticated user's events
//
// GitLab API docs: https://docs.gitlab.com/ce/api/events.html#list-currently-authenticated-user-39-s-events
func (s *EventsService) ListCurrentUserContributionEvents(opt *ListContributionEventsOptions, options ...RequestOptionFunc) ([]*ContributionEvent, *Response, error) {
	req, err := s.client.NewRequest("GET", "events", opt, options)
	if err != nil {
		return nil, nil, err
	}

	var cs []*ContributionEvent
	resp, err := s.client.Do(req, &cs)
	if err != nil {
		return nil, resp, err
	}

	return cs, resp, err
}

// ListProjectVisibleEvents gets a list of visible events for a particular project
//
// GitLab API docs: https://docs.gitlab.com/ee/api/events.html#list-a-project-s-visible-events
func (s *EventsService) ListProjectVisibleEvents(pid interface{}, opt *ListContributionEventsOptions, options ...RequestOptionFunc) ([]*ContributionEvent, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/events", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var cs []*ContributionEvent
	resp, err := s.client.Do(req, &cs)
	if err != nil {
		return nil, resp, err
	}

	return cs, resp, err
}
