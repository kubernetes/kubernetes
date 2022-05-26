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

// EpicsService handles communication with the epic related methods
// of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/epics.html
type EpicsService struct {
	client *Client
}

// EpicAuthor represents a author of the epic.
type EpicAuthor struct {
	ID        int    `json:"id"`
	State     string `json:"state"`
	WebURL    string `json:"web_url"`
	Name      string `json:"name"`
	AvatarURL string `json:"avatar_url"`
	Username  string `json:"username"`
}

// Epic represents a GitLab epic.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/epics.html
type Epic struct {
	ID                      int         `json:"id"`
	IID                     int         `json:"iid"`
	GroupID                 int         `json:"group_id"`
	ParentID                int         `json:"parent_id"`
	Title                   string      `json:"title"`
	Description             string      `json:"description"`
	State                   string      `json:"state"`
	WebURL                  string      `json:"web_url"`
	Author                  *EpicAuthor `json:"author"`
	StartDate               *ISOTime    `json:"start_date"`
	StartDateIsFixed        bool        `json:"start_date_is_fixed"`
	StartDateFixed          *ISOTime    `json:"start_date_fixed"`
	StartDateFromMilestones *ISOTime    `json:"start_date_from_milestones"`
	DueDate                 *ISOTime    `json:"due_date"`
	DueDateIsFixed          bool        `json:"due_date_is_fixed"`
	DueDateFixed            *ISOTime    `json:"due_date_fixed"`
	DueDateFromMilestones   *ISOTime    `json:"due_date_from_milestones"`
	CreatedAt               *time.Time  `json:"created_at"`
	UpdatedAt               *time.Time  `json:"updated_at"`
	Labels                  []string    `json:"labels"`
	Upvotes                 int         `json:"upvotes"`
	Downvotes               int         `json:"downvotes"`
	UserNotesCount          int         `json:"user_notes_count"`
	URL                     string      `json:"url"`
}

func (e Epic) String() string {
	return Stringify(e)
}

// ListGroupEpicsOptions represents the available ListGroupEpics() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/epics.html#list-epics-for-a-group
type ListGroupEpicsOptions struct {
	ListOptions
	AuthorID                *int       `url:"author_id,omitempty" json:"author_id,omitempty"`
	Labels                  Labels     `url:"labels,comma,omitempty" json:"labels,omitempty"`
	WithLabelDetails        *bool      `url:"with_labels_details,omitempty" json:"with_labels_details,omitempty"`
	OrderBy                 *string    `url:"order_by,omitempty" json:"order_by,omitempty"`
	Sort                    *string    `url:"sort,omitempty" json:"sort,omitempty"`
	Search                  *string    `url:"search,omitempty" json:"search,omitempty"`
	State                   *string    `url:"state,omitempty" json:"state,omitempty"`
	CreatedAfter            *time.Time `url:"created_after,omitempty" json:"created_after,omitempty"`
	CreatedBefore           *time.Time `url:"created_before,omitempty" json:"created_before,omitempty"`
	UpdatedAfter            *time.Time `url:"updated_after,omitempty" json:"updated_after,omitempty"`
	UpdatedBefore           *time.Time `url:"updated_before,omitempty" json:"updated_before,omitempty"`
	IncludeAncestorGroups   *bool      `url:"include_ancestor_groups,omitempty" json:"include_ancestor_groups,omitempty"`
	IncludeDescendantGroups *bool      `url:"include_descendant_groups,omitempty" json:"include_descendant_groups,omitempty"`
	MyReactionEmoji         *string    `url:"my_reaction_emoji,omitempty" json:"my_reaction_emoji,omitempty"`
}

// ListGroupEpics gets a list of group epics. This function accepts pagination
// parameters page and per_page to return the list of group epics.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/epics.html#list-epics-for-a-group
func (s *EpicsService) ListGroupEpics(gid interface{}, opt *ListGroupEpicsOptions, options ...RequestOptionFunc) ([]*Epic, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/epics", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var es []*Epic
	resp, err := s.client.Do(req, &es)
	if err != nil {
		return nil, resp, err
	}

	return es, resp, err
}

// GetEpic gets a single group epic.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/epics.html#single-epic
func (s *EpicsService) GetEpic(gid interface{}, epic int, options ...RequestOptionFunc) (*Epic, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/epics/%d", pathEscape(group), epic)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	e := new(Epic)
	resp, err := s.client.Do(req, e)
	if err != nil {
		return nil, resp, err
	}

	return e, resp, err
}

// GetEpicLinks gets all child epics of an epic.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/epic_links.html
func (s *EpicsService) GetEpicLinks(gid interface{}, epic int, options ...RequestOptionFunc) ([]*Epic, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/epics/%d/epics", pathEscape(group), epic)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var e []*Epic
	resp, err := s.client.Do(req, &e)
	if err != nil {
		return nil, resp, err
	}

	return e, resp, err
}

// CreateEpicOptions represents the available CreateEpic() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/epics.html#new-epic
type CreateEpicOptions struct {
	Title            *string  `url:"title,omitempty" json:"title,omitempty"`
	Description      *string  `url:"description,omitempty" json:"description,omitempty"`
	Labels           Labels   `url:"labels,comma,omitempty" json:"labels,omitempty"`
	StartDateIsFixed *bool    `url:"start_date_is_fixed,omitempty" json:"start_date_is_fixed,omitempty"`
	StartDateFixed   *ISOTime `url:"start_date_fixed,omitempty" json:"start_date_fixed,omitempty"`
	DueDateIsFixed   *bool    `url:"due_date_is_fixed,omitempty" json:"due_date_is_fixed,omitempty"`
	DueDateFixed     *ISOTime `url:"due_date_fixed,omitempty" json:"due_date_fixed,omitempty"`
}

// CreateEpic creates a new group epic.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/epics.html#new-epic
func (s *EpicsService) CreateEpic(gid interface{}, opt *CreateEpicOptions, options ...RequestOptionFunc) (*Epic, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/epics", pathEscape(group))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	e := new(Epic)
	resp, err := s.client.Do(req, e)
	if err != nil {
		return nil, resp, err
	}

	return e, resp, err
}

// UpdateEpicOptions represents the available UpdateEpic() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/epics.html#update-epic
type UpdateEpicOptions struct {
	Title            *string  `url:"title,omitempty" json:"title,omitempty"`
	Description      *string  `url:"description,omitempty" json:"description,omitempty"`
	Labels           Labels   `url:"labels,comma,omitempty" json:"labels,omitempty"`
	StartDateIsFixed *bool    `url:"start_date_is_fixed,omitempty" json:"start_date_is_fixed,omitempty"`
	StartDateFixed   *ISOTime `url:"start_date_fixed,omitempty" json:"start_date_fixed,omitempty"`
	DueDateIsFixed   *bool    `url:"due_date_is_fixed,omitempty" json:"due_date_is_fixed,omitempty"`
	DueDateFixed     *ISOTime `url:"due_date_fixed,omitempty" json:"due_date_fixed,omitempty"`
	StateEvent       *string  `url:"state_event,omitempty" json:"state_event,omitempty"`
}

// UpdateEpic updates an existing group epic. This function is also used
// to mark an epic as closed.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/epics.html#update-epic
func (s *EpicsService) UpdateEpic(gid interface{}, epic int, opt *UpdateEpicOptions, options ...RequestOptionFunc) (*Epic, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/epics/%d", pathEscape(group), epic)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	e := new(Epic)
	resp, err := s.client.Do(req, e)
	if err != nil {
		return nil, resp, err
	}

	return e, resp, err
}

// DeleteEpic deletes a single group epic.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/epics.html#delete-epic
func (s *EpicsService) DeleteEpic(gid interface{}, epic int, options ...RequestOptionFunc) (*Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("groups/%s/epics/%d", pathEscape(group), epic)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
