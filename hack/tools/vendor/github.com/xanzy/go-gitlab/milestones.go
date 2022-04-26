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

// MilestonesService handles communication with the milestone related methods
// of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/milestones.html
type MilestonesService struct {
	client *Client
}

// Milestone represents a GitLab milestone.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/milestones.html
type Milestone struct {
	ID          int        `json:"id"`
	IID         int        `json:"iid"`
	ProjectID   int        `json:"project_id"`
	Title       string     `json:"title"`
	Description string     `json:"description"`
	StartDate   *ISOTime   `json:"start_date"`
	DueDate     *ISOTime   `json:"due_date"`
	State       string     `json:"state"`
	WebURL      string     `json:"web_url"`
	UpdatedAt   *time.Time `json:"updated_at"`
	CreatedAt   *time.Time `json:"created_at"`
	Expired     *bool      `json:"expired"`
}

func (m Milestone) String() string {
	return Stringify(m)
}

// ListMilestonesOptions represents the available ListMilestones() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/milestones.html#list-project-milestones
type ListMilestonesOptions struct {
	ListOptions
	IIDs   []int   `url:"iids,omitempty" json:"iids,omitempty"`
	Title  *string `url:"title,omitempty" json:"title,omitempty"`
	State  *string `url:"state,omitempty" json:"state,omitempty"`
	Search *string `url:"search,omitempty" json:"search,omitempty"`
}

// ListMilestones returns a list of project milestones.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/milestones.html#list-project-milestones
func (s *MilestonesService) ListMilestones(pid interface{}, opt *ListMilestonesOptions, options ...RequestOptionFunc) ([]*Milestone, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/milestones", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var m []*Milestone
	resp, err := s.client.Do(req, &m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// GetMilestone gets a single project milestone.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/milestones.html#get-single-milestone
func (s *MilestonesService) GetMilestone(pid interface{}, milestone int, options ...RequestOptionFunc) (*Milestone, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/milestones/%d", pathEscape(project), milestone)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	m := new(Milestone)
	resp, err := s.client.Do(req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// CreateMilestoneOptions represents the available CreateMilestone() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/milestones.html#create-new-milestone
type CreateMilestoneOptions struct {
	Title       *string  `url:"title,omitempty" json:"title,omitempty"`
	Description *string  `url:"description,omitempty" json:"description,omitempty"`
	StartDate   *ISOTime `url:"start_date,omitempty" json:"start_date,omitempty"`
	DueDate     *ISOTime `url:"due_date,omitempty" json:"due_date,omitempty"`
}

// CreateMilestone creates a new project milestone.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/milestones.html#create-new-milestone
func (s *MilestonesService) CreateMilestone(pid interface{}, opt *CreateMilestoneOptions, options ...RequestOptionFunc) (*Milestone, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/milestones", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	m := new(Milestone)
	resp, err := s.client.Do(req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// UpdateMilestoneOptions represents the available UpdateMilestone() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/milestones.html#edit-milestone
type UpdateMilestoneOptions struct {
	Title       *string  `url:"title,omitempty" json:"title,omitempty"`
	Description *string  `url:"description,omitempty" json:"description,omitempty"`
	StartDate   *ISOTime `url:"start_date,omitempty" json:"start_date,omitempty"`
	DueDate     *ISOTime `url:"due_date,omitempty" json:"due_date,omitempty"`
	StateEvent  *string  `url:"state_event,omitempty" json:"state_event,omitempty"`
}

// UpdateMilestone updates an existing project milestone.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/milestones.html#edit-milestone
func (s *MilestonesService) UpdateMilestone(pid interface{}, milestone int, opt *UpdateMilestoneOptions, options ...RequestOptionFunc) (*Milestone, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/milestones/%d", pathEscape(project), milestone)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	m := new(Milestone)
	resp, err := s.client.Do(req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// DeleteMilestone deletes a specified project milestone.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/milestones.html#delete-project-milestone
func (s *MilestonesService) DeleteMilestone(pid interface{}, milestone int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/milestones/%d", pathEscape(project), milestone)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// GetMilestoneIssuesOptions represents the available GetMilestoneIssues() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/milestones.html#get-all-issues-assigned-to-a-single-milestone
type GetMilestoneIssuesOptions ListOptions

// GetMilestoneIssues gets all issues assigned to a single project milestone.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/milestones.html#get-all-issues-assigned-to-a-single-milestone
func (s *MilestonesService) GetMilestoneIssues(pid interface{}, milestone int, opt *GetMilestoneIssuesOptions, options ...RequestOptionFunc) ([]*Issue, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/milestones/%d/issues", pathEscape(project), milestone)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var i []*Issue
	resp, err := s.client.Do(req, &i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, err
}

// GetMilestoneMergeRequestsOptions represents the available
// GetMilestoneMergeRequests() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/milestones.html#get-all-merge-requests-assigned-to-a-single-milestone
type GetMilestoneMergeRequestsOptions ListOptions

// GetMilestoneMergeRequests gets all merge requests assigned to a single
// project milestone.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/milestones.html#get-all-merge-requests-assigned-to-a-single-milestone
func (s *MilestonesService) GetMilestoneMergeRequests(pid interface{}, milestone int, opt *GetMilestoneMergeRequestsOptions, options ...RequestOptionFunc) ([]*MergeRequest, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/milestones/%d/merge_requests", pathEscape(project), milestone)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var mr []*MergeRequest
	resp, err := s.client.Do(req, &mr)
	if err != nil {
		return nil, resp, err
	}

	return mr, resp, err
}
