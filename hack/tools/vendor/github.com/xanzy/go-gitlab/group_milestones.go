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

// GroupMilestonesService handles communication with the milestone related
// methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/group_milestones.html
type GroupMilestonesService struct {
	client *Client
}

// GroupMilestone represents a GitLab milestone.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/group_milestones.html
type GroupMilestone struct {
	ID          int        `json:"id"`
	IID         int        `json:"iid"`
	GroupID     int        `json:"group_id"`
	Title       string     `json:"title"`
	Description string     `json:"description"`
	StartDate   *ISOTime   `json:"start_date"`
	DueDate     *ISOTime   `json:"due_date"`
	State       string     `json:"state"`
	UpdatedAt   *time.Time `json:"updated_at"`
	CreatedAt   *time.Time `json:"created_at"`
	Expired     *bool      `json:"expired"`
}

func (m GroupMilestone) String() string {
	return Stringify(m)
}

// ListGroupMilestonesOptions represents the available
// ListGroupMilestones() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/group_milestones.html#list-group-milestones
type ListGroupMilestonesOptions struct {
	ListOptions
	IIDs                    []int   `url:"iids,omitempty" json:"iids,omitempty"`
	State                   *string `url:"state,omitempty" json:"state,omitempty"`
	Title                   *string `url:"title,omitempty" json:"title,omitempty"`
	Search                  *string `url:"search,omitempty" json:"search,omitempty"`
	IncludeParentMilestones *bool   `url:"include_parent_milestones,omitempty" json:"include_parent_milestones,omitempty"`
}

// ListGroupMilestones returns a list of group milestones.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/group_milestones.html#list-group-milestones
func (s *GroupMilestonesService) ListGroupMilestones(gid interface{}, opt *ListGroupMilestonesOptions, options ...RequestOptionFunc) ([]*GroupMilestone, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/milestones", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var m []*GroupMilestone
	resp, err := s.client.Do(req, &m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// GetGroupMilestone gets a single group milestone.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/group_milestones.html#get-single-milestone
func (s *GroupMilestonesService) GetGroupMilestone(gid interface{}, milestone int, options ...RequestOptionFunc) (*GroupMilestone, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/milestones/%d", pathEscape(group), milestone)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	m := new(GroupMilestone)
	resp, err := s.client.Do(req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// CreateGroupMilestoneOptions represents the available CreateGroupMilestone() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/group_milestones.html#create-new-milestone
type CreateGroupMilestoneOptions struct {
	Title       *string  `url:"title,omitempty" json:"title,omitempty"`
	Description *string  `url:"description,omitempty" json:"description,omitempty"`
	StartDate   *ISOTime `url:"start_date,omitempty" json:"start_date,omitempty"`
	DueDate     *ISOTime `url:"due_date,omitempty" json:"due_date,omitempty"`
}

// CreateGroupMilestone creates a new group milestone.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/group_milestones.html#create-new-milestone
func (s *GroupMilestonesService) CreateGroupMilestone(gid interface{}, opt *CreateGroupMilestoneOptions, options ...RequestOptionFunc) (*GroupMilestone, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/milestones", pathEscape(group))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	m := new(GroupMilestone)
	resp, err := s.client.Do(req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// UpdateGroupMilestoneOptions represents the available UpdateGroupMilestone() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/group_milestones.html#edit-milestone
type UpdateGroupMilestoneOptions struct {
	Title       *string  `url:"title,omitempty" json:"title,omitempty"`
	Description *string  `url:"description,omitempty" json:"description,omitempty"`
	StartDate   *ISOTime `url:"start_date,omitempty" json:"start_date,omitempty"`
	DueDate     *ISOTime `url:"due_date,omitempty" json:"due_date,omitempty"`
	StateEvent  *string  `url:"state_event,omitempty" json:"state_event,omitempty"`
}

// UpdateGroupMilestone updates an existing group milestone.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/group_milestones.html#edit-milestone
func (s *GroupMilestonesService) UpdateGroupMilestone(gid interface{}, milestone int, opt *UpdateGroupMilestoneOptions, options ...RequestOptionFunc) (*GroupMilestone, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/milestones/%d", pathEscape(group), milestone)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	m := new(GroupMilestone)
	resp, err := s.client.Do(req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// GetGroupMilestoneIssuesOptions represents the available GetGroupMilestoneIssues() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/group_milestones.html#get-all-issues-assigned-to-a-single-milestone
type GetGroupMilestoneIssuesOptions ListOptions

// GetGroupMilestoneIssues gets all issues assigned to a single group milestone.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/group_milestones.html#get-all-issues-assigned-to-a-single-milestone
func (s *GroupMilestonesService) GetGroupMilestoneIssues(gid interface{}, milestone int, opt *GetGroupMilestoneIssuesOptions, options ...RequestOptionFunc) ([]*Issue, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/milestones/%d/issues", pathEscape(group), milestone)

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

// GetGroupMilestoneMergeRequestsOptions represents the available
// GetGroupMilestoneMergeRequests() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/group_milestones.html#get-all-merge-requests-assigned-to-a-single-milestone
type GetGroupMilestoneMergeRequestsOptions ListOptions

// GetGroupMilestoneMergeRequests gets all merge requests assigned to a
// single group milestone.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/group_milestones.html#get-all-merge-requests-assigned-to-a-single-milestone
func (s *GroupMilestonesService) GetGroupMilestoneMergeRequests(gid interface{}, milestone int, opt *GetGroupMilestoneMergeRequestsOptions, options ...RequestOptionFunc) ([]*MergeRequest, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/milestones/%d/merge_requests", pathEscape(group), milestone)

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

// BurndownChartEvent reprensents a burnout chart event
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_milestones.html#get-all-burndown-chart-events-for-a-single-milestone-starter
type BurndownChartEvent struct {
	CreatedAt *time.Time `json:"created_at"`
	Weight    *int       `json:"weight"`
	Action    *string    `json:"action"`
}

// GetGroupMilestoneBurndownChartEventsOptions represents the available
// GetGroupMilestoneBurndownChartEventsOptions() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_milestones.html#get-all-burndown-chart-events-for-a-single-milestone-starter
type GetGroupMilestoneBurndownChartEventsOptions ListOptions

// GetGroupMilestoneBurndownChartEvents gets all merge requests assigned to a
// single group milestone.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_milestones.html#get-all-burndown-chart-events-for-a-single-milestone-starter
func (s *GroupMilestonesService) GetGroupMilestoneBurndownChartEvents(gid interface{}, milestone int, opt *GetGroupMilestoneBurndownChartEventsOptions, options ...RequestOptionFunc) ([]*BurndownChartEvent, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/milestones/%d/burndown_events", pathEscape(group), milestone)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var be []*BurndownChartEvent
	resp, err := s.client.Do(req, &be)
	if err != nil {
		return nil, resp, err
	}

	return be, resp, err
}
