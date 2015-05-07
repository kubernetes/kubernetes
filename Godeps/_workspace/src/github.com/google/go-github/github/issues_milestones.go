// Copyright 2014 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"time"
)

// Milestone represents a Github repository milestone.
type Milestone struct {
	URL          *string    `json:"url,omitempty"`
	Number       *int       `json:"number,omitempty"`
	State        *string    `json:"state,omitempty"`
	Title        *string    `json:"title,omitempty"`
	Description  *string    `json:"description,omitempty"`
	Creator      *User      `json:"creator,omitempty"`
	OpenIssues   *int       `json:"open_issues,omitempty"`
	ClosedIssues *int       `json:"closed_issues,omitempty"`
	CreatedAt    *time.Time `json:"created_at,omitempty"`
	UpdatedAt    *time.Time `json:"updated_at,omitempty"`
	DueOn        *time.Time `json:"due_on,omitempty"`
}

func (m Milestone) String() string {
	return Stringify(m)
}

// MilestoneListOptions specifies the optional parameters to the
// IssuesService.ListMilestones method.
type MilestoneListOptions struct {
	// State filters milestones based on their state. Possible values are:
	// open, closed. Default is "open".
	State string `url:"state,omitempty"`

	// Sort specifies how to sort milestones. Possible values are: due_date, completeness.
	// Default value is "due_date".
	Sort string `url:"sort,omitempty"`

	// Direction in which to sort milestones. Possible values are: asc, desc.
	// Default is "asc".
	Direction string `url:"direction,omitempty"`
}

// ListMilestones lists all milestones for a repository.
//
// GitHub API docs: https://developer.github.com/v3/issues/milestones/#list-milestones-for-a-repository
func (s *IssuesService) ListMilestones(owner string, repo string, opt *MilestoneListOptions) ([]Milestone, *Response, error) {
	u := fmt.Sprintf("/repos/%v/%v/milestones", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	milestones := new([]Milestone)
	resp, err := s.client.Do(req, milestones)
	if err != nil {
		return nil, resp, err
	}

	return *milestones, resp, err
}

// GetMilestone gets a single milestone.
//
// GitHub API docs: https://developer.github.com/v3/issues/milestones/#get-a-single-milestone
func (s *IssuesService) GetMilestone(owner string, repo string, number int) (*Milestone, *Response, error) {
	u := fmt.Sprintf("/repos/%v/%v/milestones/%d", owner, repo, number)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	milestone := new(Milestone)
	resp, err := s.client.Do(req, milestone)
	if err != nil {
		return nil, resp, err
	}

	return milestone, resp, err
}

// CreateMilestone creates a new milestone on the specified repository.
//
// GitHub API docs: https://developer.github.com/v3/issues/milestones/#create-a-milestone
func (s *IssuesService) CreateMilestone(owner string, repo string, milestone *Milestone) (*Milestone, *Response, error) {
	u := fmt.Sprintf("/repos/%v/%v/milestones", owner, repo)
	req, err := s.client.NewRequest("POST", u, milestone)
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

// EditMilestone edits a milestone.
//
// GitHub API docs: https://developer.github.com/v3/issues/milestones/#update-a-milestone
func (s *IssuesService) EditMilestone(owner string, repo string, number int, milestone *Milestone) (*Milestone, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/milestones/%d", owner, repo, number)
	req, err := s.client.NewRequest("PATCH", u, milestone)
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

// DeleteMilestone deletes a milestone.
//
// GitHub API docs: https://developer.github.com/v3/issues/milestones/#delete-a-milestone
func (s *IssuesService) DeleteMilestone(owner string, repo string, number int) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/milestones/%d", owner, repo, number)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
