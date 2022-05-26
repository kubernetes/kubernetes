// Copyright 2014 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
	"time"
)

// Milestone represents a GitHub repository milestone.
type Milestone struct {
	URL          *string    `json:"url,omitempty"`
	HTMLURL      *string    `json:"html_url,omitempty"`
	LabelsURL    *string    `json:"labels_url,omitempty"`
	ID           *int64     `json:"id,omitempty"`
	Number       *int       `json:"number,omitempty"`
	State        *string    `json:"state,omitempty"`
	Title        *string    `json:"title,omitempty"`
	Description  *string    `json:"description,omitempty"`
	Creator      *User      `json:"creator,omitempty"`
	OpenIssues   *int       `json:"open_issues,omitempty"`
	ClosedIssues *int       `json:"closed_issues,omitempty"`
	CreatedAt    *time.Time `json:"created_at,omitempty"`
	UpdatedAt    *time.Time `json:"updated_at,omitempty"`
	ClosedAt     *time.Time `json:"closed_at,omitempty"`
	DueOn        *time.Time `json:"due_on,omitempty"`
	NodeID       *string    `json:"node_id,omitempty"`
}

func (m Milestone) String() string {
	return Stringify(m)
}

// MilestoneListOptions specifies the optional parameters to the
// IssuesService.ListMilestones method.
type MilestoneListOptions struct {
	// State filters milestones based on their state. Possible values are:
	// open, closed, all. Default is "open".
	State string `url:"state,omitempty"`

	// Sort specifies how to sort milestones. Possible values are: due_on, completeness.
	// Default value is "due_on".
	Sort string `url:"sort,omitempty"`

	// Direction in which to sort milestones. Possible values are: asc, desc.
	// Default is "asc".
	Direction string `url:"direction,omitempty"`

	ListOptions
}

// ListMilestones lists all milestones for a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#list-milestones
func (s *IssuesService) ListMilestones(ctx context.Context, owner string, repo string, opts *MilestoneListOptions) ([]*Milestone, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/milestones", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var milestones []*Milestone
	resp, err := s.client.Do(ctx, req, &milestones)
	if err != nil {
		return nil, resp, err
	}

	return milestones, resp, nil
}

// GetMilestone gets a single milestone.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#get-a-milestone
func (s *IssuesService) GetMilestone(ctx context.Context, owner string, repo string, number int) (*Milestone, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/milestones/%d", owner, repo, number)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	milestone := new(Milestone)
	resp, err := s.client.Do(ctx, req, milestone)
	if err != nil {
		return nil, resp, err
	}

	return milestone, resp, nil
}

// CreateMilestone creates a new milestone on the specified repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#create-a-milestone
func (s *IssuesService) CreateMilestone(ctx context.Context, owner string, repo string, milestone *Milestone) (*Milestone, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/milestones", owner, repo)
	req, err := s.client.NewRequest("POST", u, milestone)
	if err != nil {
		return nil, nil, err
	}

	m := new(Milestone)
	resp, err := s.client.Do(ctx, req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, nil
}

// EditMilestone edits a milestone.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#update-a-milestone
func (s *IssuesService) EditMilestone(ctx context.Context, owner string, repo string, number int, milestone *Milestone) (*Milestone, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/milestones/%d", owner, repo, number)
	req, err := s.client.NewRequest("PATCH", u, milestone)
	if err != nil {
		return nil, nil, err
	}

	m := new(Milestone)
	resp, err := s.client.Do(ctx, req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, nil
}

// DeleteMilestone deletes a milestone.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#delete-a-milestone
func (s *IssuesService) DeleteMilestone(ctx context.Context, owner string, repo string, number int) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/milestones/%d", owner, repo, number)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
