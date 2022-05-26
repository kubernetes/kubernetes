// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// Label represents a GitHub label on an Issue
type Label struct {
	ID          *int64  `json:"id,omitempty"`
	URL         *string `json:"url,omitempty"`
	Name        *string `json:"name,omitempty"`
	Color       *string `json:"color,omitempty"`
	Description *string `json:"description,omitempty"`
	Default     *bool   `json:"default,omitempty"`
	NodeID      *string `json:"node_id,omitempty"`
}

func (l Label) String() string {
	return Stringify(l)
}

// ListLabels lists all labels for a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#list-labels-for-a-repository
func (s *IssuesService) ListLabels(ctx context.Context, owner string, repo string, opts *ListOptions) ([]*Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/labels", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var labels []*Label
	resp, err := s.client.Do(ctx, req, &labels)
	if err != nil {
		return nil, resp, err
	}

	return labels, resp, nil
}

// GetLabel gets a single label.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#get-a-label
func (s *IssuesService) GetLabel(ctx context.Context, owner string, repo string, name string) (*Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/labels/%v", owner, repo, name)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	label := new(Label)
	resp, err := s.client.Do(ctx, req, label)
	if err != nil {
		return nil, resp, err
	}

	return label, resp, nil
}

// CreateLabel creates a new label on the specified repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#create-a-label
func (s *IssuesService) CreateLabel(ctx context.Context, owner string, repo string, label *Label) (*Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/labels", owner, repo)
	req, err := s.client.NewRequest("POST", u, label)
	if err != nil {
		return nil, nil, err
	}

	l := new(Label)
	resp, err := s.client.Do(ctx, req, l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, nil
}

// EditLabel edits a label.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#update-a-label
func (s *IssuesService) EditLabel(ctx context.Context, owner string, repo string, name string, label *Label) (*Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/labels/%v", owner, repo, name)
	req, err := s.client.NewRequest("PATCH", u, label)
	if err != nil {
		return nil, nil, err
	}

	l := new(Label)
	resp, err := s.client.Do(ctx, req, l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, nil
}

// DeleteLabel deletes a label.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#delete-a-label
func (s *IssuesService) DeleteLabel(ctx context.Context, owner string, repo string, name string) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/labels/%v", owner, repo, name)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}

// ListLabelsByIssue lists all labels for an issue.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#list-labels-for-an-issue
func (s *IssuesService) ListLabelsByIssue(ctx context.Context, owner string, repo string, number int, opts *ListOptions) ([]*Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d/labels", owner, repo, number)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var labels []*Label
	resp, err := s.client.Do(ctx, req, &labels)
	if err != nil {
		return nil, resp, err
	}

	return labels, resp, nil
}

// AddLabelsToIssue adds labels to an issue.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#add-labels-to-an-issue
func (s *IssuesService) AddLabelsToIssue(ctx context.Context, owner string, repo string, number int, labels []string) ([]*Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d/labels", owner, repo, number)
	req, err := s.client.NewRequest("POST", u, labels)
	if err != nil {
		return nil, nil, err
	}

	var l []*Label
	resp, err := s.client.Do(ctx, req, &l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, nil
}

// RemoveLabelForIssue removes a label for an issue.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#remove-a-label-from-an-issue
func (s *IssuesService) RemoveLabelForIssue(ctx context.Context, owner string, repo string, number int, label string) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d/labels/%v", owner, repo, number, label)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// ReplaceLabelsForIssue replaces all labels for an issue.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#set-labels-for-an-issue
func (s *IssuesService) ReplaceLabelsForIssue(ctx context.Context, owner string, repo string, number int, labels []string) ([]*Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d/labels", owner, repo, number)
	req, err := s.client.NewRequest("PUT", u, labels)
	if err != nil {
		return nil, nil, err
	}

	var l []*Label
	resp, err := s.client.Do(ctx, req, &l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, nil
}

// RemoveLabelsForIssue removes all labels for an issue.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#remove-all-labels-from-an-issue
func (s *IssuesService) RemoveLabelsForIssue(ctx context.Context, owner string, repo string, number int) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d/labels", owner, repo, number)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// ListLabelsForMilestone lists labels for every issue in a milestone.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#list-labels-for-issues-in-a-milestone
func (s *IssuesService) ListLabelsForMilestone(ctx context.Context, owner string, repo string, number int, opts *ListOptions) ([]*Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/milestones/%d/labels", owner, repo, number)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var labels []*Label
	resp, err := s.client.Do(ctx, req, &labels)
	if err != nil {
		return nil, resp, err
	}

	return labels, resp, nil
}
