// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "fmt"

// Label represents a GitHub label on an Issue
type Label struct {
	URL   *string `json:"url,omitempty"`
	Name  *string `json:"name,omitempty"`
	Color *string `json:"color,omitempty"`
}

func (l Label) String() string {
	return fmt.Sprint(*l.Name)
}

// ListLabels lists all labels for a repository.
//
// GitHub API docs: http://developer.github.com/v3/issues/labels/#list-all-labels-for-this-repository
func (s *IssuesService) ListLabels(owner string, repo string, opt *ListOptions) ([]Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/labels", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	labels := new([]Label)
	resp, err := s.client.Do(req, labels)
	if err != nil {
		return nil, resp, err
	}

	return *labels, resp, err
}

// GetLabel gets a single label.
//
// GitHub API docs: http://developer.github.com/v3/issues/labels/#get-a-single-label
func (s *IssuesService) GetLabel(owner string, repo string, name string) (*Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/labels/%v", owner, repo, name)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	label := new(Label)
	resp, err := s.client.Do(req, label)
	if err != nil {
		return nil, resp, err
	}

	return label, resp, err
}

// CreateLabel creates a new label on the specified repository.
//
// GitHub API docs: http://developer.github.com/v3/issues/labels/#create-a-label
func (s *IssuesService) CreateLabel(owner string, repo string, label *Label) (*Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/labels", owner, repo)
	req, err := s.client.NewRequest("POST", u, label)
	if err != nil {
		return nil, nil, err
	}

	l := new(Label)
	resp, err := s.client.Do(req, l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, err
}

// EditLabel edits a label.
//
// GitHub API docs: http://developer.github.com/v3/issues/labels/#update-a-label
func (s *IssuesService) EditLabel(owner string, repo string, name string, label *Label) (*Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/labels/%v", owner, repo, name)
	req, err := s.client.NewRequest("PATCH", u, label)
	if err != nil {
		return nil, nil, err
	}

	l := new(Label)
	resp, err := s.client.Do(req, l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, err
}

// DeleteLabel deletes a label.
//
// GitHub API docs: http://developer.github.com/v3/issues/labels/#delete-a-label
func (s *IssuesService) DeleteLabel(owner string, repo string, name string) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/labels/%v", owner, repo, name)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// ListLabelsByIssue lists all labels for an issue.
//
// GitHub API docs: http://developer.github.com/v3/issues/labels/#list-all-labels-for-this-repository
func (s *IssuesService) ListLabelsByIssue(owner string, repo string, number int, opt *ListOptions) ([]Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d/labels", owner, repo, number)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	labels := new([]Label)
	resp, err := s.client.Do(req, labels)
	if err != nil {
		return nil, resp, err
	}

	return *labels, resp, err
}

// AddLabelsToIssue adds labels to an issue.
//
// GitHub API docs: http://developer.github.com/v3/issues/labels/#list-all-labels-for-this-repository
func (s *IssuesService) AddLabelsToIssue(owner string, repo string, number int, labels []string) ([]Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d/labels", owner, repo, number)
	req, err := s.client.NewRequest("POST", u, labels)
	if err != nil {
		return nil, nil, err
	}

	l := new([]Label)
	resp, err := s.client.Do(req, l)
	if err != nil {
		return nil, resp, err
	}

	return *l, resp, err
}

// RemoveLabelForIssue removes a label for an issue.
//
// GitHub API docs: http://developer.github.com/v3/issues/labels/#remove-a-label-from-an-issue
func (s *IssuesService) RemoveLabelForIssue(owner string, repo string, number int, label string) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d/labels/%v", owner, repo, number, label)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// ReplaceLabelsForIssue replaces all labels for an issue.
//
// GitHub API docs: http://developer.github.com/v3/issues/labels/#replace-all-labels-for-an-issue
func (s *IssuesService) ReplaceLabelsForIssue(owner string, repo string, number int, labels []string) ([]Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d/labels", owner, repo, number)
	req, err := s.client.NewRequest("PUT", u, labels)
	if err != nil {
		return nil, nil, err
	}

	l := new([]Label)
	resp, err := s.client.Do(req, l)
	if err != nil {
		return nil, resp, err
	}

	return *l, resp, err
}

// RemoveLabelsForIssue removes all labels for an issue.
//
// GitHub API docs: http://developer.github.com/v3/issues/labels/#remove-all-labels-from-an-issue
func (s *IssuesService) RemoveLabelsForIssue(owner string, repo string, number int) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%d/labels", owner, repo, number)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// ListLabelsForMilestone lists labels for every issue in a milestone.
//
// GitHub API docs: http://developer.github.com/v3/issues/labels/#get-labels-for-every-issue-in-a-milestone
func (s *IssuesService) ListLabelsForMilestone(owner string, repo string, number int, opt *ListOptions) ([]Label, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/milestones/%d/labels", owner, repo, number)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	labels := new([]Label)
	resp, err := s.client.Do(req, labels)
	if err != nil {
		return nil, resp, err
	}

	return *labels, resp, err
}
