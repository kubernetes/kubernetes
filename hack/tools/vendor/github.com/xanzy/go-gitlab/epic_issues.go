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

import "fmt"

// EpicIssuesService handles communication with the epic issue related methods
// of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/epic_issues.html
type EpicIssuesService struct {
	client *Client
}

// EpicIssueAssignment contains both the epic and issue objects returned from
// Gitlab with the assignment ID.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/epic_issues.html
type EpicIssueAssignment struct {
	ID    int    `json:"id"`
	Epic  *Epic  `json:"epic"`
	Issue *Issue `json:"issue"`
}

// ListEpicIssues get a list of epic issues.
//
// Gitlab API docs:
// https://docs.gitlab.com/ee/api/epic_issues.html#list-issues-for-an-epic
func (s *EpicIssuesService) ListEpicIssues(gid interface{}, epic int, opt *ListOptions, options ...RequestOptionFunc) ([]*Issue, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/epics/%d/issues", pathEscape(group), epic)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var is []*Issue
	resp, err := s.client.Do(req, &is)
	if err != nil {
		return nil, resp, err
	}

	return is, resp, err
}

// AssignEpicIssue assigns an existing issue to an epic.
//
// Gitlab API Docs:
// https://docs.gitlab.com/ee/api/epic_issues.html#assign-an-issue-to-the-epic
func (s *EpicIssuesService) AssignEpicIssue(gid interface{}, epic, issue int, options ...RequestOptionFunc) (*EpicIssueAssignment, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/epics/%d/issues/%d", pathEscape(group), epic, issue)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	a := new(EpicIssueAssignment)
	resp, err := s.client.Do(req, a)
	if err != nil {
		return nil, resp, err
	}

	return a, resp, err
}

// RemoveEpicIssue removes an issue from an epic.
//
// Gitlab API Docs:
// https://docs.gitlab.com/ee/api/epic_issues.html#remove-an-issue-from-the-epic
func (s *EpicIssuesService) RemoveEpicIssue(gid interface{}, epic, epicIssue int, options ...RequestOptionFunc) (*EpicIssueAssignment, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/epics/%d/issues/%d", pathEscape(group), epic, epicIssue)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	a := new(EpicIssueAssignment)
	resp, err := s.client.Do(req, a)
	if err != nil {
		return nil, resp, err
	}

	return a, resp, err
}

// UpdateEpicIsssueAssignmentOptions describes the UpdateEpicIssueAssignment()
// options.
//
// Gitlab API Docs:
// https://docs.gitlab.com/ee/api/epic_issues.html#update-epic---issue-association
type UpdateEpicIsssueAssignmentOptions struct {
	*ListOptions
	MoveBeforeID *int `url:"move_before_id,omitempty" json:"move_before_id,omitempty"`
	MoveAfterID  *int `url:"move_after_id,omitempty" json:"move_after_id,omitempty"`
}

// UpdateEpicIssueAssignment moves an issue before or after another issue in an
// epic issue list.
//
// Gitlab API Docs:
// https://docs.gitlab.com/ee/api/epic_issues.html#update-epic---issue-association
func (s *EpicIssuesService) UpdateEpicIssueAssignment(gid interface{}, epic, epicIssue int, opt *UpdateEpicIsssueAssignmentOptions, options ...RequestOptionFunc) ([]*Issue, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/epics/%d/issues/%d", pathEscape(group), epic, epicIssue)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var is []*Issue
	resp, err := s.client.Do(req, &is)
	if err != nil {
		return nil, resp, err
	}

	return is, resp, err
}
