// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "fmt"

// ListAssignees fetches all available assignees (owners and collaborators) to
// which issues may be assigned.
//
// GitHub API docs: http://developer.github.com/v3/issues/assignees/#list-assignees
func (s *IssuesService) ListAssignees(owner string, repo string, opt *ListOptions) ([]User, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/assignees", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}
	assignees := new([]User)
	resp, err := s.client.Do(req, assignees)
	if err != nil {
		return nil, resp, err
	}

	return *assignees, resp, err
}

// IsAssignee checks if a user is an assignee for the specified repository.
//
// GitHub API docs: http://developer.github.com/v3/issues/assignees/#check-assignee
func (s *IssuesService) IsAssignee(owner string, repo string, user string) (bool, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/assignees/%v", owner, repo, user)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return false, nil, err
	}
	resp, err := s.client.Do(req, nil)
	assignee, err := parseBoolResponse(err)
	return assignee, resp, err
}
