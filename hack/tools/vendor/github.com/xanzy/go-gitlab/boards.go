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
)

// IssueBoardsService handles communication with the issue board related
// methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/boards.html
type IssueBoardsService struct {
	client *Client
}

// IssueBoard represents a GitLab issue board.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/boards.html
type IssueBoard struct {
	ID        int          `json:"id"`
	Name      string       `json:"name"`
	Project   *Project     `json:"project"`
	Milestone *Milestone   `json:"milestone"`
	Lists     []*BoardList `json:"lists"`
}

func (b IssueBoard) String() string {
	return Stringify(b)
}

// BoardList represents a GitLab board list.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/boards.html
type BoardList struct {
	ID       int    `json:"id"`
	Label    *Label `json:"label"`
	Position int    `json:"position"`
}

func (b BoardList) String() string {
	return Stringify(b)
}

// CreateIssueBoardOptions represents the available CreateIssueBoard() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/boards.html#create-a-board-starter
type CreateIssueBoardOptions struct {
	Name *string `url:"name" json:"name"`
}

// CreateIssueBoard creates a new issue board.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/boards.html#create-a-board-starter
func (s *IssueBoardsService) CreateIssueBoard(pid interface{}, opt *CreateIssueBoardOptions, options ...RequestOptionFunc) (*IssueBoard, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/boards", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	board := new(IssueBoard)
	resp, err := s.client.Do(req, board)
	if err != nil {
		return nil, resp, err
	}

	return board, resp, err
}

// UpdateIssueBoardOptions represents the available UpdateIssueBoard() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/boards.html#update-a-board-starter
type UpdateIssueBoardOptions struct {
	Name        *string `url:"name,omitempty" json:"name,omitempty"`
	AssigneeID  *int    `url:"assignee_id,omitempty" json:"assignee_id,omitempty"`
	MilestoneID *int    `url:"milestone_id,omitempty" json:"milestone_id,omitempty"`
	Labels      Labels  `url:"labels,omitempty" json:"labels,omitempty"`
	Weight      *int    `url:"weight,omitempty" json:"weight,omitempty"`
}

// UpdateIssueBoard update an issue board.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/boards.html#create-a-board-starter
func (s *IssueBoardsService) UpdateIssueBoard(pid interface{}, board int, opt *UpdateIssueBoardOptions, options ...RequestOptionFunc) (*IssueBoard, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/boards/%d", pathEscape(project), board)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	is := new(IssueBoard)
	resp, err := s.client.Do(req, is)
	if err != nil {
		return nil, resp, err
	}

	return is, resp, err
}

// DeleteIssueBoard deletes an issue board.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/boards.html#delete-a-board-starter
func (s *IssueBoardsService) DeleteIssueBoard(pid interface{}, board int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/boards/%d", pathEscape(project), board)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ListIssueBoardsOptions represents the available ListIssueBoards() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/boards.html#project-board
type ListIssueBoardsOptions ListOptions

// ListIssueBoards gets a list of all issue boards in a project.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/boards.html#project-board
func (s *IssueBoardsService) ListIssueBoards(pid interface{}, opt *ListIssueBoardsOptions, options ...RequestOptionFunc) ([]*IssueBoard, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/boards", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var is []*IssueBoard
	resp, err := s.client.Do(req, &is)
	if err != nil {
		return nil, resp, err
	}

	return is, resp, err
}

// GetIssueBoard gets a single issue board of a project.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/boards.html#single-board
func (s *IssueBoardsService) GetIssueBoard(pid interface{}, board int, options ...RequestOptionFunc) (*IssueBoard, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/boards/%d", pathEscape(project), board)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	ib := new(IssueBoard)
	resp, err := s.client.Do(req, ib)
	if err != nil {
		return nil, resp, err
	}

	return ib, resp, err
}

// GetIssueBoardListsOptions represents the available GetIssueBoardLists() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/boards.html#list-board-lists
type GetIssueBoardListsOptions ListOptions

// GetIssueBoardLists gets a list of the issue board's lists. Does not include
// backlog and closed lists.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/boards.html#list-board-lists
func (s *IssueBoardsService) GetIssueBoardLists(pid interface{}, board int, opt *GetIssueBoardListsOptions, options ...RequestOptionFunc) ([]*BoardList, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/boards/%d/lists", pathEscape(project), board)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var bl []*BoardList
	resp, err := s.client.Do(req, &bl)
	if err != nil {
		return nil, resp, err
	}

	return bl, resp, err
}

// GetIssueBoardList gets a single issue board list.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/boards.html#single-board-list
func (s *IssueBoardsService) GetIssueBoardList(pid interface{}, board, list int, options ...RequestOptionFunc) (*BoardList, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/boards/%d/lists/%d",
		pathEscape(project),
		board,
		list,
	)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	bl := new(BoardList)
	resp, err := s.client.Do(req, bl)
	if err != nil {
		return nil, resp, err
	}

	return bl, resp, err
}

// CreateIssueBoardListOptions represents the available CreateIssueBoardList()
// options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/boards.html#new-board-list
type CreateIssueBoardListOptions struct {
	LabelID *int `url:"label_id" json:"label_id"`
}

// CreateIssueBoardList creates a new issue board list.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/boards.html#new-board-list
func (s *IssueBoardsService) CreateIssueBoardList(pid interface{}, board int, opt *CreateIssueBoardListOptions, options ...RequestOptionFunc) (*BoardList, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/boards/%d/lists", pathEscape(project), board)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	bl := new(BoardList)
	resp, err := s.client.Do(req, bl)
	if err != nil {
		return nil, resp, err
	}

	return bl, resp, err
}

// UpdateIssueBoardListOptions represents the available UpdateIssueBoardList()
// options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/boards.html#edit-board-list
type UpdateIssueBoardListOptions struct {
	Position *int `url:"position" json:"position"`
}

// UpdateIssueBoardList updates the position of an existing issue board list.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/boards.html#edit-board-list
func (s *IssueBoardsService) UpdateIssueBoardList(pid interface{}, board, list int, opt *UpdateIssueBoardListOptions, options ...RequestOptionFunc) (*BoardList, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/boards/%d/lists/%d",
		pathEscape(project),
		board,
		list,
	)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	bl := new(BoardList)
	resp, err := s.client.Do(req, bl)
	if err != nil {
		return nil, resp, err
	}

	return bl, resp, err
}

// DeleteIssueBoardList soft deletes an issue board list. Only for admins and
// project owners.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/boards.html#delete-a-board-list
func (s *IssueBoardsService) DeleteIssueBoardList(pid interface{}, board, list int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/boards/%d/lists/%d",
		pathEscape(project),
		board,
		list,
	)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
