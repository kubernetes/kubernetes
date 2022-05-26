//
// Copyright 2021, Arkbriar
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

// IssueLinksService handles communication with the issue relations related methods
// of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/issue_links.html
type IssueLinksService struct {
	client *Client
}

// IssueLink represents a two-way relation between two issues.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/issue_links.html
type IssueLink struct {
	SourceIssue *Issue `json:"source_issue"`
	TargetIssue *Issue `json:"target_issue"`
	LinkType    string `json:"link_type"`
}

// ListIssueRelations gets a list of related issues of a given issue,
// sorted by the relationship creation datetime (ascending).
//
// Issues will be filtered according to the user authorizations.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/issue_links.html#list-issue-relations
func (s *IssueLinksService) ListIssueRelations(pid interface{}, issueIID int, options ...RequestOptionFunc) ([]*Issue, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d/links", pathEscape(project), issueIID)

	req, err := s.client.NewRequest("GET", u, nil, options)
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

// CreateIssueLinkOptions represents the available CreateIssueLink() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/issue_links.html
type CreateIssueLinkOptions struct {
	TargetProjectID *string `json:"target_project_id"`
	TargetIssueIID  *string `json:"target_issue_iid"`
	LinkType        *string `json:"link_type"`
}

// CreateIssueLink creates a two-way relation between two issues.
// User must be allowed to update both issues in order to succeed.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/issue_links.html#create-an-issue-link
func (s *IssueLinksService) CreateIssueLink(pid interface{}, issueIID int, opt *CreateIssueLinkOptions, options ...RequestOptionFunc) (*IssueLink, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d/links", pathEscape(project), issueIID)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	i := new(IssueLink)
	resp, err := s.client.Do(req, &i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, err
}

// DeleteIssueLink deletes an issue link, thus removes the two-way relationship.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/issue_links.html#delete-an-issue-link
func (s *IssueLinksService) DeleteIssueLink(pid interface{}, issueIID, issueLinkID int, options ...RequestOptionFunc) (*IssueLink, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d/links/%d",
		pathEscape(project),
		issueIID,
		issueLinkID)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	i := new(IssueLink)
	resp, err := s.client.Do(req, &i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, err
}
