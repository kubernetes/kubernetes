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
	"bytes"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"time"
)

// IssuesService handles communication with the issue related methods
// of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/issues.html
type IssuesService struct {
	client    *Client
	timeStats *timeStatsService
}

// IssueAuthor represents a author of the issue.
type IssueAuthor struct {
	ID        int    `json:"id"`
	State     string `json:"state"`
	WebURL    string `json:"web_url"`
	Name      string `json:"name"`
	AvatarURL string `json:"avatar_url"`
	Username  string `json:"username"`
}

// IssueAssignee represents a assignee of the issue.
type IssueAssignee struct {
	ID        int    `json:"id"`
	State     string `json:"state"`
	WebURL    string `json:"web_url"`
	Name      string `json:"name"`
	AvatarURL string `json:"avatar_url"`
	Username  string `json:"username"`
}

// IssueReferences represents references of the issue.
type IssueReferences struct {
	Short    string `json:"short"`
	Relative string `json:"relative"`
	Full     string `json:"full"`
}

// IssueCloser represents a closer of the issue.
type IssueCloser struct {
	ID        int    `json:"id"`
	State     string `json:"state"`
	WebURL    string `json:"web_url"`
	Name      string `json:"name"`
	AvatarURL string `json:"avatar_url"`
	Username  string `json:"username"`
}

// IssueLinks represents links of the issue.
type IssueLinks struct {
	Self       string `json:"self"`
	Notes      string `json:"notes"`
	AwardEmoji string `json:"award_emoji"`
	Project    string `json:"project"`
}

// Issue represents a GitLab issue.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/issues.html
type Issue struct {
	ID                   int              `json:"id"`
	IID                  int              `json:"iid"`
	State                string           `json:"state"`
	Description          string           `json:"description"`
	Author               *IssueAuthor     `json:"author"`
	Milestone            *Milestone       `json:"milestone"`
	ProjectID            int              `json:"project_id"`
	Assignees            []*IssueAssignee `json:"assignees"`
	Assignee             *IssueAssignee   `json:"assignee"`
	UpdatedAt            *time.Time       `json:"updated_at"`
	ClosedAt             *time.Time       `json:"closed_at"`
	ClosedBy             *IssueCloser     `json:"closed_by"`
	Title                string           `json:"title"`
	CreatedAt            *time.Time       `json:"created_at"`
	Labels               Labels           `json:"labels"`
	LabelDetails         []*LabelDetails  `json:"label_details"`
	Upvotes              int              `json:"upvotes"`
	Downvotes            int              `json:"downvotes"`
	DueDate              *ISOTime         `json:"due_date"`
	WebURL               string           `json:"web_url"`
	References           *IssueReferences `json:"references"`
	TimeStats            *TimeStats       `json:"time_stats"`
	Confidential         bool             `json:"confidential"`
	Weight               int              `json:"weight"`
	DiscussionLocked     bool             `json:"discussion_locked"`
	Subscribed           bool             `json:"subscribed"`
	UserNotesCount       int              `json:"user_notes_count"`
	Links                *IssueLinks      `json:"_links"`
	IssueLinkID          int              `json:"issue_link_id"`
	MergeRequestCount    int              `json:"merge_requests_count"`
	EpicIssueID          int              `json:"epic_issue_id"`
	Epic                 *Epic            `json:"epic"`
	TaskCompletionStatus struct {
		Count          int `json:"count"`
		CompletedCount int `json:"completed_count"`
	} `json:"task_completion_status"`
}

func (i Issue) String() string {
	return Stringify(i)
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (i *Issue) UnmarshalJSON(data []byte) error {
	type alias Issue

	raw := make(map[string]interface{})
	err := json.Unmarshal(data, &raw)
	if err != nil {
		return err
	}

	labelDetails, ok := raw["labels"].([]interface{})
	if ok && len(labelDetails) > 0 {
		// We only want to change anything if we got label details.
		if _, ok := labelDetails[0].(map[string]interface{}); !ok {
			return json.Unmarshal(data, (*alias)(i))
		}

		labels := make([]interface{}, len(labelDetails))
		for i, details := range labelDetails {
			labels[i] = details.(map[string]interface{})["name"]
		}

		// Set the correct values
		raw["labels"] = labels
		raw["label_details"] = labelDetails

		data, err = json.Marshal(raw)
		if err != nil {
			return err
		}
	}

	return json.Unmarshal(data, (*alias)(i))
}

// Labels is a custom type with specific marshaling characteristics.
type Labels []string

// MarshalJSON implements the json.Marshaler interface.
func (l *Labels) MarshalJSON() ([]byte, error) {
	if *l == nil {
		return []byte(`null`), nil
	}
	return json.Marshal(strings.Join(*l, ","))
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (l *Labels) UnmarshalJSON(data []byte) error {
	type alias Labels
	if !bytes.HasPrefix(data, []byte("[")) {
		data = []byte(fmt.Sprintf("[%s]", string(data)))
	}
	return json.Unmarshal(data, (*alias)(l))
}

// EncodeValues implements the query.EncodeValues interface
func (l *Labels) EncodeValues(key string, v *url.Values) error {
	v.Set(key, strings.Join(*l, ","))
	return nil
}

// LabelDetails represents detailed label information.
type LabelDetails struct {
	ID              int    `json:"id"`
	Name            string `json:"name"`
	Color           string `json:"color"`
	Description     string `json:"description"`
	DescriptionHTML string `json:"description_html"`
	TextColor       string `json:"text_color"`
}

// ListIssuesOptions represents the available ListIssues() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/issues.html#list-issues
type ListIssuesOptions struct {
	ListOptions
	State              *string    `url:"state,omitempty" json:"state,omitempty"`
	Labels             Labels     `url:"labels,comma,omitempty" json:"labels,omitempty"`
	NotLabels          Labels     `url:"not[labels],comma,omitempty" json:"not[labels],omitempty"`
	WithLabelDetails   *bool      `url:"with_labels_details,omitempty" json:"with_labels_details,omitempty"`
	Milestone          *string    `url:"milestone,omitempty" json:"milestone,omitempty"`
	NotMilestone       *string    `url:"not[milestone],omitempty" json:"not[milestone],omitempty"`
	Scope              *string    `url:"scope,omitempty" json:"scope,omitempty"`
	AuthorID           *int       `url:"author_id,omitempty" json:"author_id,omitempty"`
	NotAuthorID        []int      `url:"not[author_id],omitempty" json:"not[author_id],omitempty"`
	AssigneeID         *int       `url:"assignee_id,omitempty" json:"assignee_id,omitempty"`
	NotAssigneeID      []int      `url:"not[assignee_id],omitempty" json:"not[assignee_id],omitempty"`
	AssigneeUsername   *string    `url:"assignee_username,omitempty" json:"assignee_username,omitempty"`
	MyReactionEmoji    *string    `url:"my_reaction_emoji,omitempty" json:"my_reaction_emoji,omitempty"`
	NotMyReactionEmoji []string   `url:"not[my_reaction_emoji],omitempty" json:"not[my_reaction_emoji],omitempty"`
	IIDs               []int      `url:"iids[],omitempty" json:"iids,omitempty"`
	In                 *string    `url:"in,omitempty" json:"in,omitempty"`
	OrderBy            *string    `url:"order_by,omitempty" json:"order_by,omitempty"`
	Sort               *string    `url:"sort,omitempty" json:"sort,omitempty"`
	Search             *string    `url:"search,omitempty" json:"search,omitempty"`
	CreatedAfter       *time.Time `url:"created_after,omitempty" json:"created_after,omitempty"`
	CreatedBefore      *time.Time `url:"created_before,omitempty" json:"created_before,omitempty"`
	UpdatedAfter       *time.Time `url:"updated_after,omitempty" json:"updated_after,omitempty"`
	UpdatedBefore      *time.Time `url:"updated_before,omitempty" json:"updated_before,omitempty"`
	Confidential       *bool      `url:"confidential,omitempty" json:"confidential,omitempty"`
}

// ListIssues gets all issues created by authenticated user. This function
// takes pagination parameters page and per_page to restrict the list of issues.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/issues.html#list-issues
func (s *IssuesService) ListIssues(opt *ListIssuesOptions, options ...RequestOptionFunc) ([]*Issue, *Response, error) {
	req, err := s.client.NewRequest("GET", "issues", opt, options)
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

// ListGroupIssuesOptions represents the available ListGroupIssues() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/issues.html#list-group-issues
type ListGroupIssuesOptions struct {
	ListOptions
	State              *string    `url:"state,omitempty" json:"state,omitempty"`
	Labels             Labels     `url:"labels,comma,omitempty" json:"labels,omitempty"`
	NotLabels          Labels     `url:"not[labels],comma,omitempty" json:"not[labels],omitempty"`
	WithLabelDetails   *bool      `url:"with_labels_details,omitempty" json:"with_labels_details,omitempty"`
	IIDs               []int      `url:"iids[],omitempty" json:"iids,omitempty"`
	Milestone          *string    `url:"milestone,omitempty" json:"milestone,omitempty"`
	NotMilestone       *string    `url:"not[milestone],omitempty" json:"not[milestone],omitempty"`
	Scope              *string    `url:"scope,omitempty" json:"scope,omitempty"`
	AuthorID           *int       `url:"author_id,omitempty" json:"author_id,omitempty"`
	NotAuthorID        []int      `url:"not[author_id],omitempty" json:"not[author_id],omitempty"`
	AuthorUsername     *string    `url:"author_username,omitempty" json:"author_username,omitempty"`
	AssigneeID         *int       `url:"assignee_id,omitempty" json:"assignee_id,omitempty"`
	NotAssigneeID      []int      `url:"not[assignee_id],omitempty" json:"not[assignee_id],omitempty"`
	AssigneeUsername   *string    `url:"assignee_username,omitempty" json:"assignee_username,omitempty"`
	MyReactionEmoji    *string    `url:"my_reaction_emoji,omitempty" json:"my_reaction_emoji,omitempty"`
	NotMyReactionEmoji []string   `url:"not[my_reaction_emoji],omitempty" json:"not[my_reaction_emoji],omitempty"`
	OrderBy            *string    `url:"order_by,omitempty" json:"order_by,omitempty"`
	Sort               *string    `url:"sort,omitempty" json:"sort,omitempty"`
	Search             *string    `url:"search,omitempty" json:"search,omitempty"`
	In                 *string    `url:"in,omitempty" json:"in,omitempty"`
	CreatedAfter       *time.Time `url:"created_after,omitempty" json:"created_after,omitempty"`
	CreatedBefore      *time.Time `url:"created_before,omitempty" json:"created_before,omitempty"`
	UpdatedAfter       *time.Time `url:"updated_after,omitempty" json:"updated_after,omitempty"`
	UpdatedBefore      *time.Time `url:"updated_before,omitempty" json:"updated_before,omitempty"`
}

// ListGroupIssues gets a list of group issues. This function accepts
// pagination parameters page and per_page to return the list of group issues.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/issues.html#list-group-issues
func (s *IssuesService) ListGroupIssues(pid interface{}, opt *ListGroupIssuesOptions, options ...RequestOptionFunc) ([]*Issue, *Response, error) {
	group, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/issues", pathEscape(group))

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

// ListProjectIssuesOptions represents the available ListProjectIssues() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/issues.html#list-project-issues
type ListProjectIssuesOptions struct {
	ListOptions
	IIDs               []int      `url:"iids[],omitempty" json:"iids,omitempty"`
	State              *string    `url:"state,omitempty" json:"state,omitempty"`
	Labels             Labels     `url:"labels,comma,omitempty" json:"labels,omitempty"`
	NotLabels          Labels     `url:"not[labels],comma,omitempty" json:"not[labels],omitempty"`
	WithLabelDetails   *bool      `url:"with_labels_details,omitempty" json:"with_labels_details,omitempty"`
	Milestone          *string    `url:"milestone,omitempty" json:"milestone,omitempty"`
	NotMilestone       []string   `url:"not[milestone],omitempty" json:"not[milestone],omitempty"`
	Scope              *string    `url:"scope,omitempty" json:"scope,omitempty"`
	AuthorID           *int       `url:"author_id,omitempty" json:"author_id,omitempty"`
	NotAuthorID        []int      `url:"not[author_id],omitempty" json:"not[author_id],omitempty"`
	AssigneeID         *int       `url:"assignee_id,omitempty" json:"assignee_id,omitempty"`
	NotAssigneeID      []int      `url:"not[assignee_id],omitempty" json:"not[assignee_id],omitempty"`
	AssigneeUsername   *string    `url:"assignee_username,omitempty" json:"assignee_username,omitempty"`
	MyReactionEmoji    *string    `url:"my_reaction_emoji,omitempty" json:"my_reaction_emoji,omitempty"`
	NotMyReactionEmoji []string   `url:"not[my_reaction_emoji],omitempty" json:"not[my_reaction_emoji],omitempty"`
	OrderBy            *string    `url:"order_by,omitempty" json:"order_by,omitempty"`
	Sort               *string    `url:"sort,omitempty" json:"sort,omitempty"`
	Search             *string    `url:"search,omitempty" json:"search,omitempty"`
	In                 *string    `url:"in,omitempty" json:"in,omitempty"`
	CreatedAfter       *time.Time `url:"created_after,omitempty" json:"created_after,omitempty"`
	CreatedBefore      *time.Time `url:"created_before,omitempty" json:"created_before,omitempty"`
	UpdatedAfter       *time.Time `url:"updated_after,omitempty" json:"updated_after,omitempty"`
	UpdatedBefore      *time.Time `url:"updated_before,omitempty" json:"updated_before,omitempty"`
	Confidential       *bool      `url:"confidential,omitempty" json:"confidential,omitempty"`
}

// ListProjectIssues gets a list of project issues. This function accepts
// pagination parameters page and per_page to return the list of project issues.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/issues.html#list-project-issues
func (s *IssuesService) ListProjectIssues(pid interface{}, opt *ListProjectIssuesOptions, options ...RequestOptionFunc) ([]*Issue, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues", pathEscape(project))

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

// GetIssue gets a single project issue.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/issues.html#single-issues
func (s *IssuesService) GetIssue(pid interface{}, issue int, options ...RequestOptionFunc) (*Issue, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d", pathEscape(project), issue)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	i := new(Issue)
	resp, err := s.client.Do(req, i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, err
}

// CreateIssueOptions represents the available CreateIssue() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/issues.html#new-issues
type CreateIssueOptions struct {
	IID                                *int       `url:"iid,omitempty" json:"iid,omitempty"`
	Title                              *string    `url:"title,omitempty" json:"title,omitempty"`
	Description                        *string    `url:"description,omitempty" json:"description,omitempty"`
	Confidential                       *bool      `url:"confidential,omitempty" json:"confidential,omitempty"`
	AssigneeIDs                        []int      `url:"assignee_ids,omitempty" json:"assignee_ids,omitempty"`
	MilestoneID                        *int       `url:"milestone_id,omitempty" json:"milestone_id,omitempty"`
	Labels                             Labels     `url:"labels,comma,omitempty" json:"labels,omitempty"`
	CreatedAt                          *time.Time `url:"created_at,omitempty" json:"created_at,omitempty"`
	DueDate                            *ISOTime   `url:"due_date,omitempty" json:"due_date,omitempty"`
	MergeRequestToResolveDiscussionsOf *int       `url:"merge_request_to_resolve_discussions_of,omitempty" json:"merge_request_to_resolve_discussions_of,omitempty"`
	DiscussionToResolve                *string    `url:"discussion_to_resolve,omitempty" json:"discussion_to_resolve,omitempty"`
	Weight                             *int       `url:"weight,omitempty" json:"weight,omitempty"`
}

// CreateIssue creates a new project issue.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/issues.html#new-issues
func (s *IssuesService) CreateIssue(pid interface{}, opt *CreateIssueOptions, options ...RequestOptionFunc) (*Issue, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	i := new(Issue)
	resp, err := s.client.Do(req, i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, err
}

// UpdateIssueOptions represents the available UpdateIssue() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/issues.html#edit-issue
type UpdateIssueOptions struct {
	Title            *string    `url:"title,omitempty" json:"title,omitempty"`
	Description      *string    `url:"description,omitempty" json:"description,omitempty"`
	Confidential     *bool      `url:"confidential,omitempty" json:"confidential,omitempty"`
	AssigneeIDs      []int      `url:"assignee_ids,omitempty" json:"assignee_ids,omitempty"`
	MilestoneID      *int       `url:"milestone_id,omitempty" json:"milestone_id,omitempty"`
	Labels           Labels     `url:"labels,comma,omitempty" json:"labels,omitempty"`
	AddLabels        Labels     `url:"add_labels,comma,omitempty" json:"add_labels,omitempty"`
	RemoveLabels     Labels     `url:"remove_labels,comma,omitempty" json:"remove_labels,omitempty"`
	StateEvent       *string    `url:"state_event,omitempty" json:"state_event,omitempty"`
	UpdatedAt        *time.Time `url:"updated_at,omitempty" json:"updated_at,omitempty"`
	DueDate          *ISOTime   `url:"due_date,omitempty" json:"due_date,omitempty"`
	Weight           *int       `url:"weight,omitempty" json:"weight,omitempty"`
	DiscussionLocked *bool      `url:"discussion_locked,omitempty" json:"discussion_locked,omitempty"`
}

// UpdateIssue updates an existing project issue. This function is also used
// to mark an issue as closed.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/issues.html#edit-issues
func (s *IssuesService) UpdateIssue(pid interface{}, issue int, opt *UpdateIssueOptions, options ...RequestOptionFunc) (*Issue, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d", pathEscape(project), issue)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	i := new(Issue)
	resp, err := s.client.Do(req, i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, err
}

// DeleteIssue deletes a single project issue.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/issues.html#delete-an-issue
func (s *IssuesService) DeleteIssue(pid interface{}, issue int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d", pathEscape(project), issue)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// MoveIssueOptions represents the available MoveIssue() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/issues.html#move-an-issue
type MoveIssueOptions struct {
	ToProjectID *int `url:"to_project_id,omitempty" json:"to_project_id,omitempty"`
}

// MoveIssue updates an existing project issue. This function is also used
// to mark an issue as closed.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/issues.html#move-an-issue
func (s *IssuesService) MoveIssue(pid interface{}, issue int, opt *MoveIssueOptions, options ...RequestOptionFunc) (*Issue, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d/move", pathEscape(project), issue)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	i := new(Issue)
	resp, err := s.client.Do(req, i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, err
}

// SubscribeToIssue subscribes the authenticated user to the given issue to
// receive notifications. If the user is already subscribed to the issue, the
// status code 304 is returned.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/merge_requests.html#subscribe-to-a-merge-request
func (s *IssuesService) SubscribeToIssue(pid interface{}, issue int, options ...RequestOptionFunc) (*Issue, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d/subscribe", pathEscape(project), issue)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	i := new(Issue)
	resp, err := s.client.Do(req, i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, err
}

// UnsubscribeFromIssue unsubscribes the authenticated user from the given
// issue to not receive notifications from that merge request. If the user
// is not subscribed to the issue, status code 304 is returned.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/merge_requests.html#unsubscribe-from-a-merge-request
func (s *IssuesService) UnsubscribeFromIssue(pid interface{}, issue int, options ...RequestOptionFunc) (*Issue, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d/unsubscribe", pathEscape(project), issue)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	i := new(Issue)
	resp, err := s.client.Do(req, i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, err
}

// ListMergeRequestsClosingIssueOptions represents the available
// ListMergeRequestsClosingIssue() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/issues.html#list-merge-requests-that-will-close-issue-on-merge
type ListMergeRequestsClosingIssueOptions ListOptions

// ListMergeRequestsClosingIssue gets all the merge requests that will close
// issue when merged.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/issues.html#list-merge-requests-that-will-close-issue-on-merge
func (s *IssuesService) ListMergeRequestsClosingIssue(pid interface{}, issue int, opt *ListMergeRequestsClosingIssueOptions, options ...RequestOptionFunc) ([]*MergeRequest, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("/projects/%s/issues/%d/closed_by", pathEscape(project), issue)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var m []*MergeRequest
	resp, err := s.client.Do(req, &m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// ListMergeRequestsRelatedToIssueOptions represents the available
// ListMergeRequestsRelatedToIssue() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/issues.html#list-merge-requests-related-to-issue
type ListMergeRequestsRelatedToIssueOptions ListOptions

// ListMergeRequestsRelatedToIssue gets all the merge requests that are
// related to the issue
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/issues.html#list-merge-requests-related-to-issue
func (s *IssuesService) ListMergeRequestsRelatedToIssue(pid interface{}, issue int, opt *ListMergeRequestsRelatedToIssueOptions, options ...RequestOptionFunc) ([]*MergeRequest, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("/projects/%s/issues/%d/related_merge_requests",
		pathEscape(project),
		issue,
	)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var m []*MergeRequest
	resp, err := s.client.Do(req, &m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// SetTimeEstimate sets the time estimate for a single project issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/issues.html#set-a-time-estimate-for-an-issue
func (s *IssuesService) SetTimeEstimate(pid interface{}, issue int, opt *SetTimeEstimateOptions, options ...RequestOptionFunc) (*TimeStats, *Response, error) {
	return s.timeStats.setTimeEstimate(pid, "issues", issue, opt, options...)
}

// ResetTimeEstimate resets the time estimate for a single project issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/issues.html#reset-the-time-estimate-for-an-issue
func (s *IssuesService) ResetTimeEstimate(pid interface{}, issue int, options ...RequestOptionFunc) (*TimeStats, *Response, error) {
	return s.timeStats.resetTimeEstimate(pid, "issues", issue, options...)
}

// AddSpentTime adds spent time for a single project issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/issues.html#add-spent-time-for-an-issue
func (s *IssuesService) AddSpentTime(pid interface{}, issue int, opt *AddSpentTimeOptions, options ...RequestOptionFunc) (*TimeStats, *Response, error) {
	return s.timeStats.addSpentTime(pid, "issues", issue, opt, options...)
}

// ResetSpentTime resets the spent time for a single project issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/issues.html#reset-spent-time-for-an-issue
func (s *IssuesService) ResetSpentTime(pid interface{}, issue int, options ...RequestOptionFunc) (*TimeStats, *Response, error) {
	return s.timeStats.resetSpentTime(pid, "issues", issue, options...)
}

// GetTimeSpent gets the spent time for a single project issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/issues.html#get-time-tracking-stats
func (s *IssuesService) GetTimeSpent(pid interface{}, issue int, options ...RequestOptionFunc) (*TimeStats, *Response, error) {
	return s.timeStats.getTimeSpent(pid, "issues", issue, options...)
}

// GetParticipants gets a list of issue participants.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/issues.html#participants-on-issues
func (s *IssuesService) GetParticipants(pid interface{}, issue int, options ...RequestOptionFunc) ([]*BasicUser, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d/participants", pathEscape(project), issue)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var bu []*BasicUser
	resp, err := s.client.Do(req, &bu)
	if err != nil {
		return nil, resp, err
	}

	return bu, resp, err
}
