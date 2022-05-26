//
// Copyright 2021, Eric Stevens
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

// GroupHook represents a GitLab group hook.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/groups.html#list-group-hooks
type GroupHook struct {
	ID                       int        `json:"id"`
	URL                      string     `json:"url"`
	GroupID                  int        `json:"group_id"`
	PushEvents               bool       `json:"push_events"`
	IssuesEvents             bool       `json:"issues_events"`
	ConfidentialIssuesEvents bool       `json:"confidential_issues_events"`
	ConfidentialNoteEvents   bool       `json:"confidential_note_events"`
	MergeRequestsEvents      bool       `json:"merge_requests_events"`
	TagPushEvents            bool       `json:"tag_push_events"`
	NoteEvents               bool       `json:"note_events"`
	JobEvents                bool       `json:"job_events"`
	PipelineEvents           bool       `json:"pipeline_events"`
	WikiPageEvents           bool       `json:"wiki_page_events"`
	DeploymentEvents         bool       `json:"deployment_events"`
	EnableSSLVerification    bool       `json:"enable_ssl_verification"`
	CreatedAt                *time.Time `json:"created_at"`
}

// ListGroupHooks gets a list of group hooks.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/groups.html#list-group-hooks
func (s *GroupsService) ListGroupHooks(gid interface{}) ([]*GroupHook, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/hooks", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, nil, nil)
	if err != nil {
		return nil, nil, err
	}
	var gh []*GroupHook
	resp, err := s.client.Do(req, &gh)
	if err != nil {
		return nil, resp, err
	}

	return gh, resp, err
}

// GetGroupHook gets a specific hook for a group.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/groups.html#get-group-hook
func (s *GroupsService) GetGroupHook(pid interface{}, hook int, options ...RequestOptionFunc) (*GroupHook, *Response, error) {
	group, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/hooks/%d", pathEscape(group), hook)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	gh := new(GroupHook)
	resp, err := s.client.Do(req, gh)
	if err != nil {
		return nil, resp, err
	}

	return gh, resp, err
}

// AddGroupHookOptions represents the available AddGroupHook() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/groups.html#add-group-hook
type AddGroupHookOptions struct {
	URL                      *string `url:"url,omitempty" json:"url,omitempty"`
	PushEvents               *bool   `url:"push_events,omitempty"  json:"push_events,omitempty"`
	IssuesEvents             *bool   `url:"issues_events,omitempty"  json:"issues_events,omitempty"`
	ConfidentialIssuesEvents *bool   `url:"confidential_issues_events,omitempty"  json:"confidential_issues_events,omitempty"`
	ConfidentialNoteEvents   *bool   `url:"confidential_note_events,omitempty"  json:"confidential_note_events,omitempty"`
	MergeRequestsEvents      *bool   `url:"merge_requests_events,omitempty"  json:"merge_requests_events,omitempty"`
	TagPushEvents            *bool   `url:"tag_push_events,omitempty"  json:"tag_push_events,omitempty"`
	NoteEvents               *bool   `url:"note_events,omitempty"  json:"note_events,omitempty"`
	JobEvents                *bool   `url:"job_events,omitempty"  json:"job_events,omitempty"`
	PipelineEvents           *bool   `url:"pipeline_events,omitempty"  json:"pipeline_events,omitempty"`
	WikiPageEvents           *bool   `url:"wiki_page_events,omitempty"  json:"wiki_page_events,omitempty"`
	DeploymentEvents         *bool   `url:"deployment_events,omitempty" json:"deployment_events,omitempty"`
	EnableSSLVerification    *bool   `url:"enable_ssl_verification,omitempty"  json:"enable_ssl_verification,omitempty"`
	Token                    *string `url:"token,omitempty" json:"token,omitempty"`
}

// AddGroupHook create a new group scoped webhook.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/groups.html#add-group-hook
func (s *GroupsService) AddGroupHook(gid interface{}, opt *AddGroupHookOptions, options ...RequestOptionFunc) (*GroupHook, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/hooks", pathEscape(group))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	gh := new(GroupHook)
	resp, err := s.client.Do(req, gh)
	if err != nil {
		return nil, resp, err
	}

	return gh, resp, err
}

// EditGroupHookOptions represents the available EditGroupHook() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/groups.html#edit-group-hook
type EditGroupHookOptions struct {
	URL                      *string `url:"url,omitempty" json:"url,omitempty"`
	PushEvents               *bool   `url:"push_events,omitempty" json:"push_events,omitempty"`
	IssuesEvents             *bool   `url:"issues_events,omitempty" json:"issues_events,omitempty"`
	ConfidentialIssuesEvents *bool   `url:"confidential_issues_events,omitempty" json:"confidential_issues_events,omitempty"`
	ConfidentialNoteEvents   *bool   `url:"confidential_note_events,omitempty" json:"confidential_note_events,omitempty"`
	MergeRequestsEvents      *bool   `url:"merge_requests_events,omitempty" json:"merge_requests_events,omitempty"`
	TagPushEvents            *bool   `url:"tag_push_events,omitempty" json:"tag_push_events,omitempty"`
	NoteEvents               *bool   `url:"note_events,omitempty" json:"note_events,omitempty"`
	JobEvents                *bool   `url:"job_events,omitempty" json:"job_events,omitempty"`
	PipelineEvents           *bool   `url:"pipeline_events,omitempty" json:"pipeline_events,omitempty"`
	WikiPageEvents           *bool   `url:"wiki_page_events,omitempty" json:"wiki_page_events,omitempty"`
	DeploymentEvents         *bool   `url:"deployment_events,omitempty" json:"deployment_events,omitempty"`
	EnableSSLVerification    *bool   `url:"enable_ssl_verification,omitempty" json:"enable_ssl_verification,omitempty"`
	Token                    *string `url:"token,omitempty" json:"token,omitempty"`
}

// EditGroupHook edits a hook for a specified group.
//
// Gitlab API docs:
// https://docs.gitlab.com/ce/api/groups.html#edit-group-hook
func (s *GroupsService) EditGroupHook(pid interface{}, hook int, opt *EditGroupHookOptions, options ...RequestOptionFunc) (*GroupHook, *Response, error) {
	group, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/hooks/%d", pathEscape(group), hook)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	gh := new(GroupHook)
	resp, err := s.client.Do(req, gh)
	if err != nil {
		return nil, resp, err
	}

	return gh, resp, err
}

// DeleteGroupHook removes a hook from a group. This is an idempotent
// method and can be called multiple times.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/groups.html#delete-group-hook
func (s *GroupsService) DeleteGroupHook(pid interface{}, hook int, options ...RequestOptionFunc) (*Response, error) {
	group, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("groups/%s/hooks/%d", pathEscape(group), hook)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
