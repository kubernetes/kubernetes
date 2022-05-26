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
	"errors"
	"fmt"
)

// NotificationSettingsService handles communication with the notification settings
// related methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/notification_settings.html
type NotificationSettingsService struct {
	client *Client
}

// NotificationSettings represents the Gitlab notification setting.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notification_settings.html#notification-settings
type NotificationSettings struct {
	Level             NotificationLevelValue `json:"level"`
	NotificationEmail string                 `json:"notification_email"`
	Events            *NotificationEvents    `json:"events"`
}

// NotificationEvents represents the available notification setting events.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notification_settings.html#notification-settings
type NotificationEvents struct {
	CloseIssue           bool `json:"close_issue"`
	CloseMergeRequest    bool `json:"close_merge_request"`
	FailedPipeline       bool `json:"failed_pipeline"`
	MergeMergeRequest    bool `json:"merge_merge_request"`
	NewIssue             bool `json:"new_issue"`
	NewMergeRequest      bool `json:"new_merge_request"`
	NewNote              bool `json:"new_note"`
	ReassignIssue        bool `json:"reassign_issue"`
	ReassignMergeRequest bool `json:"reassign_merge_request"`
	ReopenIssue          bool `json:"reopen_issue"`
	ReopenMergeRequest   bool `json:"reopen_merge_request"`
	SuccessPipeline      bool `json:"success_pipeline"`
}

func (ns NotificationSettings) String() string {
	return Stringify(ns)
}

// GetGlobalSettings returns current notification settings and email address.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notification_settings.html#global-notification-settings
func (s *NotificationSettingsService) GetGlobalSettings(options ...RequestOptionFunc) (*NotificationSettings, *Response, error) {
	u := "notification_settings"

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	ns := new(NotificationSettings)
	resp, err := s.client.Do(req, ns)
	if err != nil {
		return nil, resp, err
	}

	return ns, resp, err
}

// NotificationSettingsOptions represents the available options that can be passed
// to the API when updating the notification settings.
type NotificationSettingsOptions struct {
	Level                *NotificationLevelValue `url:"level,omitempty" json:"level,omitempty"`
	NotificationEmail    *string                 `url:"notification_email,omitempty" json:"notification_email,omitempty"`
	CloseIssue           *bool                   `url:"close_issue,omitempty" json:"close_issue,omitempty"`
	CloseMergeRequest    *bool                   `url:"close_merge_request,omitempty" json:"close_merge_request,omitempty"`
	FailedPipeline       *bool                   `url:"failed_pipeline,omitempty" json:"failed_pipeline,omitempty"`
	MergeMergeRequest    *bool                   `url:"merge_merge_request,omitempty" json:"merge_merge_request,omitempty"`
	NewIssue             *bool                   `url:"new_issue,omitempty" json:"new_issue,omitempty"`
	NewMergeRequest      *bool                   `url:"new_merge_request,omitempty" json:"new_merge_request,omitempty"`
	NewNote              *bool                   `url:"new_note,omitempty" json:"new_note,omitempty"`
	ReassignIssue        *bool                   `url:"reassign_issue,omitempty" json:"reassign_issue,omitempty"`
	ReassignMergeRequest *bool                   `url:"reassign_merge_request,omitempty" json:"reassign_merge_request,omitempty"`
	ReopenIssue          *bool                   `url:"reopen_issue,omitempty" json:"reopen_issue,omitempty"`
	ReopenMergeRequest   *bool                   `url:"reopen_merge_request,omitempty" json:"reopen_merge_request,omitempty"`
	SuccessPipeline      *bool                   `url:"success_pipeline,omitempty" json:"success_pipeline,omitempty"`
}

// UpdateGlobalSettings updates current notification settings and email address.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notification_settings.html#update-global-notification-settings
func (s *NotificationSettingsService) UpdateGlobalSettings(opt *NotificationSettingsOptions, options ...RequestOptionFunc) (*NotificationSettings, *Response, error) {
	if opt.Level != nil && *opt.Level == GlobalNotificationLevel {
		return nil, nil, errors.New(
			"notification level 'global' is not valid for global notification settings")
	}

	u := "notification_settings"

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	ns := new(NotificationSettings)
	resp, err := s.client.Do(req, ns)
	if err != nil {
		return nil, resp, err
	}

	return ns, resp, err
}

// GetSettingsForGroup returns current group notification settings.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notification_settings.html#group-project-level-notification-settings
func (s *NotificationSettingsService) GetSettingsForGroup(gid interface{}, options ...RequestOptionFunc) (*NotificationSettings, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/notification_settings", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	ns := new(NotificationSettings)
	resp, err := s.client.Do(req, ns)
	if err != nil {
		return nil, resp, err
	}

	return ns, resp, err
}

// GetSettingsForProject returns current project notification settings.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notification_settings.html#group-project-level-notification-settings
func (s *NotificationSettingsService) GetSettingsForProject(pid interface{}, options ...RequestOptionFunc) (*NotificationSettings, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/notification_settings", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	ns := new(NotificationSettings)
	resp, err := s.client.Do(req, ns)
	if err != nil {
		return nil, resp, err
	}

	return ns, resp, err
}

// UpdateSettingsForGroup updates current group notification settings.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notification_settings.html#update-group-project-level-notification-settings
func (s *NotificationSettingsService) UpdateSettingsForGroup(gid interface{}, opt *NotificationSettingsOptions, options ...RequestOptionFunc) (*NotificationSettings, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/notification_settings", pathEscape(group))

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	ns := new(NotificationSettings)
	resp, err := s.client.Do(req, ns)
	if err != nil {
		return nil, resp, err
	}

	return ns, resp, err
}

// UpdateSettingsForProject updates current project notification settings.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notification_settings.html#update-group-project-level-notification-settings
func (s *NotificationSettingsService) UpdateSettingsForProject(pid interface{}, opt *NotificationSettingsOptions, options ...RequestOptionFunc) (*NotificationSettings, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/notification_settings", pathEscape(project))

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	ns := new(NotificationSettings)
	resp, err := s.client.Do(req, ns)
	if err != nil {
		return nil, resp, err
	}

	return ns, resp, err
}
