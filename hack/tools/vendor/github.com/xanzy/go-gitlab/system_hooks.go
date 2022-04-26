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
	"time"
)

// SystemHooksService handles communication with the system hooks related
// methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/system_hooks.html
type SystemHooksService struct {
	client *Client
}

// Hook represents a GitLap system hook.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/system_hooks.html
type Hook struct {
	ID        int        `json:"id"`
	URL       string     `json:"url"`
	CreatedAt *time.Time `json:"created_at"`
}

func (h Hook) String() string {
	return Stringify(h)
}

// ListHooks gets a list of system hooks.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/system_hooks.html#list-system-hooks
func (s *SystemHooksService) ListHooks(options ...RequestOptionFunc) ([]*Hook, *Response, error) {
	req, err := s.client.NewRequest("GET", "hooks", nil, options)
	if err != nil {
		return nil, nil, err
	}

	var h []*Hook
	resp, err := s.client.Do(req, &h)
	if err != nil {
		return nil, resp, err
	}

	return h, resp, err
}

// AddHookOptions represents the available AddHook() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/system_hooks.html#add-new-system-hook-hook
type AddHookOptions struct {
	URL                    *string `url:"url,omitempty" json:"url,omitempty"`
	Token                  *string `url:"token,omitempty" json:"token,omitempty"`
	PushEvents             *bool   `url:"push_events,omitempty" json:"push_events,omitempty"`
	TagPushEvents          *bool   `url:"tag_push_events,omitempty" json:"tag_push_events,omitempty"`
	MergeRequestsEvents    *bool   `url:"merge_requests_events,omitempty" json:"merge_requests_events,omitempty"`
	RepositoryUpdateEvents *bool   `url:"repository_update_events,omitempty" json:"repository_update_events,omitempty"`
	EnableSSLVerification  *bool   `url:"enable_ssl_verification,omitempty" json:"enable_ssl_verification,omitempty"`
}

// AddHook adds a new system hook hook.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/system_hooks.html#add-new-system-hook-hook
func (s *SystemHooksService) AddHook(opt *AddHookOptions, options ...RequestOptionFunc) (*Hook, *Response, error) {
	req, err := s.client.NewRequest("POST", "hooks", opt, options)
	if err != nil {
		return nil, nil, err
	}

	h := new(Hook)
	resp, err := s.client.Do(req, h)
	if err != nil {
		return nil, resp, err
	}

	return h, resp, err
}

// HookEvent represents an event trigger by a GitLab system hook.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/system_hooks.html
type HookEvent struct {
	EventName  string `json:"event_name"`
	Name       string `json:"name"`
	Path       string `json:"path"`
	ProjectID  int    `json:"project_id"`
	OwnerName  string `json:"owner_name"`
	OwnerEmail string `json:"owner_email"`
}

func (h HookEvent) String() string {
	return Stringify(h)
}

// TestHook tests a system hook.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/system_hooks.html#test-system-hook
func (s *SystemHooksService) TestHook(hook int, options ...RequestOptionFunc) (*HookEvent, *Response, error) {
	u := fmt.Sprintf("hooks/%d", hook)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	h := new(HookEvent)
	resp, err := s.client.Do(req, h)
	if err != nil {
		return nil, resp, err
	}

	return h, resp, err
}

// DeleteHook deletes a system hook. This is an idempotent API function and
// returns 200 OK even if the hook is not available. If the hook is deleted it
// is also returned as JSON.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/system_hooks.html#delete-system-hook
func (s *SystemHooksService) DeleteHook(hook int, options ...RequestOptionFunc) (*Response, error) {
	u := fmt.Sprintf("hooks/%d", hook)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
