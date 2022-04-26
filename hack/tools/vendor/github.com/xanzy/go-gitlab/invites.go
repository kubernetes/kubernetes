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

// InvitesService handles communication with the invitation related
// methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/invitations.html
type InvitesService struct {
	client *Client
}

// PendingInvite represents a pending invite.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/invitations.html
type PendingInvite struct {
	ID            int              `json:"id"`
	InviteEmail   string           `json:"invite_email"`
	CreatedAt     *time.Time       `json:"created_at"`
	AccessLevel   AccessLevelValue `json:"access_level"`
	ExpiresAt     *time.Time       `json:"expires_at"`
	UserName      string           `json:"user_name"`
	CreatedByName string           `json:"created_by_name"`
}

// ListPendingInvitationsOptions represents the available
// ListPendingInvitations() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/invitations.html#list-all-invitations-pending-for-a-group-or-project
type ListPendingInvitationsOptions struct {
	ListOptions
	Query *string `url:"query,omitempty" json:"query,omitempty"`
}

// ListPendingGroupInvitations gets a list of invited group members.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/invitations.html#list-all-invitations-pending-for-a-group-or-project
func (s *InvitesService) ListPendingGroupInvitations(gid interface{}, opt *ListPendingInvitationsOptions, options ...RequestOptionFunc) ([]*PendingInvite, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/invitations", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var pis []*PendingInvite
	resp, err := s.client.Do(req, &pis)
	if err != nil {
		return nil, resp, err
	}

	return pis, resp, err
}

// ListPendingProjectInvitations gets a list of invited project members.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/invitations.html#list-all-invitations-pending-for-a-group-or-project
func (s *InvitesService) ListPendingProjectInvitations(pid interface{}, opt *ListPendingInvitationsOptions, options ...RequestOptionFunc) ([]*PendingInvite, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/invitations", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var pis []*PendingInvite
	resp, err := s.client.Do(req, &pis)
	if err != nil {
		return nil, resp, err
	}

	return pis, resp, err
}

// InvitesOptions represents the available GroupInvites() and ProjectInvites()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/invitations.html#invite-by-email-to-group-or-project
type InvitesOptions struct {
	ID          interface{}       `url:"id,omitempty" json:"id,omitempty"`
	Email       *string           `url:"email,omitempty" json:"email,omitempty"`
	AccessLevel *AccessLevelValue `url:"access_level,omitempty" json:"access_level,omitempty"`
	ExpiresAt   *ISOTime          `url:"expires_at,omitempty" json:"expires_at,omitempty"`
}

// InvitesResult represents an invitations result.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/invitations.html#invite-by-email-to-group-or-project
type InvitesResult struct {
	Status  string            `json:"status"`
	Message map[string]string `json:"message,omitempty"`
}

// GroupInvites invites new users by email to join a group.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/invitations.html#invite-by-email-to-group-or-project
func (s *InvitesService) GroupInvites(gid interface{}, opt *InvitesOptions, options ...RequestOptionFunc) (*InvitesResult, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/invitations", pathEscape(group))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	ir := new(InvitesResult)
	resp, err := s.client.Do(req, ir)
	if err != nil {
		return nil, resp, err
	}

	return ir, resp, err
}

// ProjectInvites invites new users by email to join a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/invitations.html#invite-by-email-to-group-or-project
func (s *InvitesService) ProjectInvites(pid interface{}, opt *InvitesOptions, options ...RequestOptionFunc) (*InvitesResult, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/invitations", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	ir := new(InvitesResult)
	resp, err := s.client.Do(req, ir)
	if err != nil {
		return nil, resp, err
	}

	return ir, resp, err
}
