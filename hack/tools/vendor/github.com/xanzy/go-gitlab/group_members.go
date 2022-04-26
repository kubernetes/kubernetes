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

// GroupMembersService handles communication with the group members
// related methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/members.html
type GroupMembersService struct {
	client *Client
}

// GroupMemberSAMLIdentity represents the SAML Identity link for the group member.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/members.html#list-all-members-of-a-group-or-project
// Gitlab MR for API change: https://gitlab.com/gitlab-org/gitlab/-/merge_requests/20357
// Gitlab MR for API Doc change: https://gitlab.com/gitlab-org/gitlab/-/merge_requests/25652
type GroupMemberSAMLIdentity struct {
	ExternUID      string `json:"extern_uid"`
	Provider       string `json:"provider"`
	SAMLProviderID int    `json:"saml_provider_id"`
}

// GroupMember represents a GitLab group member.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/members.html
type GroupMember struct {
	ID                int                      `json:"id"`
	Username          string                   `json:"username"`
	Name              string                   `json:"name"`
	State             string                   `json:"state"`
	AvatarURL         string                   `json:"avatar_url"`
	WebURL            string                   `json:"web_url"`
	ExpiresAt         *ISOTime                 `json:"expires_at"`
	AccessLevel       AccessLevelValue         `json:"access_level"`
	GroupSAMLIdentity *GroupMemberSAMLIdentity `json:"group_saml_identity"`
}

// ListGroupMembersOptions represents the available ListGroupMembers() and
// ListAllGroupMembers() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#list-all-members-of-a-group-or-project
type ListGroupMembersOptions struct {
	ListOptions
	Query *string `url:"query,omitempty" json:"query,omitempty"`
}

// ListGroupMembers get a list of group members viewable by the authenticated
// user. Inherited members through ancestor groups are not included.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#list-all-members-of-a-group-or-project
func (s *GroupsService) ListGroupMembers(gid interface{}, opt *ListGroupMembersOptions, options ...RequestOptionFunc) ([]*GroupMember, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/members", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var gm []*GroupMember
	resp, err := s.client.Do(req, &gm)
	if err != nil {
		return nil, resp, err
	}

	return gm, resp, err
}

// ListAllGroupMembers get a list of group members viewable by the authenticated
// user. Returns a list including inherited members through ancestor groups.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#list-all-members-of-a-group-or-project-including-inherited-members
func (s *GroupsService) ListAllGroupMembers(gid interface{}, opt *ListGroupMembersOptions, options ...RequestOptionFunc) ([]*GroupMember, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/members/all", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var gm []*GroupMember
	resp, err := s.client.Do(req, &gm)
	if err != nil {
		return nil, resp, err
	}

	return gm, resp, err
}

// AddGroupMemberOptions represents the available AddGroupMember() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#add-a-member-to-a-group-or-project
type AddGroupMemberOptions struct {
	UserID      *int              `url:"user_id,omitempty" json:"user_id,omitempty"`
	AccessLevel *AccessLevelValue `url:"access_level,omitempty" json:"access_level,omitempty"`
	ExpiresAt   *string           `url:"expires_at,omitempty" json:"expires_at"`
}

// GetGroupMember gets a member of a group.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#get-a-member-of-a-group-or-project
func (s *GroupMembersService) GetGroupMember(gid interface{}, user int, options ...RequestOptionFunc) (*GroupMember, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/members/%d", pathEscape(group), user)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	gm := new(GroupMember)
	resp, err := s.client.Do(req, gm)
	if err != nil {
		return nil, resp, err
	}

	return gm, resp, err
}

// AddGroupMember adds a user to the list of group members.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#add-a-member-to-a-group-or-project
func (s *GroupMembersService) AddGroupMember(gid interface{}, opt *AddGroupMemberOptions, options ...RequestOptionFunc) (*GroupMember, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/members", pathEscape(group))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	gm := new(GroupMember)
	resp, err := s.client.Do(req, gm)
	if err != nil {
		return nil, resp, err
	}

	return gm, resp, err
}

// ShareWithGroup shares a group with the group.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/groups.html#share-groups-with-groups
func (s *GroupMembersService) ShareWithGroup(gid interface{}, opt *ShareWithGroupOptions, options ...RequestOptionFunc) (*Group, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/share", pathEscape(group))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	g := new(Group)
	resp, err := s.client.Do(req, g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, err
}

// DeleteShareWithGroup allows to unshare a group from a group.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/groups.html#delete-link-sharing-group-with-another-group
func (s *GroupMembersService) DeleteShareWithGroup(gid interface{}, groupID int, options ...RequestOptionFunc) (*Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("groups/%s/share/%d", pathEscape(group), groupID)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// EditGroupMemberOptions represents the available EditGroupMember()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#edit-a-member-of-a-group-or-project
type EditGroupMemberOptions struct {
	AccessLevel *AccessLevelValue `url:"access_level,omitempty" json:"access_level,omitempty"`
	ExpiresAt   *string           `url:"expires_at,omitempty" json:"expires_at"`
}

// EditGroupMember updates a member of a group.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#edit-a-member-of-a-group-or-project
func (s *GroupMembersService) EditGroupMember(gid interface{}, user int, opt *EditGroupMemberOptions, options ...RequestOptionFunc) (*GroupMember, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/members/%d", pathEscape(group), user)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	gm := new(GroupMember)
	resp, err := s.client.Do(req, gm)
	if err != nil {
		return nil, resp, err
	}

	return gm, resp, err
}

// RemoveGroupMember removes user from user team.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#remove-a-member-from-a-group-or-project
func (s *GroupMembersService) RemoveGroupMember(gid interface{}, user int, options ...RequestOptionFunc) (*Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("groups/%s/members/%d", pathEscape(group), user)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
