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

// ProjectMembersService handles communication with the project members
// related methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/members.html
type ProjectMembersService struct {
	client *Client
}

// ListProjectMembersOptions represents the available ListProjectMembers() and
// ListAllProjectMembers() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#list-all-members-of-a-group-or-project
type ListProjectMembersOptions struct {
	ListOptions
	Query *string `url:"query,omitempty" json:"query,omitempty"`
}

// ListProjectMembers gets a list of a project's team members viewable by the
// authenticated user. Returns only direct members and not inherited members
// through ancestors groups.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#list-all-members-of-a-group-or-project
func (s *ProjectMembersService) ListProjectMembers(pid interface{}, opt *ListProjectMembersOptions, options ...RequestOptionFunc) ([]*ProjectMember, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/members", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var pm []*ProjectMember
	resp, err := s.client.Do(req, &pm)
	if err != nil {
		return nil, resp, err
	}

	return pm, resp, err
}

// ListAllProjectMembers gets a list of a project's team members viewable by the
// authenticated user. Returns a list including inherited members through
// ancestor groups.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#list-all-members-of-a-group-or-project-including-inherited-members
func (s *ProjectMembersService) ListAllProjectMembers(pid interface{}, opt *ListProjectMembersOptions, options ...RequestOptionFunc) ([]*ProjectMember, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/members/all", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var pm []*ProjectMember
	resp, err := s.client.Do(req, &pm)
	if err != nil {
		return nil, resp, err
	}

	return pm, resp, err
}

// GetProjectMember gets a project team member.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#get-a-member-of-a-group-or-project
func (s *ProjectMembersService) GetProjectMember(pid interface{}, user int, options ...RequestOptionFunc) (*ProjectMember, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/members/%d", pathEscape(project), user)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	pm := new(ProjectMember)
	resp, err := s.client.Do(req, pm)
	if err != nil {
		return nil, resp, err
	}

	return pm, resp, err
}

// GetInheritedProjectMember gets a project team member, including inherited
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#get-a-member-of-a-group-or-project-including-inherited-members
func (s *ProjectMembersService) GetInheritedProjectMember(pid interface{}, user int, options ...RequestOptionFunc) (*ProjectMember, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/members/all/%d", pathEscape(project), user)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	pm := new(ProjectMember)
	resp, err := s.client.Do(req, pm)
	if err != nil {
		return nil, resp, err
	}

	return pm, resp, err
}

// AddProjectMemberOptions represents the available AddProjectMember() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#add-a-member-to-a-group-or-project
type AddProjectMemberOptions struct {
	UserID      interface{}       `url:"user_id,omitempty" json:"user_id,omitempty"`
	AccessLevel *AccessLevelValue `url:"access_level,omitempty" json:"access_level,omitempty"`
	ExpiresAt   *string           `url:"expires_at,omitempty" json:"expires_at"`
}

// AddProjectMember adds a user to a project team. This is an idempotent
// method and can be called multiple times with the same parameters. Adding
// team membership to a user that is already a member does not affect the
// existing membership.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#add-a-member-to-a-group-or-project
func (s *ProjectMembersService) AddProjectMember(pid interface{}, opt *AddProjectMemberOptions, options ...RequestOptionFunc) (*ProjectMember, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/members", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	pm := new(ProjectMember)
	resp, err := s.client.Do(req, pm)
	if err != nil {
		return nil, resp, err
	}

	return pm, resp, err
}

// EditProjectMemberOptions represents the available EditProjectMember() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#edit-a-member-of-a-group-or-project
type EditProjectMemberOptions struct {
	AccessLevel *AccessLevelValue `url:"access_level,omitempty" json:"access_level,omitempty"`
	ExpiresAt   *string           `url:"expires_at,omitempty" json:"expires_at"`
}

// EditProjectMember updates a project team member to a specified access level..
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#edit-a-member-of-a-group-or-project
func (s *ProjectMembersService) EditProjectMember(pid interface{}, user int, opt *EditProjectMemberOptions, options ...RequestOptionFunc) (*ProjectMember, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/members/%d", pathEscape(project), user)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	pm := new(ProjectMember)
	resp, err := s.client.Do(req, pm)
	if err != nil {
		return nil, resp, err
	}

	return pm, resp, err
}

// DeleteProjectMember removes a user from a project team.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/members.html#remove-a-member-from-a-group-or-project
func (s *ProjectMembersService) DeleteProjectMember(pid interface{}, user int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/members/%d", pathEscape(project), user)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
