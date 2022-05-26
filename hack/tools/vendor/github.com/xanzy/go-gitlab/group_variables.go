//
// Copyright 2021, Patrick Webster
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
	"net/url"
)

// GroupVariablesService handles communication with the
// group variables related methods of the GitLab API.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_level_variables.html
type GroupVariablesService struct {
	client *Client
}

// GroupVariable represents a GitLab group Variable.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_level_variables.html
type GroupVariable struct {
	Key          string            `json:"key"`
	Value        string            `json:"value"`
	VariableType VariableTypeValue `json:"variable_type"`
	Protected    bool              `json:"protected"`
	Masked       bool              `json:"masked"`
}

func (v GroupVariable) String() string {
	return Stringify(v)
}

// ListGroupVariablesOptions represents the available options for listing variables
// for a group.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_level_variables.html#list-group-variables
type ListGroupVariablesOptions ListOptions

// ListVariables gets a list of all variables for a group.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_level_variables.html#list-group-variables
func (s *GroupVariablesService) ListVariables(gid interface{}, opt *ListGroupVariablesOptions, options ...RequestOptionFunc) ([]*GroupVariable, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/variables", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var vs []*GroupVariable
	resp, err := s.client.Do(req, &vs)
	if err != nil {
		return nil, resp, err
	}

	return vs, resp, err
}

// GetVariable gets a variable.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_level_variables.html#show-variable-details
func (s *GroupVariablesService) GetVariable(gid interface{}, key string, options ...RequestOptionFunc) (*GroupVariable, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/variables/%s", pathEscape(group), url.PathEscape(key))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	v := new(GroupVariable)
	resp, err := s.client.Do(req, v)
	if err != nil {
		return nil, resp, err
	}

	return v, resp, err
}

// CreateGroupVariableOptions represents the available CreateVariable()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_level_variables.html#create-variable
type CreateGroupVariableOptions struct {
	Key          *string            `url:"key,omitempty" json:"key,omitempty"`
	Value        *string            `url:"value,omitempty" json:"value,omitempty"`
	VariableType *VariableTypeValue `url:"variable_type,omitempty" json:"variable_type,omitempty"`
	Protected    *bool              `url:"protected,omitempty" json:"protected,omitempty"`
	Masked       *bool              `url:"masked,omitempty" json:"masked,omitempty"`
}

// CreateVariable creates a new group variable.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_level_variables.html#create-variable
func (s *GroupVariablesService) CreateVariable(gid interface{}, opt *CreateGroupVariableOptions, options ...RequestOptionFunc) (*GroupVariable, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/variables", pathEscape(group))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	v := new(GroupVariable)
	resp, err := s.client.Do(req, v)
	if err != nil {
		return nil, resp, err
	}

	return v, resp, err
}

// UpdateGroupVariableOptions represents the available UpdateVariable()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_level_variables.html#update-variable
type UpdateGroupVariableOptions struct {
	Value        *string            `url:"value,omitempty" json:"value,omitempty"`
	VariableType *VariableTypeValue `url:"variable_type,omitempty" json:"variable_type,omitempty"`
	Protected    *bool              `url:"protected,omitempty" json:"protected,omitempty"`
	Masked       *bool              `url:"masked,omitempty" json:"masked,omitempty"`
}

// UpdateVariable updates the position of an existing
// group issue board list.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_level_variables.html#update-variable
func (s *GroupVariablesService) UpdateVariable(gid interface{}, key string, opt *UpdateGroupVariableOptions, options ...RequestOptionFunc) (*GroupVariable, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/variables/%s", pathEscape(group), url.PathEscape(key))

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	v := new(GroupVariable)
	resp, err := s.client.Do(req, v)
	if err != nil {
		return nil, resp, err
	}

	return v, resp, err
}

// RemoveVariable removes a group's variable.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_level_variables.html#remove-variable
func (s *GroupVariablesService) RemoveVariable(gid interface{}, key string, options ...RequestOptionFunc) (*Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("groups/%s/variables/%s", pathEscape(group), url.PathEscape(key))

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
