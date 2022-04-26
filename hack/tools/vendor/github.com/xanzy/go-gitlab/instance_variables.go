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

// InstanceVariablesService handles communication with the
// instance level CI variables related methods of the GitLab API.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_level_ci_variables.html
type InstanceVariablesService struct {
	client *Client
}

// InstanceVariable represents a GitLab instance level CI Variable.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_level_ci_variables.html
type InstanceVariable struct {
	Key          string            `json:"key"`
	Value        string            `json:"value"`
	VariableType VariableTypeValue `json:"variable_type"`
	Protected    bool              `json:"protected"`
	Masked       bool              `json:"masked"`
}

func (v InstanceVariable) String() string {
	return Stringify(v)
}

// ListInstanceVariablesOptions represents the available options for listing variables
// for an instance.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_level_ci_variables.html#list-all-instance-variables
type ListInstanceVariablesOptions ListOptions

// ListVariables gets a list of all variables for an instance.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_level_ci_variables.html#list-all-instance-variables
func (s *InstanceVariablesService) ListVariables(opt *ListInstanceVariablesOptions, options ...RequestOptionFunc) ([]*InstanceVariable, *Response, error) {
	u := "admin/ci/variables"

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var vs []*InstanceVariable
	resp, err := s.client.Do(req, &vs)
	if err != nil {
		return nil, resp, err
	}

	return vs, resp, err
}

// GetVariable gets a variable.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_level_ci_variables.html#show-instance-variable-details
func (s *InstanceVariablesService) GetVariable(key string, options ...RequestOptionFunc) (*InstanceVariable, *Response, error) {
	u := fmt.Sprintf("admin/ci/variables/%s", url.PathEscape(key))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	v := new(InstanceVariable)
	resp, err := s.client.Do(req, v)
	if err != nil {
		return nil, resp, err
	}

	return v, resp, err
}

// CreateInstanceVariableOptions represents the available CreateVariable()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_level_ci_variables.html#create-instance-variable
type CreateInstanceVariableOptions struct {
	Key          *string            `url:"key,omitempty" json:"key,omitempty"`
	Value        *string            `url:"value,omitempty" json:"value,omitempty"`
	VariableType *VariableTypeValue `url:"variable_type,omitempty" json:"variable_type,omitempty"`
	Protected    *bool              `url:"protected,omitempty" json:"protected,omitempty"`
	Masked       *bool              `url:"masked,omitempty" json:"masked,omitempty"`
}

// CreateVariable creates a new instance level CI variable.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_level_ci_variables.html#create-instance-variable
func (s *InstanceVariablesService) CreateVariable(opt *CreateInstanceVariableOptions, options ...RequestOptionFunc) (*InstanceVariable, *Response, error) {
	u := "admin/ci/variables"

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	v := new(InstanceVariable)
	resp, err := s.client.Do(req, v)
	if err != nil {
		return nil, resp, err
	}

	return v, resp, err
}

// UpdateInstanceVariableOptions represents the available UpdateVariable()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_level_ci_variables.html#update-instance-variable
type UpdateInstanceVariableOptions struct {
	Value        *string            `url:"value,omitempty" json:"value,omitempty"`
	VariableType *VariableTypeValue `url:"variable_type,omitempty" json:"variable_type,omitempty"`
	Protected    *bool              `url:"protected,omitempty" json:"protected,omitempty"`
	Masked       *bool              `url:"masked,omitempty" json:"masked,omitempty"`
}

// UpdateVariable updates the position of an existing
// instance level CI variable.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_level_ci_variables.html#update-instance-variable
func (s *InstanceVariablesService) UpdateVariable(key string, opt *UpdateInstanceVariableOptions, options ...RequestOptionFunc) (*InstanceVariable, *Response, error) {
	u := fmt.Sprintf("admin/ci/variables/%s", url.PathEscape(key))

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	v := new(InstanceVariable)
	resp, err := s.client.Do(req, v)
	if err != nil {
		return nil, resp, err
	}

	return v, resp, err
}

// RemoveVariable removes an instance level CI variable.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/instance_level_ci_variables.html#remove-instance-variable
func (s *InstanceVariablesService) RemoveVariable(key string, options ...RequestOptionFunc) (*Response, error) {
	u := fmt.Sprintf("admin/ci/variables/%s", url.PathEscape(key))

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
