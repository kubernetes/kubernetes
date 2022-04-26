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

// CustomAttributesService handles communication with the group, project and
// user custom attributes related methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/custom_attributes.html
type CustomAttributesService struct {
	client *Client
}

// CustomAttribute struct is used to unmarshal response to api calls.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/custom_attributes.html
type CustomAttribute struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// ListCustomUserAttributes lists the custom attributes of the specified user.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/custom_attributes.html#list-custom-attributes
func (s *CustomAttributesService) ListCustomUserAttributes(user int, options ...RequestOptionFunc) ([]*CustomAttribute, *Response, error) {
	return s.listCustomAttributes("users", user, options...)
}

// ListCustomGroupAttributes lists the custom attributes of the specified group.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/custom_attributes.html#list-custom-attributes
func (s *CustomAttributesService) ListCustomGroupAttributes(group int, options ...RequestOptionFunc) ([]*CustomAttribute, *Response, error) {
	return s.listCustomAttributes("groups", group, options...)
}

// ListCustomProjectAttributes lists the custom attributes of the specified project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/custom_attributes.html#list-custom-attributes
func (s *CustomAttributesService) ListCustomProjectAttributes(project int, options ...RequestOptionFunc) ([]*CustomAttribute, *Response, error) {
	return s.listCustomAttributes("projects", project, options...)
}

func (s *CustomAttributesService) listCustomAttributes(resource string, id int, options ...RequestOptionFunc) ([]*CustomAttribute, *Response, error) {
	u := fmt.Sprintf("%s/%d/custom_attributes", resource, id)
	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var cas []*CustomAttribute
	resp, err := s.client.Do(req, &cas)
	if err != nil {
		return nil, resp, err
	}
	return cas, resp, err
}

// GetCustomUserAttribute returns the user attribute with a speciifc key.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/custom_attributes.html#single-custom-attribute
func (s *CustomAttributesService) GetCustomUserAttribute(user int, key string, options ...RequestOptionFunc) (*CustomAttribute, *Response, error) {
	return s.getCustomAttribute("users", user, key, options...)
}

// GetCustomGroupAttribute returns the group attribute with a speciifc key.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/custom_attributes.html#single-custom-attribute
func (s *CustomAttributesService) GetCustomGroupAttribute(group int, key string, options ...RequestOptionFunc) (*CustomAttribute, *Response, error) {
	return s.getCustomAttribute("groups", group, key, options...)
}

// GetCustomProjectAttribute returns the project attribute with a speciifc key.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/custom_attributes.html#single-custom-attribute
func (s *CustomAttributesService) GetCustomProjectAttribute(project int, key string, options ...RequestOptionFunc) (*CustomAttribute, *Response, error) {
	return s.getCustomAttribute("projects", project, key, options...)
}

func (s *CustomAttributesService) getCustomAttribute(resource string, id int, key string, options ...RequestOptionFunc) (*CustomAttribute, *Response, error) {
	u := fmt.Sprintf("%s/%d/custom_attributes/%s", resource, id, key)
	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var ca *CustomAttribute
	resp, err := s.client.Do(req, &ca)
	if err != nil {
		return nil, resp, err
	}
	return ca, resp, err
}

// SetCustomUserAttribute sets the custom attributes of the specified user.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/custom_attributes.html#set-custom-attribute
func (s *CustomAttributesService) SetCustomUserAttribute(user int, c CustomAttribute, options ...RequestOptionFunc) (*CustomAttribute, *Response, error) {
	return s.setCustomAttribute("users", user, c, options...)
}

// SetCustomGroupAttribute sets the custom attributes of the specified group.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/custom_attributes.html#set-custom-attribute
func (s *CustomAttributesService) SetCustomGroupAttribute(group int, c CustomAttribute, options ...RequestOptionFunc) (*CustomAttribute, *Response, error) {
	return s.setCustomAttribute("groups", group, c, options...)
}

// SetCustomProjectAttribute sets the custom attributes of the specified project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/custom_attributes.html#set-custom-attribute
func (s *CustomAttributesService) SetCustomProjectAttribute(project int, c CustomAttribute, options ...RequestOptionFunc) (*CustomAttribute, *Response, error) {
	return s.setCustomAttribute("projects", project, c, options...)
}

func (s *CustomAttributesService) setCustomAttribute(resource string, id int, c CustomAttribute, options ...RequestOptionFunc) (*CustomAttribute, *Response, error) {
	u := fmt.Sprintf("%s/%d/custom_attributes/%s", resource, id, c.Key)
	req, err := s.client.NewRequest("PUT", u, c, options)
	if err != nil {
		return nil, nil, err
	}

	ca := new(CustomAttribute)
	resp, err := s.client.Do(req, ca)
	if err != nil {
		return nil, resp, err
	}
	return ca, resp, err
}

// DeleteCustomUserAttribute removes the custom attribute of the specified user.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/custom_attributes.html#delete-custom-attribute
func (s *CustomAttributesService) DeleteCustomUserAttribute(user int, key string, options ...RequestOptionFunc) (*Response, error) {
	return s.deleteCustomAttribute("users", user, key, options...)
}

// DeleteCustomGroupAttribute removes the custom attribute of the specified group.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/custom_attributes.html#delete-custom-attribute
func (s *CustomAttributesService) DeleteCustomGroupAttribute(group int, key string, options ...RequestOptionFunc) (*Response, error) {
	return s.deleteCustomAttribute("groups", group, key, options...)
}

// DeleteCustomProjectAttribute removes the custom attribute of the specified project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/custom_attributes.html#delete-custom-attribute
func (s *CustomAttributesService) DeleteCustomProjectAttribute(project int, key string, options ...RequestOptionFunc) (*Response, error) {
	return s.deleteCustomAttribute("projects", project, key, options...)
}

func (s *CustomAttributesService) deleteCustomAttribute(resource string, id int, key string, options ...RequestOptionFunc) (*Response, error) {
	u := fmt.Sprintf("%s/%d/custom_attributes/%s", resource, id, key)
	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}
