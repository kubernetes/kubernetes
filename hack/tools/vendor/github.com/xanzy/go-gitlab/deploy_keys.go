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

// DeployKeysService handles communication with the keys related methods
// of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/deploy_keys.html
type DeployKeysService struct {
	client *Client
}

// DeployKey represents a GitLab deploy key.
type DeployKey struct {
	ID        int        `json:"id"`
	Title     string     `json:"title"`
	Key       string     `json:"key"`
	CanPush   *bool      `json:"can_push"`
	CreatedAt *time.Time `json:"created_at"`
}

func (k DeployKey) String() string {
	return Stringify(k)
}

// ListAllDeployKeys gets a list of all deploy keys
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/deploy_keys.html#list-all-deploy-keys
func (s *DeployKeysService) ListAllDeployKeys(options ...RequestOptionFunc) ([]*DeployKey, *Response, error) {
	req, err := s.client.NewRequest("GET", "deploy_keys", nil, options)
	if err != nil {
		return nil, nil, err
	}

	var ks []*DeployKey
	resp, err := s.client.Do(req, &ks)
	if err != nil {
		return nil, resp, err
	}

	return ks, resp, err
}

// ListProjectDeployKeysOptions represents the available ListProjectDeployKeys()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/deploy_keys.html#list-project-deploy-keys
type ListProjectDeployKeysOptions ListOptions

// ListProjectDeployKeys gets a list of a project's deploy keys
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/deploy_keys.html#list-project-deploy-keys
func (s *DeployKeysService) ListProjectDeployKeys(pid interface{}, opt *ListProjectDeployKeysOptions, options ...RequestOptionFunc) ([]*DeployKey, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/deploy_keys", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var ks []*DeployKey
	resp, err := s.client.Do(req, &ks)
	if err != nil {
		return nil, resp, err
	}

	return ks, resp, err
}

// GetDeployKey gets a single deploy key.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/deploy_keys.html#single-deploy-key
func (s *DeployKeysService) GetDeployKey(pid interface{}, deployKey int, options ...RequestOptionFunc) (*DeployKey, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/deploy_keys/%d", pathEscape(project), deployKey)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	k := new(DeployKey)
	resp, err := s.client.Do(req, k)
	if err != nil {
		return nil, resp, err
	}

	return k, resp, err
}

// AddDeployKeyOptions represents the available ADDDeployKey() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/deploy_keys.html#add-deploy-key
type AddDeployKeyOptions struct {
	Title   *string `url:"title,omitempty" json:"title,omitempty"`
	Key     *string `url:"key,omitempty" json:"key,omitempty"`
	CanPush *bool   `url:"can_push,omitempty" json:"can_push,omitempty"`
}

// AddDeployKey creates a new deploy key for a project. If deploy key already
// exists in another project - it will be joined to project but only if
// original one was is accessible by same user.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/deploy_keys.html#add-deploy-key
func (s *DeployKeysService) AddDeployKey(pid interface{}, opt *AddDeployKeyOptions, options ...RequestOptionFunc) (*DeployKey, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/deploy_keys", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	k := new(DeployKey)
	resp, err := s.client.Do(req, k)
	if err != nil {
		return nil, resp, err
	}

	return k, resp, err
}

// DeleteDeployKey deletes a deploy key from a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/deploy_keys.html#delete-deploy-key
func (s *DeployKeysService) DeleteDeployKey(pid interface{}, deployKey int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/deploy_keys/%d", pathEscape(project), deployKey)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// EnableDeployKey enables a deploy key.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/deploy_keys.html#enable-deploy-key
func (s *DeployKeysService) EnableDeployKey(pid interface{}, deployKey int, options ...RequestOptionFunc) (*DeployKey, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/deploy_keys/%d/enable", pathEscape(project), deployKey)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	k := new(DeployKey)
	resp, err := s.client.Do(req, k)
	if err != nil {
		return nil, resp, err
	}

	return k, resp, err
}

// UpdateDeployKeyOptions represents the available UpdateDeployKey() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/deploy_keys.html#update-deploy-key
type UpdateDeployKeyOptions struct {
	Title   *string `url:"title,omitempty" json:"title,omitempty"`
	CanPush *bool   `url:"can_push,omitempty" json:"can_push,omitempty"`
}

// UpdateDeployKey updates a deploy key for a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/deploy_keys.html#update-deploy-key
func (s *DeployKeysService) UpdateDeployKey(pid interface{}, deployKey int, opt *UpdateDeployKeyOptions, options ...RequestOptionFunc) (*DeployKey, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/deploy_keys/%d", pathEscape(project), deployKey)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	k := new(DeployKey)
	resp, err := s.client.Do(req, k)
	if err != nil {
		return nil, resp, err
	}

	return k, resp, err
}
