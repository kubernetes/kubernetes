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

// CIYMLTemplatesService handles communication with the gitlab
// CI YML templates related methods of the GitLab API.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/templates/gitlab_ci_ymls.html
type CIYMLTemplatesService struct {
	client *Client
}

// CIYMLTemplate represents a GitLab CI YML template.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/templates/gitlab_ci_ymls.html
type CIYMLTemplate struct {
	Name    string `json:"name"`
	Content string `json:"content"`
}

// ListCIYMLTemplatesOptions represents the available ListAllTemplates() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/templates/gitignores.html#list-gitignore-templates
type ListCIYMLTemplatesOptions ListOptions

// ListAllTemplates get all GitLab CI YML templates.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/templates/gitlab_ci_ymls.html#list-gitlab-ci-yml-templates
func (s *CIYMLTemplatesService) ListAllTemplates(opt *ListCIYMLTemplatesOptions, options ...RequestOptionFunc) ([]*CIYMLTemplate, *Response, error) {
	req, err := s.client.NewRequest("GET", "templates/gitlab_ci_ymls", opt, options)
	if err != nil {
		return nil, nil, err
	}

	var cts []*CIYMLTemplate
	resp, err := s.client.Do(req, &cts)
	if err != nil {
		return nil, resp, err
	}

	return cts, resp, err
}

// GetTemplate get a single GitLab CI YML template.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/templates/gitlab_ci_ymls.html#single-gitlab-ci-yml-template
func (s *CIYMLTemplatesService) GetTemplate(key string, options ...RequestOptionFunc) (*CIYMLTemplate, *Response, error) {
	u := fmt.Sprintf("templates/gitlab_ci_ymls/%s", pathEscape(key))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	ct := new(CIYMLTemplate)
	resp, err := s.client.Do(req, ct)
	if err != nil {
		return nil, resp, err
	}

	return ct, resp, err
}
