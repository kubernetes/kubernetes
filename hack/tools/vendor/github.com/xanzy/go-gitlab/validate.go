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

import "fmt"

// ValidateService handles communication with the validation related methods of
// the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/lint.html
type ValidateService struct {
	client *Client
}

// LintResult represents the linting results.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/lint.html
type LintResult struct {
	Status string   `json:"status"`
	Errors []string `json:"errors"`
}

// ProjectLintResult represents the linting results by project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/lint.html#validate-a-projects-ci-configuration
type ProjectLintResult struct {
	Valid      bool     `json:"valid"`
	Errors     []string `json:"errors"`
	Warnings   []string `json:"warnings"`
	MergedYaml string   `json:"merged_yaml"`
}

// Lint validates .gitlab-ci.yml content.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/lint.html
func (s *ValidateService) Lint(content string, options ...RequestOptionFunc) (*LintResult, *Response, error) {
	var opts struct {
		Content string `url:"content,omitempty" json:"content,omitempty"`
	}
	opts.Content = content

	req, err := s.client.NewRequest("POST", "ci/lint", &opts, options)
	if err != nil {
		return nil, nil, err
	}

	l := new(LintResult)
	resp, err := s.client.Do(req, l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, nil
}

// ProjectNamespaceLintOptions represents the available ProjectNamespaceLint() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/lint.html#validate-a-ci-yaml-configuration-with-a-namespace
type ProjectNamespaceLintOptions struct {
	Content *string `url:"content,omitempty" json:"content,omitempty"`
	DryRun  *bool   `url:"dry_run,omitempty" json:"dry_run,omitempty"`
}

// ProjectNamespaceLint validates .gitlab-ci.yml content by project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/lint.html#validate-a-ci-yaml-configuration-with-a-namespace
func (s *ValidateService) ProjectNamespaceLint(pid interface{}, opt *ProjectNamespaceLintOptions, options ...RequestOptionFunc) (*ProjectLintResult, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/ci/lint", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, &opt, options)
	if err != nil {
		return nil, nil, err
	}

	l := new(ProjectLintResult)
	resp, err := s.client.Do(req, l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, nil
}

// ProjectLintOptions represents the available ProjectLint() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/lint.html#validate-a-projects-ci-configuration
type ProjectLintOptions struct {
	DryRun *bool `url:"dry_run,omitempty" json:"dry_run,omitempty"`
}

// ProjectLint validates .gitlab-ci.yml content by project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/lint.html#validate-a-projects-ci-configuration
func (s *ValidateService) ProjectLint(pid interface{}, opt *ProjectLintOptions, options ...RequestOptionFunc) (*ProjectLintResult, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/ci/lint", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, &opt, options)
	if err != nil {
		return nil, nil, err
	}

	l := new(ProjectLintResult)
	resp, err := s.client.Do(req, l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, nil
}
