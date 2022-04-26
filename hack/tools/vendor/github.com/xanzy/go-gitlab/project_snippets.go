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
	"bytes"
	"fmt"
)

// ProjectSnippetsService handles communication with the project snippets
// related methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/project_snippets.html
type ProjectSnippetsService struct {
	client *Client
}

// ListProjectSnippetsOptions represents the available ListSnippets() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/project_snippets.html#list-snippets
type ListProjectSnippetsOptions ListOptions

// ListSnippets gets a list of project snippets.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/project_snippets.html#list-snippets
func (s *ProjectSnippetsService) ListSnippets(pid interface{}, opt *ListProjectSnippetsOptions, options ...RequestOptionFunc) ([]*Snippet, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/snippets", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var ps []*Snippet
	resp, err := s.client.Do(req, &ps)
	if err != nil {
		return nil, resp, err
	}

	return ps, resp, err
}

// GetSnippet gets a single project snippet
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/project_snippets.html#single-snippet
func (s *ProjectSnippetsService) GetSnippet(pid interface{}, snippet int, options ...RequestOptionFunc) (*Snippet, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/snippets/%d", pathEscape(project), snippet)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	ps := new(Snippet)
	resp, err := s.client.Do(req, ps)
	if err != nil {
		return nil, resp, err
	}

	return ps, resp, err
}

// CreateProjectSnippetOptions represents the available CreateSnippet() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/project_snippets.html#create-new-snippet
type CreateProjectSnippetOptions struct {
	Title       *string          `url:"title,omitempty" json:"title,omitempty"`
	FileName    *string          `url:"file_name,omitempty" json:"file_name,omitempty"`
	Description *string          `url:"description,omitempty" json:"description,omitempty"`
	Content     *string          `url:"content,omitempty" json:"content,omitempty"`
	Visibility  *VisibilityValue `url:"visibility,omitempty" json:"visibility,omitempty"`
}

// CreateSnippet creates a new project snippet. The user must have permission
// to create new snippets.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/project_snippets.html#create-new-snippet
func (s *ProjectSnippetsService) CreateSnippet(pid interface{}, opt *CreateProjectSnippetOptions, options ...RequestOptionFunc) (*Snippet, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/snippets", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	ps := new(Snippet)
	resp, err := s.client.Do(req, ps)
	if err != nil {
		return nil, resp, err
	}

	return ps, resp, err
}

// UpdateProjectSnippetOptions represents the available UpdateSnippet() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/project_snippets.html#update-snippet
type UpdateProjectSnippetOptions struct {
	Title       *string          `url:"title,omitempty" json:"title,omitempty"`
	FileName    *string          `url:"file_name,omitempty" json:"file_name,omitempty"`
	Description *string          `url:"description,omitempty" json:"description,omitempty"`
	Content     *string          `url:"content,omitempty" json:"content,omitempty"`
	Visibility  *VisibilityValue `url:"visibility,omitempty" json:"visibility,omitempty"`
}

// UpdateSnippet updates an existing project snippet. The user must have
// permission to change an existing snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/project_snippets.html#update-snippet
func (s *ProjectSnippetsService) UpdateSnippet(pid interface{}, snippet int, opt *UpdateProjectSnippetOptions, options ...RequestOptionFunc) (*Snippet, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/snippets/%d", pathEscape(project), snippet)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	ps := new(Snippet)
	resp, err := s.client.Do(req, ps)
	if err != nil {
		return nil, resp, err
	}

	return ps, resp, err
}

// DeleteSnippet deletes an existing project snippet. This is an idempotent
// function and deleting a non-existent snippet still returns a 200 OK status
// code.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/project_snippets.html#delete-snippet
func (s *ProjectSnippetsService) DeleteSnippet(pid interface{}, snippet int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/snippets/%d", pathEscape(project), snippet)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// SnippetContent returns the raw project snippet as plain text.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/project_snippets.html#snippet-content
func (s *ProjectSnippetsService) SnippetContent(pid interface{}, snippet int, options ...RequestOptionFunc) ([]byte, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/snippets/%d/raw", pathEscape(project), snippet)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var b bytes.Buffer
	resp, err := s.client.Do(req, &b)
	if err != nil {
		return nil, resp, err
	}

	return b.Bytes(), resp, err
}
