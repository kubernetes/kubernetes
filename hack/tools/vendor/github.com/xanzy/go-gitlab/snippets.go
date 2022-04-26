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
	"time"
)

// SnippetsService handles communication with the snippets
// related methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/snippets.html
type SnippetsService struct {
	client *Client
}

// Snippet represents a GitLab snippet.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/snippets.html
type Snippet struct {
	ID          int    `json:"id"`
	Title       string `json:"title"`
	FileName    string `json:"file_name"`
	Description string `json:"description"`
	Author      struct {
		ID        int        `json:"id"`
		Username  string     `json:"username"`
		Email     string     `json:"email"`
		Name      string     `json:"name"`
		State     string     `json:"state"`
		CreatedAt *time.Time `json:"created_at"`
	} `json:"author"`
	UpdatedAt *time.Time `json:"updated_at"`
	CreatedAt *time.Time `json:"created_at"`
	WebURL    string     `json:"web_url"`
	RawURL    string     `json:"raw_url"`
}

func (s Snippet) String() string {
	return Stringify(s)
}

// ListSnippetsOptions represents the available ListSnippets() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/snippets.html#list-snippets
type ListSnippetsOptions ListOptions

// ListSnippets gets a list of snippets.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/snippets.html#list-snippets
func (s *SnippetsService) ListSnippets(opt *ListSnippetsOptions, options ...RequestOptionFunc) ([]*Snippet, *Response, error) {
	req, err := s.client.NewRequest("GET", "snippets", opt, options)
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

// GetSnippet gets a single snippet
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/snippets.html#single-snippet
func (s *SnippetsService) GetSnippet(snippet int, options ...RequestOptionFunc) (*Snippet, *Response, error) {
	u := fmt.Sprintf("snippets/%d", snippet)

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

// CreateSnippetOptions represents the available CreateSnippet() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/snippets.html#create-new-snippet
type CreateSnippetOptions struct {
	Title       *string          `url:"title,omitempty" json:"title,omitempty"`
	FileName    *string          `url:"file_name,omitempty" json:"file_name,omitempty"`
	Description *string          `url:"description,omitempty" json:"description,omitempty"`
	Content     *string          `url:"content,omitempty" json:"content,omitempty"`
	Visibility  *VisibilityValue `url:"visibility,omitempty" json:"visibility,omitempty"`
}

// CreateSnippet creates a new snippet. The user must have permission
// to create new snippets.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/snippets.html#create-new-snippet
func (s *SnippetsService) CreateSnippet(opt *CreateSnippetOptions, options ...RequestOptionFunc) (*Snippet, *Response, error) {
	req, err := s.client.NewRequest("POST", "snippets", opt, options)
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

// UpdateSnippetOptions represents the available UpdateSnippet() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/snippets.html#update-snippet
type UpdateSnippetOptions struct {
	Title       *string          `url:"title,omitempty" json:"title,omitempty"`
	FileName    *string          `url:"file_name,omitempty" json:"file_name,omitempty"`
	Description *string          `url:"description,omitempty" json:"description,omitempty"`
	Content     *string          `url:"content,omitempty" json:"content,omitempty"`
	Visibility  *VisibilityValue `url:"visibility,omitempty" json:"visibility,omitempty"`
}

// UpdateSnippet updates an existing snippet. The user must have
// permission to change an existing snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/snippets.html#update-snippet
func (s *SnippetsService) UpdateSnippet(snippet int, opt *UpdateSnippetOptions, options ...RequestOptionFunc) (*Snippet, *Response, error) {
	u := fmt.Sprintf("snippets/%d", snippet)

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

// DeleteSnippet deletes an existing snippet. This is an idempotent
// function and deleting a non-existent snippet still returns a 200 OK status
// code.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/snippets.html#delete-snippet
func (s *SnippetsService) DeleteSnippet(snippet int, options ...RequestOptionFunc) (*Response, error) {
	u := fmt.Sprintf("snippets/%d", snippet)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// SnippetContent returns the raw snippet as plain text.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/snippets.html#snippet-content
func (s *SnippetsService) SnippetContent(snippet int, options ...RequestOptionFunc) ([]byte, *Response, error) {
	u := fmt.Sprintf("snippets/%d/raw", snippet)

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

// ExploreSnippetsOptions represents the available ExploreSnippets() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/snippets.html#explore-all-public-snippets
type ExploreSnippetsOptions ListOptions

// ExploreSnippets gets the list of public snippets.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/snippets.html#explore-all-public-snippets
func (s *SnippetsService) ExploreSnippets(opt *ExploreSnippetsOptions, options ...RequestOptionFunc) ([]*Snippet, *Response, error) {
	req, err := s.client.NewRequest("GET", "snippets/public", nil, options)
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
