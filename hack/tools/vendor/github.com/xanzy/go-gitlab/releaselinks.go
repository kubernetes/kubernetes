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

// ReleaseLinksService handles communication with the release link methods
// of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/releases/links.html
type ReleaseLinksService struct {
	client *Client
}

// ReleaseLink represents a release link.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/releases/links.html
type ReleaseLink struct {
	ID       int    `json:"id"`
	Name     string `json:"name"`
	URL      string `json:"url"`
	External bool   `json:"external"`
}

// ListReleaseLinksOptions represents ListReleaseLinks() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/releases/links.html#get-links
type ListReleaseLinksOptions ListOptions

// ListReleaseLinks gets assets as links from a Release.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/releases/links.html#get-links
func (s *ReleaseLinksService) ListReleaseLinks(pid interface{}, tagName string, opt *ListReleaseLinksOptions, options ...RequestOptionFunc) ([]*ReleaseLink, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/releases/%s/assets/links", pathEscape(project), tagName)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var rls []*ReleaseLink
	resp, err := s.client.Do(req, &rls)
	if err != nil {
		return nil, resp, err
	}

	return rls, resp, err
}

// GetReleaseLink returns a link from release assets.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/releases/links.html#get-a-link
func (s *ReleaseLinksService) GetReleaseLink(pid interface{}, tagName string, link int, options ...RequestOptionFunc) (*ReleaseLink, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/releases/%s/assets/links/%d",
		pathEscape(project),
		tagName,
		link)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	rl := new(ReleaseLink)
	resp, err := s.client.Do(req, rl)
	if err != nil {
		return nil, resp, err
	}

	return rl, resp, err
}

// CreateReleaseLinkOptions represents CreateReleaseLink() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/releases/links.html#create-a-link
type CreateReleaseLinkOptions struct {
	Name *string `url:"name" json:"name"`
	URL  *string `url:"url" json:"url"`
}

// CreateReleaseLink creates a link.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/releases/links.html#create-a-link
func (s *ReleaseLinksService) CreateReleaseLink(pid interface{}, tagName string, opt *CreateReleaseLinkOptions, options ...RequestOptionFunc) (*ReleaseLink, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/releases/%s/assets/links", pathEscape(project), tagName)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	rl := new(ReleaseLink)
	resp, err := s.client.Do(req, rl)
	if err != nil {
		return nil, resp, err
	}

	return rl, resp, err
}

// UpdateReleaseLinkOptions represents UpdateReleaseLink() options.
//
// You have to specify at least one of Name of URL.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/releases/links.html#update-a-link
type UpdateReleaseLinkOptions struct {
	Name *string `url:"name,omitempty" json:"name,omitempty"`
	URL  *string `url:"url,omitempty" json:"url,omitempty"`
}

// UpdateReleaseLink updates an asset link.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/releases/links.html#update-a-link
func (s *ReleaseLinksService) UpdateReleaseLink(pid interface{}, tagName string, link int, opt *UpdateReleaseLinkOptions, options ...RequestOptionFunc) (*ReleaseLink, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/releases/%s/assets/links/%d",
		pathEscape(project),
		tagName,
		link)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	rl := new(ReleaseLink)
	resp, err := s.client.Do(req, rl)
	if err != nil {
		return nil, resp, err
	}

	return rl, resp, err
}

// DeleteReleaseLink deletes a link from release.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/releases/links.html#delete-a-link
func (s *ReleaseLinksService) DeleteReleaseLink(pid interface{}, tagName string, link int, options ...RequestOptionFunc) (*ReleaseLink, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/releases/%s/assets/links/%d",
		pathEscape(project),
		tagName,
		link,
	)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	rl := new(ReleaseLink)
	resp, err := s.client.Do(req, rl)
	if err != nil {
		return nil, resp, err
	}

	return rl, resp, err
}
