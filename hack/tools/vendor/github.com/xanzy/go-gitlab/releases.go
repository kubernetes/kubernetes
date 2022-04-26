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

// ReleasesService handles communication with the releases methods
// of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/releases/index.html
type ReleasesService struct {
	client *Client
}

// Release represents a project release.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/releases/index.html#list-releases
type Release struct {
	TagName         string     `json:"tag_name"`
	Name            string     `json:"name"`
	Description     string     `json:"description,omitempty"`
	DescriptionHTML string     `json:"description_html,omitempty"`
	CreatedAt       *time.Time `json:"created_at,omitempty"`
	Author          struct {
		ID        int    `json:"id"`
		Name      string `json:"name"`
		Username  string `json:"username"`
		State     string `json:"state"`
		AvatarURL string `json:"avatar_url"`
		WebURL    string `json:"web_url"`
	} `json:"author"`
	Commit Commit `json:"commit"`
	Assets struct {
		Count   int `json:"count"`
		Sources []struct {
			Format string `json:"format"`
			URL    string `json:"url"`
		} `json:"sources"`
		Links []*ReleaseLink `json:"links"`
	} `json:"assets"`
}

// ListReleasesOptions represents ListReleases() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/releases/index.html#list-releases
type ListReleasesOptions ListOptions

// ListReleases gets a pagenated of releases accessible by the authenticated user.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/releases/index.html#list-releases
func (s *ReleasesService) ListReleases(pid interface{}, opt *ListReleasesOptions, options ...RequestOptionFunc) ([]*Release, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/releases", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var rs []*Release
	resp, err := s.client.Do(req, &rs)
	if err != nil {
		return nil, resp, err
	}

	return rs, resp, err
}

// GetRelease returns a single release, identified by a tag name.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/releases/index.html#get-a-release-by-a-tag-name
func (s *ReleasesService) GetRelease(pid interface{}, tagName string, options ...RequestOptionFunc) (*Release, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/releases/%s", pathEscape(project), tagName)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	r := new(Release)
	resp, err := s.client.Do(req, r)
	if err != nil {
		return nil, resp, err
	}

	return r, resp, err
}

// ReleaseAssets represents release assets in CreateRelease() options
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/releases/index.html#create-a-release
type ReleaseAssets struct {
	Links []*ReleaseAssetLink `url:"links" json:"links"`
}

// ReleaseAssetLink represents release asset link in CreateRelease() options
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/releases/index.html#create-a-release
type ReleaseAssetLink struct {
	Name string `url:"name" json:"name"`
	URL  string `url:"url" json:"url"`
}

// CreateReleaseOptions represents CreateRelease() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/releases/index.html#create-a-release
type CreateReleaseOptions struct {
	Name        *string        `url:"name" json:"name"`
	TagName     *string        `url:"tag_name" json:"tag_name"`
	Description *string        `url:"description" json:"description"`
	Ref         *string        `url:"ref,omitempty" json:"ref,omitempty"`
	Milestones  []string       `url:"milestones,omitempty" json:"milestones,omitempty"`
	Assets      *ReleaseAssets `url:"assets,omitempty" json:"assets,omitempty"`
	ReleasedAt  *time.Time     `url:"released_at,omitempty" json:"released_at,omitempty"`
}

// CreateRelease creates a release.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/releases/index.html#create-a-release
func (s *ReleasesService) CreateRelease(pid interface{}, opts *CreateReleaseOptions, options ...RequestOptionFunc) (*Release, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/releases", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opts, options)
	if err != nil {
		return nil, nil, err
	}

	r := new(Release)
	resp, err := s.client.Do(req, r)
	if err != nil {
		return nil, resp, err
	}

	return r, resp, err
}

// UpdateReleaseOptions represents UpdateRelease() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/releases/index.html#update-a-release
type UpdateReleaseOptions struct {
	Name        *string    `url:"name" json:"name"`
	Description *string    `url:"description" json:"description"`
	Milestones  []string   `url:"milestones,omitempty" json:"milestones,omitempty"`
	ReleasedAt  *time.Time `url:"released_at,omitempty" json:"released_at,omitempty"`
}

// UpdateRelease updates a release.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/releases/index.html#update-a-release
func (s *ReleasesService) UpdateRelease(pid interface{}, tagName string, opts *UpdateReleaseOptions, options ...RequestOptionFunc) (*Release, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/releases/%s", pathEscape(project), tagName)

	req, err := s.client.NewRequest("PUT", u, opts, options)
	if err != nil {
		return nil, nil, err
	}

	r := new(Release)
	resp, err := s.client.Do(req, &r)
	if err != nil {
		return nil, resp, err
	}

	return r, resp, err
}

// DeleteRelease deletes a release.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/releases/index.html#delete-a-release
func (s *ReleasesService) DeleteRelease(pid interface{}, tagName string, options ...RequestOptionFunc) (*Release, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/releases/%s", pathEscape(project), tagName)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	r := new(Release)
	resp, err := s.client.Do(req, r)
	if err != nil {
		return nil, resp, err
	}

	return r, resp, err
}
