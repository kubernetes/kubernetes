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

// ProjectBadge represents a project badge.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_badges.html#list-all-badges-of-a-project
type ProjectBadge struct {
	ID               int    `json:"id"`
	LinkURL          string `json:"link_url"`
	ImageURL         string `json:"image_url"`
	RenderedLinkURL  string `json:"rendered_link_url"`
	RenderedImageURL string `json:"rendered_image_url"`
	// Kind represents a project badge kind. Can be empty, when used PreviewProjectBadge().
	Kind string `json:"kind"`
}

// ProjectBadgesService handles communication with the project badges
// related methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/project_badges.html
type ProjectBadgesService struct {
	client *Client
}

// ListProjectBadgesOptions represents the available ListProjectBadges()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_badges.html#list-all-badges-of-a-project
type ListProjectBadgesOptions ListOptions

// ListProjectBadges gets a list of a project's badges and its group badges.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_badges.html#list-all-badges-of-a-project
func (s *ProjectBadgesService) ListProjectBadges(pid interface{}, opt *ListProjectBadgesOptions, options ...RequestOptionFunc) ([]*ProjectBadge, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/badges", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var pb []*ProjectBadge
	resp, err := s.client.Do(req, &pb)
	if err != nil {
		return nil, resp, err
	}

	return pb, resp, err
}

// GetProjectBadge gets a project badge.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_badges.html#get-a-badge-of-a-project
func (s *ProjectBadgesService) GetProjectBadge(pid interface{}, badge int, options ...RequestOptionFunc) (*ProjectBadge, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/badges/%d", pathEscape(project), badge)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	pb := new(ProjectBadge)
	resp, err := s.client.Do(req, pb)
	if err != nil {
		return nil, resp, err
	}

	return pb, resp, err
}

// AddProjectBadgeOptions represents the available AddProjectBadge() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_badges.html#add-a-badge-to-a-project
type AddProjectBadgeOptions struct {
	LinkURL  *string `url:"link_url,omitempty" json:"link_url,omitempty"`
	ImageURL *string `url:"image_url,omitempty" json:"image_url,omitempty"`
}

// AddProjectBadge adds a badge to a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_badges.html#add-a-badge-to-a-project
func (s *ProjectBadgesService) AddProjectBadge(pid interface{}, opt *AddProjectBadgeOptions, options ...RequestOptionFunc) (*ProjectBadge, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/badges", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	pb := new(ProjectBadge)
	resp, err := s.client.Do(req, pb)
	if err != nil {
		return nil, resp, err
	}

	return pb, resp, err
}

// EditProjectBadgeOptions represents the available EditProjectBadge() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_badges.html#edit-a-badge-of-a-project
type EditProjectBadgeOptions struct {
	LinkURL  *string `url:"link_url,omitempty" json:"link_url,omitempty"`
	ImageURL *string `url:"image_url,omitempty" json:"image_url,omitempty"`
}

// EditProjectBadge updates a badge of a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_badges.html#edit-a-badge-of-a-project
func (s *ProjectBadgesService) EditProjectBadge(pid interface{}, badge int, opt *EditProjectBadgeOptions, options ...RequestOptionFunc) (*ProjectBadge, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/badges/%d", pathEscape(project), badge)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	pb := new(ProjectBadge)
	resp, err := s.client.Do(req, pb)
	if err != nil {
		return nil, resp, err
	}

	return pb, resp, err
}

// DeleteProjectBadge removes a badge from a project. Only project's
// badges will be removed by using this endpoint.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_badges.html#remove-a-badge-from-a-project
func (s *ProjectBadgesService) DeleteProjectBadge(pid interface{}, badge int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/badges/%d", pathEscape(project), badge)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ProjectBadgePreviewOptions represents the available PreviewProjectBadge() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_badges.html#preview-a-badge-from-a-project
type ProjectBadgePreviewOptions struct {
	LinkURL  *string `url:"link_url,omitempty" json:"link_url,omitempty"`
	ImageURL *string `url:"image_url,omitempty" json:"image_url,omitempty"`
}

// PreviewProjectBadge returns how the link_url and image_url final URLs would be after
// resolving the placeholder interpolation.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/project_badges.html#preview-a-badge-from-a-project
func (s *ProjectBadgesService) PreviewProjectBadge(pid interface{}, opt *ProjectBadgePreviewOptions, options ...RequestOptionFunc) (*ProjectBadge, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/badges/render", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	pb := new(ProjectBadge)
	resp, err := s.client.Do(req, &pb)
	if err != nil {
		return nil, resp, err
	}

	return pb, resp, err
}
