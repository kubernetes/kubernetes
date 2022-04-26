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

// GroupBadgesService handles communication with the group badges
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_badges.html
type GroupBadgesService struct {
	client *Client
}

// BadgeKind represents a GitLab Badge Kind
type BadgeKind string

// all possible values Badge Kind
const (
	ProjectBadgeKind BadgeKind = "project"
	GroupBadgeKind   BadgeKind = "group"
)

// GroupBadge represents a group badge.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_badges.html
type GroupBadge struct {
	ID               int       `json:"id"`
	LinkURL          string    `json:"link_url"`
	ImageURL         string    `json:"image_url"`
	RenderedLinkURL  string    `json:"rendered_link_url"`
	RenderedImageURL string    `json:"rendered_image_url"`
	Kind             BadgeKind `json:"kind"`
}

// ListGroupBadgesOptions represents the available ListGroupBadges() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_badges.html#list-all-badges-of-a-group
type ListGroupBadgesOptions ListOptions

// ListGroupBadges gets a list of a group badges.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_badges.html#list-all-badges-of-a-group
func (s *GroupBadgesService) ListGroupBadges(gid interface{}, opt *ListGroupBadgesOptions, options ...RequestOptionFunc) ([]*GroupBadge, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/badges", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var gb []*GroupBadge
	resp, err := s.client.Do(req, &gb)
	if err != nil {
		return nil, resp, err
	}

	return gb, resp, err
}

// GetGroupBadge gets a group badge.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_badges.html#get-a-badge-of-a-group
func (s *GroupBadgesService) GetGroupBadge(gid interface{}, badge int, options ...RequestOptionFunc) (*GroupBadge, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/badges/%d", pathEscape(group), badge)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	gb := new(GroupBadge)
	resp, err := s.client.Do(req, gb)
	if err != nil {
		return nil, resp, err
	}

	return gb, resp, err
}

// AddGroupBadgeOptions represents the available AddGroupBadge() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_badges.html#add-a-badge-to-a-group
type AddGroupBadgeOptions struct {
	LinkURL  *string `url:"link_url,omitempty" json:"link_url,omitempty"`
	ImageURL *string `url:"image_url,omitempty" json:"image_url,omitempty"`
}

// AddGroupBadge adds a badge to a group.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_badges.html#add-a-badge-to-a-group
func (s *GroupBadgesService) AddGroupBadge(gid interface{}, opt *AddGroupBadgeOptions, options ...RequestOptionFunc) (*GroupBadge, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/badges", pathEscape(group))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	gb := new(GroupBadge)
	resp, err := s.client.Do(req, gb)
	if err != nil {
		return nil, resp, err
	}

	return gb, resp, err
}

// EditGroupBadgeOptions represents the available EditGroupBadge() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_badges.html#edit-a-badge-of-a-group
type EditGroupBadgeOptions struct {
	LinkURL  *string `url:"link_url,omitempty" json:"link_url,omitempty"`
	ImageURL *string `url:"image_url,omitempty" json:"image_url,omitempty"`
}

// EditGroupBadge updates a badge of a group.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_badges.html#edit-a-badge-of-a-group
func (s *GroupBadgesService) EditGroupBadge(gid interface{}, badge int, opt *EditGroupBadgeOptions, options ...RequestOptionFunc) (*GroupBadge, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/badges/%d", pathEscape(group), badge)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	gb := new(GroupBadge)
	resp, err := s.client.Do(req, gb)
	if err != nil {
		return nil, resp, err
	}

	return gb, resp, err
}

// DeleteGroupBadge removes a badge from a group.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_badges.html#remove-a-badge-from-a-group
func (s *GroupBadgesService) DeleteGroupBadge(gid interface{}, badge int, options ...RequestOptionFunc) (*Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("groups/%s/badges/%d", pathEscape(group), badge)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// GroupBadgePreviewOptions represents the available PreviewGroupBadge() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_badges.html#preview-a-badge-from-a-group
type GroupBadgePreviewOptions struct {
	LinkURL  *string `url:"link_url,omitempty" json:"link_url,omitempty"`
	ImageURL *string `url:"image_url,omitempty" json:"image_url,omitempty"`
}

// PreviewGroupBadge returns how the link_url and image_url final URLs would be after
// resolving the placeholder interpolation.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/group_badges.html#preview-a-badge-from-a-group
func (s *GroupBadgesService) PreviewGroupBadge(gid interface{}, opt *GroupBadgePreviewOptions, options ...RequestOptionFunc) (*GroupBadge, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/badges/render", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	gb := new(GroupBadge)
	resp, err := s.client.Do(req, &gb)
	if err != nil {
		return nil, resp, err
	}

	return gb, resp, err
}
