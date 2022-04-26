// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "context"

// ActivityService handles communication with the activity related
// methods of the GitHub API.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/activity/
type ActivityService service

// FeedLink represents a link to a related resource.
type FeedLink struct {
	HRef *string `json:"href,omitempty"`
	Type *string `json:"type,omitempty"`
}

// Feeds represents timeline resources in Atom format.
type Feeds struct {
	TimelineURL                 *string  `json:"timeline_url,omitempty"`
	UserURL                     *string  `json:"user_url,omitempty"`
	CurrentUserPublicURL        *string  `json:"current_user_public_url,omitempty"`
	CurrentUserURL              *string  `json:"current_user_url,omitempty"`
	CurrentUserActorURL         *string  `json:"current_user_actor_url,omitempty"`
	CurrentUserOrganizationURL  *string  `json:"current_user_organization_url,omitempty"`
	CurrentUserOrganizationURLs []string `json:"current_user_organization_urls,omitempty"`
	Links                       *struct {
		Timeline                 *FeedLink   `json:"timeline,omitempty"`
		User                     *FeedLink   `json:"user,omitempty"`
		CurrentUserPublic        *FeedLink   `json:"current_user_public,omitempty"`
		CurrentUser              *FeedLink   `json:"current_user,omitempty"`
		CurrentUserActor         *FeedLink   `json:"current_user_actor,omitempty"`
		CurrentUserOrganization  *FeedLink   `json:"current_user_organization,omitempty"`
		CurrentUserOrganizations []*FeedLink `json:"current_user_organizations,omitempty"`
	} `json:"_links,omitempty"`
}

// ListFeeds lists all the feeds available to the authenticated user.
//
// GitHub provides several timeline resources in Atom format:
//     Timeline: The GitHub global public timeline
//     User: The public timeline for any user, using URI template
//     Current user public: The public timeline for the authenticated user
//     Current user: The private timeline for the authenticated user
//     Current user actor: The private timeline for activity created by the
//         authenticated user
//     Current user organizations: The private timeline for the organizations
//         the authenticated user is a member of.
//
// Note: Private feeds are only returned when authenticating via Basic Auth
// since current feed URIs use the older, non revocable auth tokens.
func (s *ActivityService) ListFeeds(ctx context.Context) (*Feeds, *Response, error) {
	req, err := s.client.NewRequest("GET", "feeds", nil)
	if err != nil {
		return nil, nil, err
	}

	f := &Feeds{}
	resp, err := s.client.Do(ctx, req, f)
	if err != nil {
		return nil, resp, err
	}

	return f, resp, nil
}
