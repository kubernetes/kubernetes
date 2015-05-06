// Copyright 2014 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "fmt"

// Subscription identifies a repository or thread subscription.
type Subscription struct {
	Subscribed *bool      `json:"subscribed,omitempty"`
	Ignored    *bool      `json:"ignored,omitempty"`
	Reason     *string    `json:"reason,omitempty"`
	CreatedAt  *Timestamp `json:"created_at,omitempty"`
	URL        *string    `json:"url,omitempty"`

	// only populated for repository subscriptions
	RepositoryURL *string `json:"repository_url,omitempty"`

	// only populated for thread subscriptions
	ThreadURL *string `json:"thread_url,omitempty"`
}

// ListWatchers lists watchers of a particular repo.
//
// GitHub API Docs: http://developer.github.com/v3/activity/watching/#list-watchers
func (s *ActivityService) ListWatchers(owner, repo string, opt *ListOptions) ([]User, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/subscribers", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	watchers := new([]User)
	resp, err := s.client.Do(req, watchers)
	if err != nil {
		return nil, resp, err
	}

	return *watchers, resp, err
}

// ListWatched lists the repositories the specified user is watching.  Passing
// the empty string will fetch watched repos for the authenticated user.
//
// GitHub API Docs: https://developer.github.com/v3/activity/watching/#list-repositories-being-watched
func (s *ActivityService) ListWatched(user string) ([]Repository, *Response, error) {
	var u string
	if user != "" {
		u = fmt.Sprintf("users/%v/subscriptions", user)
	} else {
		u = "user/subscriptions"
	}
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	watched := new([]Repository)
	resp, err := s.client.Do(req, watched)
	if err != nil {
		return nil, resp, err
	}

	return *watched, resp, err
}

// GetRepositorySubscription returns the subscription for the specified
// repository for the authenticated user.  If the authenticated user is not
// watching the repository, a nil Subscription is returned.
//
// GitHub API Docs: https://developer.github.com/v3/activity/watching/#get-a-repository-subscription
func (s *ActivityService) GetRepositorySubscription(owner, repo string) (*Subscription, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/subscription", owner, repo)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	sub := new(Subscription)
	resp, err := s.client.Do(req, sub)
	if err != nil {
		// if it's just a 404, don't return that as an error
		_, err = parseBoolResponse(err)
		return nil, resp, err
	}

	return sub, resp, err
}

// SetRepositorySubscription sets the subscription for the specified repository
// for the authenticated user.
//
// GitHub API Docs: https://developer.github.com/v3/activity/watching/#set-a-repository-subscription
func (s *ActivityService) SetRepositorySubscription(owner, repo string, subscription *Subscription) (*Subscription, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/subscription", owner, repo)

	req, err := s.client.NewRequest("PUT", u, subscription)
	if err != nil {
		return nil, nil, err
	}

	sub := new(Subscription)
	resp, err := s.client.Do(req, sub)
	if err != nil {
		return nil, resp, err
	}

	return sub, resp, err
}

// DeleteRepositorySubscription deletes the subscription for the specified
// repository for the authenticated user.
//
// GitHub API Docs: https://developer.github.com/v3/activity/watching/#delete-a-repository-subscription
func (s *ActivityService) DeleteRepositorySubscription(owner, repo string) (*Response, error) {
	u := fmt.Sprintf("repos/%s/%s/subscription", owner, repo)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
