// Copyright 2016 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// TrafficReferrer represent information about traffic from a referrer .
type TrafficReferrer struct {
	Referrer *string `json:"referrer,omitempty"`
	Count    *int    `json:"count,omitempty"`
	Uniques  *int    `json:"uniques,omitempty"`
}

// TrafficPath represent information about the traffic on a path of the repo.
type TrafficPath struct {
	Path    *string `json:"path,omitempty"`
	Title   *string `json:"title,omitempty"`
	Count   *int    `json:"count,omitempty"`
	Uniques *int    `json:"uniques,omitempty"`
}

// TrafficData represent information about a specific timestamp in views or clones list.
type TrafficData struct {
	Timestamp *Timestamp `json:"timestamp,omitempty"`
	Count     *int       `json:"count,omitempty"`
	Uniques   *int       `json:"uniques,omitempty"`
}

// TrafficViews represent information about the number of views in the last 14 days.
type TrafficViews struct {
	Views   []*TrafficData `json:"views,omitempty"`
	Count   *int           `json:"count,omitempty"`
	Uniques *int           `json:"uniques,omitempty"`
}

// TrafficClones represent information about the number of clones in the last 14 days.
type TrafficClones struct {
	Clones  []*TrafficData `json:"clones,omitempty"`
	Count   *int           `json:"count,omitempty"`
	Uniques *int           `json:"uniques,omitempty"`
}

// TrafficBreakdownOptions specifies the parameters to methods that support breakdown per day or week.
// Can be one of: day, week. Default: day.
type TrafficBreakdownOptions struct {
	Per string `url:"per,omitempty"`
}

// ListTrafficReferrers list the top 10 referrers over the last 14 days.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-top-referral-sources
func (s *RepositoriesService) ListTrafficReferrers(ctx context.Context, owner, repo string) ([]*TrafficReferrer, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/traffic/popular/referrers", owner, repo)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var trafficReferrers []*TrafficReferrer
	resp, err := s.client.Do(ctx, req, &trafficReferrers)
	if err != nil {
		return nil, resp, err
	}

	return trafficReferrers, resp, nil
}

// ListTrafficPaths list the top 10 popular content over the last 14 days.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-top-referral-paths
func (s *RepositoriesService) ListTrafficPaths(ctx context.Context, owner, repo string) ([]*TrafficPath, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/traffic/popular/paths", owner, repo)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var paths []*TrafficPath
	resp, err := s.client.Do(ctx, req, &paths)
	if err != nil {
		return nil, resp, err
	}

	return paths, resp, nil
}

// ListTrafficViews get total number of views for the last 14 days and breaks it down either per day or week.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-page-views
func (s *RepositoriesService) ListTrafficViews(ctx context.Context, owner, repo string, opts *TrafficBreakdownOptions) (*TrafficViews, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/traffic/views", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	trafficViews := new(TrafficViews)
	resp, err := s.client.Do(ctx, req, &trafficViews)
	if err != nil {
		return nil, resp, err
	}

	return trafficViews, resp, nil
}

// ListTrafficClones get total number of clones for the last 14 days and breaks it down either per day or week for the last 14 days.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-repository-clones
func (s *RepositoriesService) ListTrafficClones(ctx context.Context, owner, repo string, opts *TrafficBreakdownOptions) (*TrafficClones, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/traffic/clones", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	trafficClones := new(TrafficClones)
	resp, err := s.client.Do(ctx, req, &trafficClones)
	if err != nil {
		return nil, resp, err
	}

	return trafficClones, resp, nil
}
