// Copyright 2017 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
	"time"
)

// Metric represents the different fields for one file in community health files.
type Metric struct {
	Name    *string `json:"name"`
	Key     *string `json:"key"`
	URL     *string `json:"url"`
	HTMLURL *string `json:"html_url"`
}

// CommunityHealthFiles represents the different files in the community health metrics response.
type CommunityHealthFiles struct {
	CodeOfConduct       *Metric `json:"code_of_conduct"`
	Contributing        *Metric `json:"contributing"`
	IssueTemplate       *Metric `json:"issue_template"`
	PullRequestTemplate *Metric `json:"pull_request_template"`
	License             *Metric `json:"license"`
	Readme              *Metric `json:"readme"`
}

// CommunityHealthMetrics represents a response containing the community metrics of a repository.
type CommunityHealthMetrics struct {
	HealthPercentage *int                  `json:"health_percentage"`
	Files            *CommunityHealthFiles `json:"files"`
	UpdatedAt        *time.Time            `json:"updated_at"`
}

// GetCommunityHealthMetrics retrieves all the community health  metrics for a  repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-community-profile-metrics
func (s *RepositoriesService) GetCommunityHealthMetrics(ctx context.Context, owner, repo string) (*CommunityHealthMetrics, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/community/profile", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeRepositoryCommunityHealthMetricsPreview)

	metrics := &CommunityHealthMetrics{}
	resp, err := s.client.Do(ctx, req, metrics)
	if err != nil {
		return nil, resp, err
	}

	return metrics, resp, nil
}
