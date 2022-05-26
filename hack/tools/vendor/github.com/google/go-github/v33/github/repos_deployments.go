// Copyright 2014 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// Deployment represents a deployment in a repo
type Deployment struct {
	URL           *string         `json:"url,omitempty"`
	ID            *int64          `json:"id,omitempty"`
	SHA           *string         `json:"sha,omitempty"`
	Ref           *string         `json:"ref,omitempty"`
	Task          *string         `json:"task,omitempty"`
	Payload       json.RawMessage `json:"payload,omitempty"`
	Environment   *string         `json:"environment,omitempty"`
	Description   *string         `json:"description,omitempty"`
	Creator       *User           `json:"creator,omitempty"`
	CreatedAt     *Timestamp      `json:"created_at,omitempty"`
	UpdatedAt     *Timestamp      `json:"updated_at,omitempty"`
	StatusesURL   *string         `json:"statuses_url,omitempty"`
	RepositoryURL *string         `json:"repository_url,omitempty"`
	NodeID        *string         `json:"node_id,omitempty"`
}

// DeploymentRequest represents a deployment request
type DeploymentRequest struct {
	Ref                   *string     `json:"ref,omitempty"`
	Task                  *string     `json:"task,omitempty"`
	AutoMerge             *bool       `json:"auto_merge,omitempty"`
	RequiredContexts      *[]string   `json:"required_contexts,omitempty"`
	Payload               interface{} `json:"payload,omitempty"`
	Environment           *string     `json:"environment,omitempty"`
	Description           *string     `json:"description,omitempty"`
	TransientEnvironment  *bool       `json:"transient_environment,omitempty"`
	ProductionEnvironment *bool       `json:"production_environment,omitempty"`
}

// DeploymentsListOptions specifies the optional parameters to the
// RepositoriesService.ListDeployments method.
type DeploymentsListOptions struct {
	// SHA of the Deployment.
	SHA string `url:"sha,omitempty"`

	// List deployments for a given ref.
	Ref string `url:"ref,omitempty"`

	// List deployments for a given task.
	Task string `url:"task,omitempty"`

	// List deployments for a given environment.
	Environment string `url:"environment,omitempty"`

	ListOptions
}

// ListDeployments lists the deployments of a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#list-deployments
func (s *RepositoriesService) ListDeployments(ctx context.Context, owner, repo string, opts *DeploymentsListOptions) ([]*Deployment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/deployments", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var deployments []*Deployment
	resp, err := s.client.Do(ctx, req, &deployments)
	if err != nil {
		return nil, resp, err
	}

	return deployments, resp, nil
}

// GetDeployment returns a single deployment of a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-a-deployment
func (s *RepositoriesService) GetDeployment(ctx context.Context, owner, repo string, deploymentID int64) (*Deployment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/deployments/%v", owner, repo, deploymentID)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	deployment := new(Deployment)
	resp, err := s.client.Do(ctx, req, deployment)
	if err != nil {
		return nil, resp, err
	}

	return deployment, resp, nil
}

// CreateDeployment creates a new deployment for a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#create-a-deployment
func (s *RepositoriesService) CreateDeployment(ctx context.Context, owner, repo string, request *DeploymentRequest) (*Deployment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/deployments", owner, repo)

	req, err := s.client.NewRequest("POST", u, request)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	acceptHeaders := []string{mediaTypeDeploymentStatusPreview, mediaTypeExpandDeploymentStatusPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	d := new(Deployment)
	resp, err := s.client.Do(ctx, req, d)
	if err != nil {
		return nil, resp, err
	}

	return d, resp, nil
}

// DeleteDeployment deletes an existing deployment for a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#delete-a-deployment
func (s *RepositoriesService) DeleteDeployment(ctx context.Context, owner, repo string, deploymentID int64) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/deployments/%v", owner, repo, deploymentID)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}

// DeploymentStatus represents the status of a
// particular deployment.
type DeploymentStatus struct {
	ID *int64 `json:"id,omitempty"`
	// State is the deployment state.
	// Possible values are: "pending", "success", "failure", "error",
	// "inactive", "in_progress", "queued".
	State          *string    `json:"state,omitempty"`
	Creator        *User      `json:"creator,omitempty"`
	Description    *string    `json:"description,omitempty"`
	Environment    *string    `json:"environment,omitempty"`
	NodeID         *string    `json:"node_id,omitempty"`
	CreatedAt      *Timestamp `json:"created_at,omitempty"`
	UpdatedAt      *Timestamp `json:"updated_at,omitempty"`
	TargetURL      *string    `json:"target_url,omitempty"`
	DeploymentURL  *string    `json:"deployment_url,omitempty"`
	RepositoryURL  *string    `json:"repository_url,omitempty"`
	EnvironmentURL *string    `json:"environment_url,omitempty"`
	LogURL         *string    `json:"log_url,omitempty"`
	URL            *string    `json:"url,omitempty"`
}

// DeploymentStatusRequest represents a deployment request
type DeploymentStatusRequest struct {
	State          *string `json:"state,omitempty"`
	LogURL         *string `json:"log_url,omitempty"`
	Description    *string `json:"description,omitempty"`
	Environment    *string `json:"environment,omitempty"`
	EnvironmentURL *string `json:"environment_url,omitempty"`
	AutoInactive   *bool   `json:"auto_inactive,omitempty"`
}

// ListDeploymentStatuses lists the statuses of a given deployment of a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#list-deployment-statuses
func (s *RepositoriesService) ListDeploymentStatuses(ctx context.Context, owner, repo string, deployment int64, opts *ListOptions) ([]*DeploymentStatus, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/deployments/%v/statuses", owner, repo, deployment)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	acceptHeaders := []string{mediaTypeDeploymentStatusPreview, mediaTypeExpandDeploymentStatusPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	var statuses []*DeploymentStatus
	resp, err := s.client.Do(ctx, req, &statuses)
	if err != nil {
		return nil, resp, err
	}

	return statuses, resp, nil
}

// GetDeploymentStatus returns a single deployment status of a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-a-deployment-status
func (s *RepositoriesService) GetDeploymentStatus(ctx context.Context, owner, repo string, deploymentID, deploymentStatusID int64) (*DeploymentStatus, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/deployments/%v/statuses/%v", owner, repo, deploymentID, deploymentStatusID)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	acceptHeaders := []string{mediaTypeDeploymentStatusPreview, mediaTypeExpandDeploymentStatusPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	d := new(DeploymentStatus)
	resp, err := s.client.Do(ctx, req, d)
	if err != nil {
		return nil, resp, err
	}

	return d, resp, nil
}

// CreateDeploymentStatus creates a new status for a deployment.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#create-a-deployment-status
func (s *RepositoriesService) CreateDeploymentStatus(ctx context.Context, owner, repo string, deployment int64, request *DeploymentStatusRequest) (*DeploymentStatus, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/deployments/%v/statuses", owner, repo, deployment)

	req, err := s.client.NewRequest("POST", u, request)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	acceptHeaders := []string{mediaTypeDeploymentStatusPreview, mediaTypeExpandDeploymentStatusPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	d := new(DeploymentStatus)
	resp, err := s.client.Do(ctx, req, d)
	if err != nil {
		return nil, resp, err
	}

	return d, resp, nil
}
