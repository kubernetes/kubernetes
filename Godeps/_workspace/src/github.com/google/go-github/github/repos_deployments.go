// Copyright 2014 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"encoding/json"
	"fmt"
)

// Deployment represents a deployment in a repo
type Deployment struct {
	URL         *string         `json:"url,omitempty"`
	ID          *int            `json:"id,omitempty"`
	SHA         *string         `json:"sha,omitempty"`
	Ref         *string         `json:"ref,omitempty"`
	Task        *string         `json:"task,omitempty"`
	Payload     json.RawMessage `json:"payload,omitempty"`
	Environment *string         `json:"environment,omitempty"`
	Description *string         `json:"description,omitempty"`
	Creator     *User           `json:"creator,omitempty"`
	CreatedAt   *Timestamp      `json:"created_at,omitempty"`
	UpdatedAt   *Timestamp      `json:"pushed_at,omitempty"`
}

// DeploymentRequest represents a deployment request
type DeploymentRequest struct {
	Ref              *string   `json:"ref,omitempty"`
	Task             *string   `json:"task,omitempty"`
	AutoMerge        *bool     `json:"auto_merge,omitempty"`
	RequiredContexts *[]string `json:"required_contexts,omitempty"`
	Payload          *string   `json:"payload,omitempty"`
	Environment      *string   `json:"environment,omitempty"`
	Description      *string   `json:"description,omitempty"`
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
// GitHub API docs: https://developer.github.com/v3/repos/deployments/#list-deployments
func (s *RepositoriesService) ListDeployments(owner, repo string, opt *DeploymentsListOptions) ([]Deployment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/deployments", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	deployments := new([]Deployment)
	resp, err := s.client.Do(req, deployments)
	if err != nil {
		return nil, resp, err
	}

	return *deployments, resp, err
}

// CreateDeployment creates a new deployment for a repository.
//
// GitHub API docs: https://developer.github.com/v3/repos/deployments/#create-a-deployment
func (s *RepositoriesService) CreateDeployment(owner, repo string, request *DeploymentRequest) (*Deployment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/deployments", owner, repo)

	req, err := s.client.NewRequest("POST", u, request)
	if err != nil {
		return nil, nil, err
	}

	d := new(Deployment)
	resp, err := s.client.Do(req, d)
	if err != nil {
		return nil, resp, err
	}

	return d, resp, err
}

// DeploymentStatus represents the status of a
// particular deployment.
type DeploymentStatus struct {
	ID          *int       `json:"id,omitempty"`
	State       *string    `json:"state,omitempty"`
	Creator     *User      `json:"creator,omitempty"`
	Description *string    `json:"description,omitempty"`
	TargetURL   *string    `json:"target_url,omitempty"`
	CreatedAt   *Timestamp `json:"created_at,omitempty"`
	UpdatedAt   *Timestamp `json:"pushed_at,omitempty"`
}

// DeploymentStatusRequest represents a deployment request
type DeploymentStatusRequest struct {
	State       *string `json:"state,omitempty"`
	TargetURL   *string `json:"target_url,omitempty"`
	Description *string `json:"description,omitempty"`
}

// ListDeploymentStatuses lists the statuses of a given deployment of a repository.
//
// GitHub API docs: https://developer.github.com/v3/repos/deployments/#list-deployment-statuses
func (s *RepositoriesService) ListDeploymentStatuses(owner, repo string, deployment int, opt *ListOptions) ([]DeploymentStatus, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/deployments/%v/statuses", owner, repo, deployment)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	statuses := new([]DeploymentStatus)
	resp, err := s.client.Do(req, statuses)
	if err != nil {
		return nil, resp, err
	}

	return *statuses, resp, err
}

// CreateDeploymentStatus creates a new status for a deployment.
//
// GitHub API docs: https://developer.github.com/v3/repos/deployments/#create-a-deployment-status
func (s *RepositoriesService) CreateDeploymentStatus(owner, repo string, deployment int, request *DeploymentStatusRequest) (*DeploymentStatus, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/deployments/%v/statuses", owner, repo, deployment)

	req, err := s.client.NewRequest("POST", u, request)
	if err != nil {
		return nil, nil, err
	}

	d := new(DeploymentStatus)
	resp, err := s.client.Do(req, d)
	if err != nil {
		return nil, resp, err
	}

	return d, resp, err
}
