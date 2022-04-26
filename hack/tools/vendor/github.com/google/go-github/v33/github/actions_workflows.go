// Copyright 2020 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// Workflow represents a repository action workflow.
type Workflow struct {
	ID        *int64     `json:"id,omitempty"`
	NodeID    *string    `json:"node_id,omitempty"`
	Name      *string    `json:"name,omitempty"`
	Path      *string    `json:"path,omitempty"`
	State     *string    `json:"state,omitempty"`
	CreatedAt *Timestamp `json:"created_at,omitempty"`
	UpdatedAt *Timestamp `json:"updated_at,omitempty"`
	URL       *string    `json:"url,omitempty"`
	HTMLURL   *string    `json:"html_url,omitempty"`
	BadgeURL  *string    `json:"badge_url,omitempty"`
}

// Workflows represents a slice of repository action workflows.
type Workflows struct {
	TotalCount *int        `json:"total_count,omitempty"`
	Workflows  []*Workflow `json:"workflows,omitempty"`
}

// WorkflowUsage represents a usage of a specific workflow.
type WorkflowUsage struct {
	Billable *WorkflowEnvironment `json:"billable,omitempty"`
}

// WorkflowEnvironment represents different runner environments available for a workflow.
type WorkflowEnvironment struct {
	Ubuntu  *WorkflowBill `json:"UBUNTU,omitempty"`
	MacOS   *WorkflowBill `json:"MACOS,omitempty"`
	Windows *WorkflowBill `json:"WINDOWS,omitempty"`
}

// WorkflowBill specifies billable time for a specific environment in a workflow.
type WorkflowBill struct {
	TotalMS *int64 `json:"total_ms,omitempty"`
}

// CreateWorkflowDispatchEventRequest represents a request to create a workflow dispatch event.
type CreateWorkflowDispatchEventRequest struct {
	// Ref represents the reference of the workflow run.
	// The reference can be a branch, tag, or a commit SHA.
	// Ref is required when creating a workflow dispatch event.
	Ref string `json:"ref"`
	// Inputs represents input keys and values configured in the workflow file.
	// The maximum number of properties is 10.
	// Default: Any default properties configured in the workflow file will be used when `inputs` are omitted.
	Inputs map[string]interface{} `json:"inputs,omitempty"`
}

// ListWorkflows lists all workflows in a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#list-repository-workflows
func (s *ActionsService) ListWorkflows(ctx context.Context, owner, repo string, opts *ListOptions) (*Workflows, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/actions/workflows", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	workflows := new(Workflows)
	resp, err := s.client.Do(ctx, req, &workflows)
	if err != nil {
		return nil, resp, err
	}

	return workflows, resp, nil
}

// GetWorkflowByID gets a specific workflow by ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#get-a-workflow
func (s *ActionsService) GetWorkflowByID(ctx context.Context, owner, repo string, workflowID int64) (*Workflow, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/workflows/%v", owner, repo, workflowID)

	return s.getWorkflow(ctx, u)
}

// GetWorkflowByFileName gets a specific workflow by file name.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#get-a-workflow
func (s *ActionsService) GetWorkflowByFileName(ctx context.Context, owner, repo, workflowFileName string) (*Workflow, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/workflows/%v", owner, repo, workflowFileName)

	return s.getWorkflow(ctx, u)
}

func (s *ActionsService) getWorkflow(ctx context.Context, url string) (*Workflow, *Response, error) {
	req, err := s.client.NewRequest("GET", url, nil)
	if err != nil {
		return nil, nil, err
	}

	workflow := new(Workflow)
	resp, err := s.client.Do(ctx, req, workflow)
	if err != nil {
		return nil, resp, err
	}

	return workflow, resp, nil
}

// GetWorkflowUsageByID gets a specific workflow usage by ID in the unit of billable milliseconds.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#get-workflow-usage
func (s *ActionsService) GetWorkflowUsageByID(ctx context.Context, owner, repo string, workflowID int64) (*WorkflowUsage, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/workflows/%v/timing", owner, repo, workflowID)

	return s.getWorkflowUsage(ctx, u)
}

// GetWorkflowUsageByFileName gets a specific workflow usage by file name in the unit of billable milliseconds.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#get-workflow-usage
func (s *ActionsService) GetWorkflowUsageByFileName(ctx context.Context, owner, repo, workflowFileName string) (*WorkflowUsage, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/workflows/%v/timing", owner, repo, workflowFileName)

	return s.getWorkflowUsage(ctx, u)
}

func (s *ActionsService) getWorkflowUsage(ctx context.Context, url string) (*WorkflowUsage, *Response, error) {
	req, err := s.client.NewRequest("GET", url, nil)
	if err != nil {
		return nil, nil, err
	}

	workflowUsage := new(WorkflowUsage)
	resp, err := s.client.Do(ctx, req, workflowUsage)
	if err != nil {
		return nil, resp, err
	}

	return workflowUsage, resp, nil
}

// CreateWorkflowDispatchEventByID manually triggers a GitHub Actions workflow run.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#create-a-workflow-dispatch-event
func (s *ActionsService) CreateWorkflowDispatchEventByID(ctx context.Context, owner, repo string, workflowID int64, event CreateWorkflowDispatchEventRequest) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/workflows/%v/dispatches", owner, repo, workflowID)

	return s.createWorkflowDispatchEvent(ctx, u, &event)
}

// CreateWorkflowDispatchEventByFileName manually triggers a GitHub Actions workflow run.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#create-a-workflow-dispatch-event
func (s *ActionsService) CreateWorkflowDispatchEventByFileName(ctx context.Context, owner, repo, workflowFileName string, event CreateWorkflowDispatchEventRequest) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/workflows/%v/dispatches", owner, repo, workflowFileName)

	return s.createWorkflowDispatchEvent(ctx, u, &event)
}

func (s *ActionsService) createWorkflowDispatchEvent(ctx context.Context, url string, event *CreateWorkflowDispatchEventRequest) (*Response, error) {
	req, err := s.client.NewRequest("POST", url, event)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// EnableWorkflowByID enables a workflow and sets the state of the workflow to "active".
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#enable-a-workflow
func (s *ActionsService) EnableWorkflowByID(ctx context.Context, owner, repo string, workflowID int64) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/workflows/%v/enable", owner, repo, workflowID)
	return s.enableWorkflow(ctx, u)
}

// EnableWorkflowByFileName enables a workflow and sets the state of the workflow to "active".
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#enable-a-workflow
func (s *ActionsService) EnableWorkflowByFileName(ctx context.Context, owner, repo, workflowFileName string) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/workflows/%v/enable", owner, repo, workflowFileName)
	return s.enableWorkflow(ctx, u)
}

func (s *ActionsService) enableWorkflow(ctx context.Context, url string) (*Response, error) {
	req, err := s.client.NewRequest("PUT", url, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// DisableWorkflowByID disables a workflow and sets the state of the workflow to "disabled_manually".
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#disable-a-workflow
func (s *ActionsService) DisableWorkflowByID(ctx context.Context, owner, repo string, workflowID int64) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/workflows/%v/disable", owner, repo, workflowID)
	return s.enableWorkflow(ctx, u)
}

// DisableWorkflowByFileName disables a workflow and sets the state of the workflow to "disabled_manually".
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#disable-a-workflow
func (s *ActionsService) DisableWorkflowByFileName(ctx context.Context, owner, repo, workflowFileName string) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/workflows/%v/disable", owner, repo, workflowFileName)
	return s.enableWorkflow(ctx, u)
}

func (s *ActionsService) disableWorkflow(ctx context.Context, url string) (*Response, error) {
	req, err := s.client.NewRequest("PUT", url, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
