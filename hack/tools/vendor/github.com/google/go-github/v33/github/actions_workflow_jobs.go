// Copyright 2020 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
)

// TaskStep represents a single task step from a sequence of tasks of a job.
type TaskStep struct {
	Name        *string    `json:"name,omitempty"`
	Status      *string    `json:"status,omitempty"`
	Conclusion  *string    `json:"conclusion,omitempty"`
	Number      *int64     `json:"number,omitempty"`
	StartedAt   *Timestamp `json:"started_at,omitempty"`
	CompletedAt *Timestamp `json:"completed_at,omitempty"`
}

// WorkflowJob represents a repository action workflow job.
type WorkflowJob struct {
	ID          *int64      `json:"id,omitempty"`
	RunID       *int64      `json:"run_id,omitempty"`
	RunURL      *string     `json:"run_url,omitempty"`
	NodeID      *string     `json:"node_id,omitempty"`
	HeadSHA     *string     `json:"head_sha,omitempty"`
	URL         *string     `json:"url,omitempty"`
	HTMLURL     *string     `json:"html_url,omitempty"`
	Status      *string     `json:"status,omitempty"`
	Conclusion  *string     `json:"conclusion,omitempty"`
	StartedAt   *Timestamp  `json:"started_at,omitempty"`
	CompletedAt *Timestamp  `json:"completed_at,omitempty"`
	Name        *string     `json:"name,omitempty"`
	Steps       []*TaskStep `json:"steps,omitempty"`
	CheckRunURL *string     `json:"check_run_url,omitempty"`
}

// Jobs represents a slice of repository action workflow job.
type Jobs struct {
	TotalCount *int           `json:"total_count,omitempty"`
	Jobs       []*WorkflowJob `json:"jobs,omitempty"`
}

// ListWorkflowJobsOptions specifies optional parameters to ListWorkflowJobs.
type ListWorkflowJobsOptions struct {
	// Filter specifies how jobs should be filtered by their completed_at timestamp.
	// Possible values are:
	//     latest - Returns jobs from the most recent execution of the workflow run
	//     all - Returns all jobs for a workflow run, including from old executions of the workflow run
	//
	// Default value is "latest".
	Filter string `url:"filter,omitempty"`
	ListOptions
}

// ListWorkflowJobs lists all jobs for a workflow run.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#list-jobs-for-a-workflow-run
func (s *ActionsService) ListWorkflowJobs(ctx context.Context, owner, repo string, runID int64, opts *ListWorkflowJobsOptions) (*Jobs, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/actions/runs/%v/jobs", owner, repo, runID)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	jobs := new(Jobs)
	resp, err := s.client.Do(ctx, req, &jobs)
	if err != nil {
		return nil, resp, err
	}

	return jobs, resp, nil
}

// GetWorkflowJobByID gets a specific job in a workflow run by ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#get-a-job-for-a-workflow-run
func (s *ActionsService) GetWorkflowJobByID(ctx context.Context, owner, repo string, jobID int64) (*WorkflowJob, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/jobs/%v", owner, repo, jobID)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	job := new(WorkflowJob)
	resp, err := s.client.Do(ctx, req, job)
	if err != nil {
		return nil, resp, err
	}

	return job, resp, nil
}

// GetWorkflowJobLogs gets a redirect URL to download a plain text file of logs for a workflow job.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#download-job-logs-for-a-workflow-run
func (s *ActionsService) GetWorkflowJobLogs(ctx context.Context, owner, repo string, jobID int64, followRedirects bool) (*url.URL, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/jobs/%v/logs", owner, repo, jobID)

	resp, err := s.getWorkflowLogsFromURL(ctx, u, followRedirects)
	if err != nil {
		return nil, nil, err
	}

	if resp.StatusCode != http.StatusFound {
		return nil, newResponse(resp), fmt.Errorf("unexpected status code: %s", resp.Status)
	}
	parsedURL, err := url.Parse(resp.Header.Get("Location"))
	return parsedURL, newResponse(resp), err
}

func (s *ActionsService) getWorkflowLogsFromURL(ctx context.Context, u string, followRedirects bool) (*http.Response, error) {
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, err
	}

	var resp *http.Response
	// Use http.DefaultTransport if no custom Transport is configured
	req = withContext(ctx, req)
	if s.client.client.Transport == nil {
		resp, err = http.DefaultTransport.RoundTrip(req)
	} else {
		resp, err = s.client.client.Transport.RoundTrip(req)
	}
	if err != nil {
		return nil, err
	}
	resp.Body.Close()

	// If redirect response is returned, follow it
	if followRedirects && resp.StatusCode == http.StatusMovedPermanently {
		u = resp.Header.Get("Location")
		resp, err = s.getWorkflowLogsFromURL(ctx, u, false)
	}
	return resp, err

}
