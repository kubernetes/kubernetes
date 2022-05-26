//
// Copyright 2021, Igor Varavko
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
	"time"
)

// PipelinesService handles communication with the repositories related
// methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/pipelines.html
type PipelinesService struct {
	client *Client
}

// PipelineVariable represents a pipeline variable.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/pipelines.html
type PipelineVariable struct {
	Key          string `json:"key"`
	Value        string `json:"value"`
	VariableType string `json:"variable_type"`
}

// Pipeline represents a GitLab pipeline.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/pipelines.html
type Pipeline struct {
	ID             int             `json:"id"`
	Status         string          `json:"status"`
	Ref            string          `json:"ref"`
	SHA            string          `json:"sha"`
	BeforeSHA      string          `json:"before_sha"`
	Tag            bool            `json:"tag"`
	YamlErrors     string          `json:"yaml_errors"`
	User           *BasicUser      `json:"user"`
	UpdatedAt      *time.Time      `json:"updated_at"`
	CreatedAt      *time.Time      `json:"created_at"`
	StartedAt      *time.Time      `json:"started_at"`
	FinishedAt     *time.Time      `json:"finished_at"`
	CommittedAt    *time.Time      `json:"committed_at"`
	Duration       int             `json:"duration"`
	Coverage       string          `json:"coverage"`
	WebURL         string          `json:"web_url"`
	DetailedStatus *DetailedStatus `json:"detailed_status"`
}

// DetailedStatus contains detailed information about the status of a pipeline.
type DetailedStatus struct {
	Icon         string `json:"icon"`
	Text         string `json:"text"`
	Label        string `json:"label"`
	Group        string `json:"group"`
	Tooltip      string `json:"tooltip"`
	HasDetails   bool   `json:"has_details"`
	DetailsPath  string `json:"details_path"`
	Illustration struct {
		Image string `json:"image"`
	} `json:"illustration"`
	Favicon string `json:"favicon"`
}

func (p Pipeline) String() string {
	return Stringify(p)
}

// PipelineTestReport contains a detailed report of a test run.
type PipelineTestReport struct {
	TotalTime    float64              `json:"total_time"`
	TotalCount   int                  `json:"total_count"`
	SuccessCount int                  `json:"success_count"`
	FailedCount  int                  `json:"failed_count"`
	SkippedCount int                  `json:"skipped_count"`
	ErrorCount   int                  `json:"error_count"`
	TestSuites   []PipelineTestSuites `json:"test_suites"`
}

// PipelineTestSuites contains test suites results.
type PipelineTestSuites struct {
	Name         string              `json:"name"`
	TotalTime    float64             `json:"total_time"`
	TotalCount   int                 `json:"total_count"`
	SuccessCount int                 `json:"success_count"`
	FailedCount  int                 `json:"failed_count"`
	SkippedCount int                 `json:"skipped_count"`
	ErrorCount   int                 `json:"error_count"`
	TestCases    []PipelineTestCases `json:"test_cases"`
}

// PipelineTestCases contains test cases details.
type PipelineTestCases struct {
	Status         string         `json:"status"`
	Name           string         `json:"name"`
	Classname      string         `json:"classname"`
	File           string         `json:"file"`
	ExecutionTime  float64        `json:"execution_time"`
	SystemOutput   string         `json:"system_output"`
	StackTrace     string         `json:"stack_trace"`
	AttachmentURL  string         `json:"attachment_url"`
	RecentFailures RecentFailures `json:"recent_failures"`
}

// RecentFailures contains failures count for the project's default branch.
type RecentFailures struct {
	Count      int    `json:"count"`
	BaseBranch string `json:"base_branch"`
}

func (p PipelineTestReport) String() string {
	return Stringify(p)
}

// PipelineInfo shows the basic entities of a pipeline, mostly used as fields
// on other assets, like Commit.
type PipelineInfo struct {
	ID        int        `json:"id"`
	Status    string     `json:"status"`
	Ref       string     `json:"ref"`
	SHA       string     `json:"sha"`
	WebURL    string     `json:"web_url"`
	UpdatedAt *time.Time `json:"updated_at"`
	CreatedAt *time.Time `json:"created_at"`
}

func (p PipelineInfo) String() string {
	return Stringify(p)
}

// ListProjectPipelinesOptions represents the available ListProjectPipelines() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/pipelines.html#list-project-pipelines
type ListProjectPipelinesOptions struct {
	ListOptions
	Scope         *string          `url:"scope,omitempty" json:"scope,omitempty"`
	Status        *BuildStateValue `url:"status,omitempty" json:"status,omitempty"`
	Ref           *string          `url:"ref,omitempty" json:"ref,omitempty"`
	SHA           *string          `url:"sha,omitempty" json:"sha,omitempty"`
	YamlErrors    *bool            `url:"yaml_errors,omitempty" json:"yaml_errors,omitempty"`
	Name          *string          `url:"name,omitempty" json:"name,omitempty"`
	Username      *string          `url:"username,omitempty" json:"username,omitempty"`
	UpdatedAfter  *time.Time       `url:"updated_after,omitempty" json:"updated_after,omitempty"`
	UpdatedBefore *time.Time       `url:"updated_before,omitempty" json:"updated_before,omitempty"`
	OrderBy       *string          `url:"order_by,omitempty" json:"order_by,omitempty"`
	Sort          *string          `url:"sort,omitempty" json:"sort,omitempty"`
}

// ListProjectPipelines gets a list of project piplines.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/pipelines.html#list-project-pipelines
func (s *PipelinesService) ListProjectPipelines(pid interface{}, opt *ListProjectPipelinesOptions, options ...RequestOptionFunc) ([]*PipelineInfo, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipelines", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var p []*PipelineInfo
	resp, err := s.client.Do(req, &p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// GetPipeline gets a single project pipeline.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/pipelines.html#get-a-single-pipeline
func (s *PipelinesService) GetPipeline(pid interface{}, pipeline int, options ...RequestOptionFunc) (*Pipeline, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipelines/%d", pathEscape(project), pipeline)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(Pipeline)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// GetPipelineVariables gets the variables of a single project pipeline.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/pipelines.html#get-variables-of-a-pipeline
func (s *PipelinesService) GetPipelineVariables(pid interface{}, pipeline int, options ...RequestOptionFunc) ([]*PipelineVariable, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipelines/%d/variables", pathEscape(project), pipeline)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var p []*PipelineVariable
	resp, err := s.client.Do(req, &p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// GetPipelineTestReport gets the test report of a single project pipeline.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/pipelines.html#get-a-pipelines-test-report
func (s *PipelinesService) GetPipelineTestReport(pid interface{}, pipeline int) (*PipelineTestReport, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipelines/%d/test_report", pathEscape(project), pipeline)

	req, err := s.client.NewRequest("GET", u, nil, nil)
	if err != nil {
		return nil, nil, err
	}

	p := new(PipelineTestReport)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// CreatePipelineOptions represents the available CreatePipeline() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/pipelines.html#create-a-new-pipeline
type CreatePipelineOptions struct {
	Ref       *string             `url:"ref" json:"ref"`
	Variables []*PipelineVariable `url:"variables,omitempty" json:"variables,omitempty"`
}

// CreatePipeline creates a new project pipeline.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/pipelines.html#create-a-new-pipeline
func (s *PipelinesService) CreatePipeline(pid interface{}, opt *CreatePipelineOptions, options ...RequestOptionFunc) (*Pipeline, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipeline", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(Pipeline)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// RetryPipelineBuild retries failed builds in a pipeline
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipelines.html#retry-failed-builds-in-a-pipeline
func (s *PipelinesService) RetryPipelineBuild(pid interface{}, pipeline int, options ...RequestOptionFunc) (*Pipeline, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipelines/%d/retry", pathEscape(project), pipeline)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(Pipeline)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// CancelPipelineBuild cancels a pipeline builds
//
// GitLab API docs:
//https://docs.gitlab.com/ce/api/pipelines.html#cancel-a-pipelines-builds
func (s *PipelinesService) CancelPipelineBuild(pid interface{}, pipeline int, options ...RequestOptionFunc) (*Pipeline, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipelines/%d/cancel", pathEscape(project), pipeline)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(Pipeline)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// DeletePipeline deletes an existing pipeline.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipelines.html#delete-a-pipeline
func (s *PipelinesService) DeletePipeline(pid interface{}, pipeline int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/pipelines/%d", pathEscape(project), pipeline)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
