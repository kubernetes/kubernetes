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
	"time"
)

// PipelineSchedulesService handles communication with the pipeline
// schedules related methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/pipeline_schedules.html
type PipelineSchedulesService struct {
	client *Client
}

// PipelineSchedule represents a pipeline schedule.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html
type PipelineSchedule struct {
	ID           int        `json:"id"`
	Description  string     `json:"description"`
	Ref          string     `json:"ref"`
	Cron         string     `json:"cron"`
	CronTimezone string     `json:"cron_timezone"`
	NextRunAt    *time.Time `json:"next_run_at"`
	Active       bool       `json:"active"`
	CreatedAt    *time.Time `json:"created_at"`
	UpdatedAt    *time.Time `json:"updated_at"`
	Owner        *User      `json:"owner"`
	LastPipeline struct {
		ID     int    `json:"id"`
		SHA    string `json:"sha"`
		Ref    string `json:"ref"`
		Status string `json:"status"`
	} `json:"last_pipeline"`
	Variables []*PipelineVariable `json:"variables"`
}

// ListPipelineSchedulesOptions represents the available ListPipelineTriggers() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_triggers.html#list-project-triggers
type ListPipelineSchedulesOptions ListOptions

// ListPipelineSchedules gets a list of project triggers.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html
func (s *PipelineSchedulesService) ListPipelineSchedules(pid interface{}, opt *ListPipelineSchedulesOptions, options ...RequestOptionFunc) ([]*PipelineSchedule, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipeline_schedules", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var ps []*PipelineSchedule
	resp, err := s.client.Do(req, &ps)
	if err != nil {
		return nil, resp, err
	}

	return ps, resp, err
}

// GetPipelineSchedule gets a pipeline schedule.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html
func (s *PipelineSchedulesService) GetPipelineSchedule(pid interface{}, schedule int, options ...RequestOptionFunc) (*PipelineSchedule, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipeline_schedules/%d", pathEscape(project), schedule)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(PipelineSchedule)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// CreatePipelineScheduleOptions represents the available
// CreatePipelineSchedule() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html#create-a-new-pipeline-schedule
type CreatePipelineScheduleOptions struct {
	Description  *string `url:"description" json:"description"`
	Ref          *string `url:"ref" json:"ref"`
	Cron         *string `url:"cron" json:"cron"`
	CronTimezone *string `url:"cron_timezone,omitempty" json:"cron_timezone,omitempty"`
	Active       *bool   `url:"active,omitempty" json:"active,omitempty"`
}

// CreatePipelineSchedule creates a pipeline schedule.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html#create-a-new-pipeline-schedule
func (s *PipelineSchedulesService) CreatePipelineSchedule(pid interface{}, opt *CreatePipelineScheduleOptions, options ...RequestOptionFunc) (*PipelineSchedule, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipeline_schedules", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(PipelineSchedule)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// EditPipelineScheduleOptions represents the available
// EditPipelineSchedule() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html#create-a-new-pipeline-schedule
type EditPipelineScheduleOptions struct {
	Description  *string `url:"description,omitempty" json:"description,omitempty"`
	Ref          *string `url:"ref,omitempty" json:"ref,omitempty"`
	Cron         *string `url:"cron,omitempty" json:"cron,omitempty"`
	CronTimezone *string `url:"cron_timezone,omitempty" json:"cron_timezone,omitempty"`
	Active       *bool   `url:"active,omitempty" json:"active,omitempty"`
}

// EditPipelineSchedule edits a pipeline schedule.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html#edit-a-pipeline-schedule
func (s *PipelineSchedulesService) EditPipelineSchedule(pid interface{}, schedule int, opt *EditPipelineScheduleOptions, options ...RequestOptionFunc) (*PipelineSchedule, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipeline_schedules/%d", pathEscape(project), schedule)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(PipelineSchedule)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// TakeOwnershipOfPipelineSchedule sets the owner of the specified
// pipeline schedule to the user issuing the request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html#take-ownership-of-a-pipeline-schedule
func (s *PipelineSchedulesService) TakeOwnershipOfPipelineSchedule(pid interface{}, schedule int, options ...RequestOptionFunc) (*PipelineSchedule, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipeline_schedules/%d/take_ownership", pathEscape(project), schedule)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(PipelineSchedule)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// DeletePipelineSchedule deletes a pipeline schedule.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html#delete-a-pipeline-schedule
func (s *PipelineSchedulesService) DeletePipelineSchedule(pid interface{}, schedule int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/pipeline_schedules/%d", pathEscape(project), schedule)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// RunPipelineSchedule triggers a new scheduled pipeline to run immediately.
//
// Gitlab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html#run-a-scheduled-pipeline-immediately
func (s *PipelineSchedulesService) RunPipelineSchedule(pid interface{}, schedule int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/pipeline_schedules/%d/play", pathEscape(project), schedule)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// CreatePipelineScheduleVariableOptions represents the available
// CreatePipelineScheduleVariable() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html#create-a-new-pipeline-schedule
type CreatePipelineScheduleVariableOptions struct {
	Key          *string `url:"key" json:"key"`
	Value        *string `url:"value" json:"value"`
	VariableType *string `url:"variable_type,omitempty" json:"variable_type,omitempty"`
}

// CreatePipelineScheduleVariable creates a pipeline schedule variable.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html#create-a-new-pipeline-schedule
func (s *PipelineSchedulesService) CreatePipelineScheduleVariable(pid interface{}, schedule int, opt *CreatePipelineScheduleVariableOptions, options ...RequestOptionFunc) (*PipelineVariable, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipeline_schedules/%d/variables", pathEscape(project), schedule)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(PipelineVariable)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// EditPipelineScheduleVariableOptions represents the available
// EditPipelineScheduleVariable() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html#edit-a-pipeline-schedule-variable
type EditPipelineScheduleVariableOptions struct {
	Value        *string `url:"value" json:"value"`
	VariableType *string `url:"variable_type,omitempty" json:"variable_type,omitempty"`
}

// EditPipelineScheduleVariable creates a pipeline schedule variable.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html#edit-a-pipeline-schedule-variable
func (s *PipelineSchedulesService) EditPipelineScheduleVariable(pid interface{}, schedule int, key string, opt *EditPipelineScheduleVariableOptions, options ...RequestOptionFunc) (*PipelineVariable, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipeline_schedules/%d/variables/%s", pathEscape(project), schedule, key)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(PipelineVariable)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// DeletePipelineScheduleVariable creates a pipeline schedule variable.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_schedules.html#delete-a-pipeline-schedule-variable
func (s *PipelineSchedulesService) DeletePipelineScheduleVariable(pid interface{}, schedule int, key string, options ...RequestOptionFunc) (*PipelineVariable, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipeline_schedules/%d/variables/%s", pathEscape(project), schedule, key)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(PipelineVariable)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}
