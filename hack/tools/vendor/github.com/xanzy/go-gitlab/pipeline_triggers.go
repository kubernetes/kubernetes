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

// PipelineTriggersService handles Project pipeline triggers.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_triggers.html
type PipelineTriggersService struct {
	client *Client
}

// PipelineTrigger represents a project pipeline trigger.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_triggers.html#pipeline-triggers
type PipelineTrigger struct {
	ID          int        `json:"id"`
	Description string     `json:"description"`
	CreatedAt   *time.Time `json:"created_at"`
	DeletedAt   *time.Time `json:"deleted_at"`
	LastUsed    *time.Time `json:"last_used"`
	Token       string     `json:"token"`
	UpdatedAt   *time.Time `json:"updated_at"`
	Owner       *User      `json:"owner"`
}

// ListPipelineTriggersOptions represents the available ListPipelineTriggers() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_triggers.html#list-project-triggers
type ListPipelineTriggersOptions ListOptions

// ListPipelineTriggers gets a list of project triggers.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_triggers.html#list-project-triggers
func (s *PipelineTriggersService) ListPipelineTriggers(pid interface{}, opt *ListPipelineTriggersOptions, options ...RequestOptionFunc) ([]*PipelineTrigger, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/triggers", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var pt []*PipelineTrigger
	resp, err := s.client.Do(req, &pt)
	if err != nil {
		return nil, resp, err
	}

	return pt, resp, err
}

// GetPipelineTrigger gets a specific pipeline trigger for a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_triggers.html#get-trigger-details
func (s *PipelineTriggersService) GetPipelineTrigger(pid interface{}, trigger int, options ...RequestOptionFunc) (*PipelineTrigger, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/triggers/%d", pathEscape(project), trigger)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	pt := new(PipelineTrigger)
	resp, err := s.client.Do(req, pt)
	if err != nil {
		return nil, resp, err
	}

	return pt, resp, err
}

// AddPipelineTriggerOptions represents the available AddPipelineTrigger() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_triggers.html#create-a-project-trigger
type AddPipelineTriggerOptions struct {
	Description *string `url:"description,omitempty" json:"description,omitempty"`
}

// AddPipelineTrigger adds a pipeline trigger to a specified project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_triggers.html#create-a-project-trigger
func (s *PipelineTriggersService) AddPipelineTrigger(pid interface{}, opt *AddPipelineTriggerOptions, options ...RequestOptionFunc) (*PipelineTrigger, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/triggers", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	pt := new(PipelineTrigger)
	resp, err := s.client.Do(req, pt)
	if err != nil {
		return nil, resp, err
	}

	return pt, resp, err
}

// EditPipelineTriggerOptions represents the available EditPipelineTrigger() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_triggers.html#update-a-project-trigger
type EditPipelineTriggerOptions struct {
	Description *string `url:"description,omitempty" json:"description,omitempty"`
}

// EditPipelineTrigger edits a trigger for a specified project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_triggers.html#update-a-project-trigger
func (s *PipelineTriggersService) EditPipelineTrigger(pid interface{}, trigger int, opt *EditPipelineTriggerOptions, options ...RequestOptionFunc) (*PipelineTrigger, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/triggers/%d", pathEscape(project), trigger)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	pt := new(PipelineTrigger)
	resp, err := s.client.Do(req, pt)
	if err != nil {
		return nil, resp, err
	}

	return pt, resp, err
}

// TakeOwnershipOfPipelineTrigger sets the owner of the specified
// pipeline trigger to the user issuing the request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_triggers.html#take-ownership-of-a-project-trigger
func (s *PipelineTriggersService) TakeOwnershipOfPipelineTrigger(pid interface{}, trigger int, options ...RequestOptionFunc) (*PipelineTrigger, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/triggers/%d/take_ownership", pathEscape(project), trigger)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	pt := new(PipelineTrigger)
	resp, err := s.client.Do(req, pt)
	if err != nil {
		return nil, resp, err
	}

	return pt, resp, err
}

// DeletePipelineTrigger removes a trigger from a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/pipeline_triggers.html#remove-a-project-trigger
func (s *PipelineTriggersService) DeletePipelineTrigger(pid interface{}, trigger int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/triggers/%d", pathEscape(project), trigger)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// RunPipelineTriggerOptions represents the available RunPipelineTrigger() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/ci/triggers/README.html#triggering-a-pipeline
type RunPipelineTriggerOptions struct {
	Ref       *string           `url:"ref" json:"ref"`
	Token     *string           `url:"token" json:"token"`
	Variables map[string]string `url:"variables,omitempty" json:"variables,omitempty"`
}

// RunPipelineTrigger starts a trigger from a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/ci/triggers/README.html#triggering-a-pipeline
func (s *PipelineTriggersService) RunPipelineTrigger(pid interface{}, opt *RunPipelineTriggerOptions, options ...RequestOptionFunc) (*Pipeline, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/trigger/pipeline", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	pt := new(Pipeline)
	resp, err := s.client.Do(req, pt)
	if err != nil {
		return nil, resp, err
	}

	return pt, resp, err
}
