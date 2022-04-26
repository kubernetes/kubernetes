//
// Copyright 2021, Arkbriar
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
	"bytes"
	"fmt"
	"time"
)

// JobsService handles communication with the ci builds related methods
// of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/jobs.html
type JobsService struct {
	client *Client
}

// Job represents a ci build.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/jobs.html
type Job struct {
	Commit            *Commit    `json:"commit"`
	Coverage          float64    `json:"coverage"`
	AllowFailure      bool       `json:"allow_failure"`
	CreatedAt         *time.Time `json:"created_at"`
	StartedAt         *time.Time `json:"started_at"`
	FinishedAt        *time.Time `json:"finished_at"`
	Duration          float64    `json:"duration"`
	ArtifactsExpireAt *time.Time `json:"artifacts_expire_at"`
	ID                int        `json:"id"`
	Name              string     `json:"name"`
	Pipeline          struct {
		ID     int    `json:"id"`
		Ref    string `json:"ref"`
		Sha    string `json:"sha"`
		Status string `json:"status"`
	} `json:"pipeline"`
	Ref       string `json:"ref"`
	Artifacts []struct {
		FileType   string `json:"file_type"`
		Filename   string `json:"filename"`
		Size       int    `json:"size"`
		FileFormat string `json:"file_format"`
	} `json:"artifacts"`
	ArtifactsFile struct {
		Filename string `json:"filename"`
		Size     int    `json:"size"`
	} `json:"artifacts_file"`
	Runner struct {
		ID          int    `json:"id"`
		Description string `json:"description"`
		Active      bool   `json:"active"`
		IsShared    bool   `json:"is_shared"`
		Name        string `json:"name"`
	} `json:"runner"`
	Stage  string `json:"stage"`
	Status string `json:"status"`
	Tag    bool   `json:"tag"`
	WebURL string `json:"web_url"`
	User   *User  `json:"user"`
}

// Bridge represents a pipeline bridge.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/jobs.html#list-pipeline-bridges
type Bridge struct {
	Commit             *Commit       `json:"commit"`
	Coverage           float64       `json:"coverage"`
	AllowFailure       bool          `json:"allow_failure"`
	CreatedAt          *time.Time    `json:"created_at"`
	StartedAt          *time.Time    `json:"started_at"`
	FinishedAt         *time.Time    `json:"finished_at"`
	Duration           float64       `json:"duration"`
	ID                 int           `json:"id"`
	Name               string        `json:"name"`
	Pipeline           PipelineInfo  `json:"pipeline"`
	Ref                string        `json:"ref"`
	Stage              string        `json:"stage"`
	Status             string        `json:"status"`
	Tag                bool          `json:"tag"`
	WebURL             string        `json:"web_url"`
	User               *User         `json:"user"`
	DownstreamPipeline *PipelineInfo `json:"downstream_pipeline"`
}

// ListJobsOptions are options for two list apis
type ListJobsOptions struct {
	ListOptions
	Scope []BuildStateValue `url:"scope[],omitempty" json:"scope,omitempty"`
}

// ListProjectJobs gets a list of jobs in a project.
//
// The scope of jobs to show, one or array of: created, pending, running,
// failed, success, canceled, skipped; showing all jobs if none provided
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/jobs.html#list-project-jobs
func (s *JobsService) ListProjectJobs(pid interface{}, opts *ListJobsOptions, options ...RequestOptionFunc) ([]*Job, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/jobs", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opts, options)
	if err != nil {
		return nil, nil, err
	}

	var jobs []*Job
	resp, err := s.client.Do(req, &jobs)
	if err != nil {
		return nil, resp, err
	}

	return jobs, resp, err
}

// ListPipelineJobs gets a list of jobs for specific pipeline in a
// project. If the pipeline ID is not found, it will respond with 404.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/jobs.html#list-pipeline-jobs
func (s *JobsService) ListPipelineJobs(pid interface{}, pipelineID int, opts *ListJobsOptions, options ...RequestOptionFunc) ([]*Job, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipelines/%d/jobs", pathEscape(project), pipelineID)

	req, err := s.client.NewRequest("GET", u, opts, options)
	if err != nil {
		return nil, nil, err
	}

	var jobs []*Job
	resp, err := s.client.Do(req, &jobs)
	if err != nil {
		return nil, resp, err
	}

	return jobs, resp, err
}

// ListPipelineBridges gets a list of bridges for specific pipeline in a
// project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/jobs.html#list-pipeline-jobs
func (s *JobsService) ListPipelineBridges(pid interface{}, pipelineID int, opts *ListJobsOptions, options ...RequestOptionFunc) ([]*Bridge, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/pipelines/%d/bridges", pathEscape(project), pipelineID)

	req, err := s.client.NewRequest("GET", u, opts, options)
	if err != nil {
		return nil, nil, err
	}

	var bridges []*Bridge
	resp, err := s.client.Do(req, &bridges)
	if err != nil {
		return nil, resp, err
	}

	return bridges, resp, err
}

// GetJob gets a single job of a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/jobs.html#get-a-single-job
func (s *JobsService) GetJob(pid interface{}, jobID int, options ...RequestOptionFunc) (*Job, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/jobs/%d", pathEscape(project), jobID)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	job := new(Job)
	resp, err := s.client.Do(req, job)
	if err != nil {
		return nil, resp, err
	}

	return job, resp, err
}

// GetJobArtifacts get jobs artifacts of a project
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/job_artifacts.html#get-job-artifacts
func (s *JobsService) GetJobArtifacts(pid interface{}, jobID int, options ...RequestOptionFunc) (*bytes.Reader, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/jobs/%d/artifacts", pathEscape(project), jobID)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	artifactsBuf := new(bytes.Buffer)
	resp, err := s.client.Do(req, artifactsBuf)
	if err != nil {
		return nil, resp, err
	}

	return bytes.NewReader(artifactsBuf.Bytes()), resp, err
}

// DownloadArtifactsFileOptions represents the available DownloadArtifactsFile()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/job_artifacts.html#download-the-artifacts-archive
type DownloadArtifactsFileOptions struct {
	Job *string `url:"job" json:"job"`
}

// DownloadArtifactsFile download the artifacts file from the given
// reference name and job provided the job finished successfully.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/job_artifacts.html#download-the-artifacts-archive
func (s *JobsService) DownloadArtifactsFile(pid interface{}, refName string, opt *DownloadArtifactsFileOptions, options ...RequestOptionFunc) (*bytes.Reader, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/jobs/artifacts/%s/download", pathEscape(project), refName)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	artifactsBuf := new(bytes.Buffer)
	resp, err := s.client.Do(req, artifactsBuf)
	if err != nil {
		return nil, resp, err
	}

	return bytes.NewReader(artifactsBuf.Bytes()), resp, err
}

// DownloadSingleArtifactsFile download a file from the artifacts from the
// given reference name and job provided the job finished successfully.
// Only a single file is going to be extracted from the archive and streamed
// to a client.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/job_artifacts.html#download-a-single-artifact-file-by-job-id
func (s *JobsService) DownloadSingleArtifactsFile(pid interface{}, jobID int, artifactPath string, options ...RequestOptionFunc) (*bytes.Reader, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}

	u := fmt.Sprintf(
		"projects/%s/jobs/%d/artifacts/%s",
		pathEscape(project),
		jobID,
		artifactPath,
	)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	artifactBuf := new(bytes.Buffer)
	resp, err := s.client.Do(req, artifactBuf)
	if err != nil {
		return nil, resp, err
	}

	return bytes.NewReader(artifactBuf.Bytes()), resp, err
}

// GetTraceFile gets a trace of a specific job of a project
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/jobs.html#get-a-trace-file
func (s *JobsService) GetTraceFile(pid interface{}, jobID int, options ...RequestOptionFunc) (*bytes.Reader, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/jobs/%d/trace", pathEscape(project), jobID)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	traceBuf := new(bytes.Buffer)
	resp, err := s.client.Do(req, traceBuf)
	if err != nil {
		return nil, resp, err
	}

	return bytes.NewReader(traceBuf.Bytes()), resp, err
}

// CancelJob cancels a single job of a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/jobs.html#cancel-a-job
func (s *JobsService) CancelJob(pid interface{}, jobID int, options ...RequestOptionFunc) (*Job, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/jobs/%d/cancel", pathEscape(project), jobID)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	job := new(Job)
	resp, err := s.client.Do(req, job)
	if err != nil {
		return nil, resp, err
	}

	return job, resp, err
}

// RetryJob retries a single job of a project
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/jobs.html#retry-a-job
func (s *JobsService) RetryJob(pid interface{}, jobID int, options ...RequestOptionFunc) (*Job, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/jobs/%d/retry", pathEscape(project), jobID)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	job := new(Job)
	resp, err := s.client.Do(req, job)
	if err != nil {
		return nil, resp, err
	}

	return job, resp, err
}

// EraseJob erases a single job of a project, removes a job
// artifacts and a job trace.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/jobs.html#erase-a-job
func (s *JobsService) EraseJob(pid interface{}, jobID int, options ...RequestOptionFunc) (*Job, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/jobs/%d/erase", pathEscape(project), jobID)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	job := new(Job)
	resp, err := s.client.Do(req, job)
	if err != nil {
		return nil, resp, err
	}

	return job, resp, err
}

// KeepArtifacts prevents artifacts from being deleted when
// expiration is set.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/job_artifacts.html#keep-artifacts
func (s *JobsService) KeepArtifacts(pid interface{}, jobID int, options ...RequestOptionFunc) (*Job, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/jobs/%d/artifacts/keep", pathEscape(project), jobID)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	job := new(Job)
	resp, err := s.client.Do(req, job)
	if err != nil {
		return nil, resp, err
	}

	return job, resp, err
}

// PlayJob triggers a manual action to start a job.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/jobs.html#play-a-job
func (s *JobsService) PlayJob(pid interface{}, jobID int, options ...RequestOptionFunc) (*Job, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/jobs/%d/play", pathEscape(project), jobID)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	job := new(Job)
	resp, err := s.client.Do(req, job)
	if err != nil {
		return nil, resp, err
	}

	return job, resp, err
}

// DeleteArtifacts delete artifacts of a job
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/job_artifacts.html#delete-artifacts
func (s *JobsService) DeleteArtifacts(pid interface{}, jobID int, options ...RequestOptionFunc) (*Job, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/jobs/%d/artifacts", pathEscape(project), jobID)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	job := new(Job)
	resp, err := s.client.Do(req, job)
	if err != nil {
		return nil, resp, err
	}

	return job, resp, err
}
