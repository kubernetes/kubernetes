// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bigquery

import (
	"errors"

	"golang.org/x/net/context"
	bq "google.golang.org/api/bigquery/v2"
)

// A Job represents an operation which has been submitted to BigQuery for processing.
type Job struct {
	service   service
	projectID string
	jobID     string

	isQuery bool
}

// JobFromID creates a Job which refers to an existing BigQuery job. The job
// need not have been created by this package. For example, the job may have
// been created in the BigQuery console.
func (c *Client) JobFromID(ctx context.Context, id string) (*Job, error) {
	jobType, err := c.service.getJobType(ctx, c.projectID, id)
	if err != nil {
		return nil, err
	}

	return &Job{
		service:   c.service,
		projectID: c.projectID,
		jobID:     id,
		isQuery:   jobType == queryJobType,
	}, nil
}

func (j *Job) ID() string {
	return j.jobID
}

// State is one of a sequence of states that a Job progresses through as it is processed.
type State int

const (
	Pending State = iota
	Running
	Done
)

// JobStatus contains the current State of a job, and errors encountered while processing that job.
type JobStatus struct {
	State State

	err error

	// All errors encountered during the running of the job.
	// Not all Errors are fatal, so errors here do not necessarily mean that the job has completed or was unsuccessful.
	Errors []*Error
}

// jobOption is an Option which modifies a bq.Job proto.
// This is used for configuring values that apply to all operations, such as setting a jobReference.
type jobOption interface {
	customizeJob(job *bq.Job, projectID string)
}

type jobID string

// JobID returns an Option that sets the job ID of a BigQuery job.
// If this Option is not used, a job ID is generated automatically.
func JobID(ID string) Option {
	return jobID(ID)
}

func (opt jobID) implementsOption() {}

func (opt jobID) customizeJob(job *bq.Job, projectID string) {
	job.JobReference = &bq.JobReference{
		JobId:     string(opt),
		ProjectId: projectID,
	}
}

// Done reports whether the job has completed.
// After Done returns true, the Err method will return an error if the job completed unsuccesfully.
func (s *JobStatus) Done() bool {
	return s.State == Done
}

// Err returns the error that caused the job to complete unsuccesfully (if any).
func (s *JobStatus) Err() error {
	return s.err
}

// Status returns the current status of the job.  It fails if the Status could not be determined.
func (j *Job) Status(ctx context.Context) (*JobStatus, error) {
	return j.service.jobStatus(ctx, j.projectID, j.jobID)
}

func (j *Job) implementsReadSource() {}

func (j *Job) customizeReadQuery(cursor *readQueryConf) error {
	// There are mulitple kinds of jobs, but only a query job is suitable for reading.
	if !j.isQuery {
		return errors.New("Cannot read from a non-query job")
	}

	cursor.projectID = j.projectID
	cursor.jobID = j.jobID
	return nil
}
