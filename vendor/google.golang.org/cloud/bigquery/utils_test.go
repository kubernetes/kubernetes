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
	"golang.org/x/net/context"
	bq "google.golang.org/api/bigquery/v2"
)

var defaultTable = &Table{
	ProjectID: "project-id",
	DatasetID: "dataset-id",
	TableID:   "table-id",
}

var defaultGCS = &GCSReference{
	uris: []string{"uri"},
}

var defaultQuery = &Query{
	Q:                "query string",
	DefaultProjectID: "def-project-id",
	DefaultDatasetID: "def-dataset-id",
}

type testService struct {
	*bq.Job

	service
}

func (s *testService) insertJob(ctx context.Context, job *bq.Job, projectID string) (*Job, error) {
	s.Job = job
	return &Job{}, nil
}

func (s *testService) jobStatus(ctx context.Context, projectID, jobID string) (*JobStatus, error) {
	return &JobStatus{State: Done}, nil
}
