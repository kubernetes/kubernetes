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

import bq "google.golang.org/api/bigquery/v2"

// Query represents a query to be executed. Use Client.Query to create a query.
type Query struct {
	// The query to execute. See https://cloud.google.com/bigquery/query-reference for details.
	Q string

	// DefaultProjectID and DefaultDatasetID specify the dataset to use for unqualified table names in the query.
	// If DefaultProjectID is set, DefaultDatasetID must also be set.
	DefaultProjectID string
	DefaultDatasetID string

	client *Client
}

func (q *Query) implementsSource() {}

func (q *Query) implementsReadSource() {}

func (q *Query) customizeQuerySrc(conf *bq.JobConfigurationQuery) {
	conf.Query = q.Q
	if q.DefaultProjectID != "" || q.DefaultDatasetID != "" {
		conf.DefaultDataset = &bq.DatasetReference{
			DatasetId: q.DefaultDatasetID,
			ProjectId: q.DefaultProjectID,
		}
	}
}
