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
	"fmt"

	"golang.org/x/net/context"
	bq "google.golang.org/api/bigquery/v2"
)

type queryOption interface {
	customizeQuery(conf *bq.JobConfigurationQuery, projectID string)
}

// DisableQueryCache returns an Option that prevents results being fetched from the query cache.
// If this Option is not used, results are fetched from the cache if they are available.
// The query cache is a best-effort cache that is flushed whenever tables in the query are modified.
// Cached results are only available when TableID is unspecified in the query's destination Table.
// For more information, see https://cloud.google.com/bigquery/querying-data#querycaching
func DisableQueryCache() Option { return disableQueryCache{} }

type disableQueryCache struct{}

func (opt disableQueryCache) implementsOption() {}

func (opt disableQueryCache) customizeQuery(conf *bq.JobConfigurationQuery, projectID string) {
	f := false
	conf.UseQueryCache = &f
}

// DisableFlattenedResults returns an Option that prevents results being flattened.
// If this Option is not used, results from nested and repeated fields are flattened.
// DisableFlattenedResults implies AllowLargeResults
// For more information, see https://cloud.google.com/bigquery/docs/data#nested
func DisableFlattenedResults() Option { return disableFlattenedResults{} }

type disableFlattenedResults struct{}

func (opt disableFlattenedResults) implementsOption() {}

func (opt disableFlattenedResults) customizeQuery(conf *bq.JobConfigurationQuery, projectID string) {
	f := false
	conf.FlattenResults = &f
	// DisableFlattenedResults implies AllowLargeResults
	allowLargeResults{}.customizeQuery(conf, projectID)
}

// AllowLargeResults returns an Option that allows the query to produce arbitrarily large result tables.
// The destination must be a table.
// When using this option, queries will take longer to execute, even if the result set is small.
// For additional limitations, see https://cloud.google.com/bigquery/querying-data#largequeryresults
func AllowLargeResults() Option { return allowLargeResults{} }

type allowLargeResults struct{}

func (opt allowLargeResults) implementsOption() {}

func (opt allowLargeResults) customizeQuery(conf *bq.JobConfigurationQuery, projectID string) {
	conf.AllowLargeResults = true
}

// JobPriority returns an Option that causes a query to be scheduled with the specified priority.
// The default priority is InteractivePriority.
// For more information, see https://cloud.google.com/bigquery/querying-data#batchqueries
func JobPriority(priority string) Option { return jobPriority(priority) }

type jobPriority string

func (opt jobPriority) implementsOption() {}

func (opt jobPriority) customizeQuery(conf *bq.JobConfigurationQuery, projectID string) {
	conf.Priority = string(opt)
}

const (
	BatchPriority       = "BATCH"
	InteractivePriority = "INTERACTIVE"
)

func (c *Client) query(ctx context.Context, dst *Table, src *Query, options []Option) (*Job, error) {
	job, options := initJobProto(c.projectID, options)
	payload := &bq.JobConfigurationQuery{}

	dst.customizeQueryDst(payload, c.projectID)
	src.customizeQuerySrc(payload, c.projectID)

	for _, opt := range options {
		o, ok := opt.(queryOption)
		if !ok {
			return nil, fmt.Errorf("option (%#v) not applicable to dst/src pair: dst: %T ; src: %T", opt, dst, src)
		}
		o.customizeQuery(payload, c.projectID)
	}

	job.Configuration = &bq.JobConfiguration{
		Query: payload,
	}
	j, err := c.service.insertJob(ctx, job, c.projectID)
	if err != nil {
		return nil, err
	}
	j.isQuery = true
	return j, nil
}
