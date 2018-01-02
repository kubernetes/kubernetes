// Copyright 2016 Google Inc. All Rights Reserved.
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

package logging

import (
	"fmt"

	vkit "cloud.google.com/go/logging/apiv2"
	gax "github.com/googleapis/gax-go"
	"golang.org/x/net/context"
	"google.golang.org/api/iterator"
	logpb "google.golang.org/genproto/googleapis/logging/v2"
)

// Metric describes a logs-based metric. The value of the metric is the
// number of log entries that match a logs filter.
//
// Metrics are a feature of Stackdriver Monitoring.
// See https://cloud.google.com/monitoring/api/v3/metrics for more about them.
type Metric struct {
	// ID is a client-assigned metric identifier. Example:
	// "severe_errors".  Metric identifiers are limited to 1000
	// characters and can include only the following characters: A-Z,
	// a-z, 0-9, and the special characters _-.,+!*',()%/\.  The
	// forward-slash character (/) denotes a hierarchy of name pieces,
	// and it cannot be the first character of the name.
	ID string

	// Description describes this metric. It is used in documentation.
	Description string

	// Filter is an advanced logs filter (see
	// https://cloud.google.com/logging/docs/view/advanced_filters).
	// Example: "logName:syslog AND severity>=ERROR".
	Filter string
}

// CreateMetric creates a logs-based metric.
func (c *Client) CreateMetric(ctx context.Context, m *Metric) error {
	_, err := c.mClient.CreateLogMetric(ctx, &logpb.CreateLogMetricRequest{
		Parent: c.parent(),
		Metric: toLogMetric(m),
	})
	return err
}

// DeleteMetric deletes a log-based metric.
// The provided metric ID is the metric identifier. For example, "severe_errors".
func (c *Client) DeleteMetric(ctx context.Context, metricID string) error {
	return c.mClient.DeleteLogMetric(ctx, &logpb.DeleteLogMetricRequest{
		MetricName: c.metricPath(metricID),
	})
}

// Metric gets a logs-based metric.
// The provided metric ID is the metric identifier. For example, "severe_errors".
// Requires ReadScope or AdminScope.
func (c *Client) Metric(ctx context.Context, metricID string) (*Metric, error) {
	lm, err := c.mClient.GetLogMetric(ctx, &logpb.GetLogMetricRequest{
		MetricName: c.metricPath(metricID),
	})
	if err != nil {
		return nil, err
	}
	return fromLogMetric(lm), nil
}

// UpdateMetric creates a logs-based metric if it does not exist, or updates an
// existing one.
func (c *Client) UpdateMetric(ctx context.Context, m *Metric) error {
	_, err := c.mClient.UpdateLogMetric(ctx, &logpb.UpdateLogMetricRequest{
		MetricName: c.metricPath(m.ID),
		Metric:     toLogMetric(m),
	})
	return err
}

func (c *Client) metricPath(metricID string) string {
	return fmt.Sprintf("%s/metrics/%s", c.parent(), metricID)
}

// Metrics returns a MetricIterator for iterating over all Metrics in the Client's project.
// Requires ReadScope or AdminScope.
func (c *Client) Metrics(ctx context.Context) *MetricIterator {
	it := &MetricIterator{
		ctx:    ctx,
		client: c.mClient,
		req:    &logpb.ListLogMetricsRequest{Parent: c.parent()},
	}
	it.pageInfo, it.nextFunc = iterator.NewPageInfo(
		it.fetch,
		func() int { return len(it.items) },
		func() interface{} { b := it.items; it.items = nil; return b })
	return it
}

// A MetricIterator iterates over Metrics.
type MetricIterator struct {
	ctx      context.Context
	client   *vkit.MetricsClient
	pageInfo *iterator.PageInfo
	nextFunc func() error
	req      *logpb.ListLogMetricsRequest
	items    []*Metric
}

// PageInfo supports pagination. See the google.golang.org/api/iterator package for details.
func (it *MetricIterator) PageInfo() *iterator.PageInfo { return it.pageInfo }

// Next returns the next result. Its second return value is Done if there are
// no more results. Once Next returns Done, all subsequent calls will return
// Done.
func (it *MetricIterator) Next() (*Metric, error) {
	if err := it.nextFunc(); err != nil {
		return nil, err
	}
	item := it.items[0]
	it.items = it.items[1:]
	return item, nil
}

func (it *MetricIterator) fetch(pageSize int, pageToken string) (string, error) {
	// TODO(jba): Do this a nicer way if the generated code supports one.
	// TODO(jba): If the above TODO can't be done, find a way to pass metadata in the call.
	client := logpb.NewMetricsServiceV2Client(it.client.Connection())
	var res *logpb.ListLogMetricsResponse
	err := gax.Invoke(it.ctx, func(ctx context.Context) error {
		it.req.PageSize = trunc32(pageSize)
		it.req.PageToken = pageToken
		var err error
		res, err = client.ListLogMetrics(ctx, it.req)
		return err
	}, it.client.CallOptions.ListLogMetrics...)
	if err != nil {
		return "", err
	}
	for _, sp := range res.Metrics {
		it.items = append(it.items, fromLogMetric(sp))
	}
	return res.NextPageToken, nil
}

func toLogMetric(m *Metric) *logpb.LogMetric {
	return &logpb.LogMetric{
		Name:        m.ID,
		Description: m.Description,
		Filter:      m.Filter,
	}
}

func fromLogMetric(lm *logpb.LogMetric) *Metric {
	return &Metric{
		ID:          lm.Name,
		Description: lm.Description,
		Filter:      lm.Filter,
	}
}
