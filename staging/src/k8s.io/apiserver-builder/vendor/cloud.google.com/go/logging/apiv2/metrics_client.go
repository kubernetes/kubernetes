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

// AUTO-GENERATED CODE. DO NOT EDIT.

package logging

import (
	"fmt"
	"math"
	"runtime"
	"time"

	gax "github.com/googleapis/gax-go"
	"golang.org/x/net/context"
	"google.golang.org/api/option"
	"google.golang.org/api/transport"
	loggingpb "google.golang.org/genproto/googleapis/logging/v2"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
)

var (
	metricsParentPathTemplate = gax.MustCompilePathTemplate("projects/{project}")
	metricsMetricPathTemplate = gax.MustCompilePathTemplate("projects/{project}/metrics/{metric}")
)

// MetricsCallOptions contains the retry settings for each method of this client.
type MetricsCallOptions struct {
	ListLogMetrics  []gax.CallOption
	GetLogMetric    []gax.CallOption
	CreateLogMetric []gax.CallOption
	UpdateLogMetric []gax.CallOption
	DeleteLogMetric []gax.CallOption
}

func defaultMetricsClientOptions() []option.ClientOption {
	return []option.ClientOption{
		option.WithEndpoint("logging.googleapis.com:443"),
		option.WithScopes(
			"https://www.googleapis.com/auth/cloud-platform",
			"https://www.googleapis.com/auth/cloud-platform.read-only",
			"https://www.googleapis.com/auth/logging.admin",
			"https://www.googleapis.com/auth/logging.read",
			"https://www.googleapis.com/auth/logging.write",
		),
	}
}

func defaultMetricsCallOptions() *MetricsCallOptions {
	retry := map[[2]string][]gax.CallOption{
		{"default", "idempotent"}: {
			gax.WithRetry(func() gax.Retryer {
				return gax.OnCodes([]codes.Code{
					codes.DeadlineExceeded,
					codes.Unavailable,
				}, gax.Backoff{
					Initial:    100 * time.Millisecond,
					Max:        1000 * time.Millisecond,
					Multiplier: 1.2,
				})
			}),
		},
	}

	return &MetricsCallOptions{
		ListLogMetrics:  retry[[2]string{"default", "idempotent"}],
		GetLogMetric:    retry[[2]string{"default", "idempotent"}],
		CreateLogMetric: retry[[2]string{"default", "non_idempotent"}],
		UpdateLogMetric: retry[[2]string{"default", "non_idempotent"}],
		DeleteLogMetric: retry[[2]string{"default", "idempotent"}],
	}
}

// MetricsClient is a client for interacting with MetricsServiceV2.
type MetricsClient struct {
	// The connection to the service.
	conn *grpc.ClientConn

	// The gRPC API client.
	client loggingpb.MetricsServiceV2Client

	// The call options for this service.
	CallOptions *MetricsCallOptions

	// The metadata to be sent with each request.
	metadata map[string][]string
}

// NewMetricsClient creates a new metrics service client.
//
// Service for configuring logs-based metrics.
func NewMetricsClient(ctx context.Context, opts ...option.ClientOption) (*MetricsClient, error) {
	conn, err := transport.DialGRPC(ctx, append(defaultMetricsClientOptions(), opts...)...)
	if err != nil {
		return nil, err
	}
	c := &MetricsClient{
		conn:        conn,
		client:      loggingpb.NewMetricsServiceV2Client(conn),
		CallOptions: defaultMetricsCallOptions(),
	}
	c.SetGoogleClientInfo("gax", gax.Version)
	return c, nil
}

// Connection returns the client's connection to the API service.
func (c *MetricsClient) Connection() *grpc.ClientConn {
	return c.conn
}

// Close closes the connection to the API service. The user should invoke this when
// the client is no longer required.
func (c *MetricsClient) Close() error {
	return c.conn.Close()
}

// SetGoogleClientInfo sets the name and version of the application in
// the `x-goog-api-client` header passed on each request. Intended for
// use by Google-written clients.
func (c *MetricsClient) SetGoogleClientInfo(name, version string) {
	c.metadata = map[string][]string{
		"x-goog-api-client": {fmt.Sprintf("%s/%s %s gax/%s go/%s", name, version, gapicNameVersion, gax.Version, runtime.Version())},
	}
}

// ParentPath returns the path for the parent resource.
func MetricsParentPath(project string) string {
	path, err := metricsParentPathTemplate.Render(map[string]string{
		"project": project,
	})
	if err != nil {
		panic(err)
	}
	return path
}

// MetricPath returns the path for the metric resource.
func MetricsMetricPath(project string, metric string) string {
	path, err := metricsMetricPathTemplate.Render(map[string]string{
		"project": project,
		"metric":  metric,
	})
	if err != nil {
		panic(err)
	}
	return path
}

// ListLogMetrics lists logs-based metrics.
func (c *MetricsClient) ListLogMetrics(ctx context.Context, req *loggingpb.ListLogMetricsRequest) *LogMetricIterator {
	ctx = metadata.NewContext(ctx, c.metadata)
	it := &LogMetricIterator{}
	it.apiCall = func() error {
		var resp *loggingpb.ListLogMetricsResponse
		err := gax.Invoke(ctx, func(ctx context.Context) error {
			var err error
			req.PageToken = it.nextPageToken
			req.PageSize = it.pageSize
			resp, err = c.client.ListLogMetrics(ctx, req)
			return err
		}, c.CallOptions.ListLogMetrics...)
		if err != nil {
			return err
		}
		if resp.NextPageToken == "" {
			it.atLastPage = true
		}
		it.nextPageToken = resp.NextPageToken
		it.items = resp.Metrics
		return nil
	}
	return it
}

// GetLogMetric gets a logs-based metric.
func (c *MetricsClient) GetLogMetric(ctx context.Context, req *loggingpb.GetLogMetricRequest) (*loggingpb.LogMetric, error) {
	ctx = metadata.NewContext(ctx, c.metadata)
	var resp *loggingpb.LogMetric
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		resp, err = c.client.GetLogMetric(ctx, req)
		return err
	}, c.CallOptions.GetLogMetric...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// CreateLogMetric creates a logs-based metric.
func (c *MetricsClient) CreateLogMetric(ctx context.Context, req *loggingpb.CreateLogMetricRequest) (*loggingpb.LogMetric, error) {
	ctx = metadata.NewContext(ctx, c.metadata)
	var resp *loggingpb.LogMetric
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		resp, err = c.client.CreateLogMetric(ctx, req)
		return err
	}, c.CallOptions.CreateLogMetric...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// UpdateLogMetric creates or updates a logs-based metric.
func (c *MetricsClient) UpdateLogMetric(ctx context.Context, req *loggingpb.UpdateLogMetricRequest) (*loggingpb.LogMetric, error) {
	ctx = metadata.NewContext(ctx, c.metadata)
	var resp *loggingpb.LogMetric
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		resp, err = c.client.UpdateLogMetric(ctx, req)
		return err
	}, c.CallOptions.UpdateLogMetric...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// DeleteLogMetric deletes a logs-based metric.
func (c *MetricsClient) DeleteLogMetric(ctx context.Context, req *loggingpb.DeleteLogMetricRequest) error {
	ctx = metadata.NewContext(ctx, c.metadata)
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		_, err = c.client.DeleteLogMetric(ctx, req)
		return err
	}, c.CallOptions.DeleteLogMetric...)
	return err
}

// LogMetricIterator manages a stream of *loggingpb.LogMetric.
type LogMetricIterator struct {
	// The current page data.
	items         []*loggingpb.LogMetric
	atLastPage    bool
	currentIndex  int
	pageSize      int32
	nextPageToken string
	apiCall       func() error
}

// NextPage returns the next page of results.
// It will return at most the number of results specified by the last call to SetPageSize.
// If SetPageSize was never called or was called with a value less than 1,
// the page size is determined by the underlying service.
//
// NextPage may return a second return value of Done along with the last page of results. After
// NextPage returns Done, all subsequent calls to NextPage will return (nil, Done).
//
// Next and NextPage should not be used with the same iterator.
func (it *LogMetricIterator) NextPage() ([]*loggingpb.LogMetric, error) {
	if it.atLastPage {
		// We already returned Done with the last page of items. Continue to
		// return Done, but with no items.
		return nil, Done
	}
	if err := it.apiCall(); err != nil {
		return nil, err
	}
	if it.atLastPage {
		return it.items, Done
	}
	return it.items, nil
}

// Next returns the next result. Its second return value is Done if there are no more results.
// Once next returns Done, all subsequent calls will return Done.
//
// Internally, Next retrieves results in bulk. You can call SetPageSize as a performance hint to
// affect how many results are retrieved in a single RPC.
//
// SetPageToken should not be called when using Next.
//
// Next and NextPage should not be used with the same iterator.
func (it *LogMetricIterator) Next() (*loggingpb.LogMetric, error) {
	for it.currentIndex >= len(it.items) {
		if it.atLastPage {
			return nil, Done
		}
		if err := it.apiCall(); err != nil {
			return nil, err
		}
		it.currentIndex = 0
	}
	result := it.items[it.currentIndex]
	it.currentIndex++
	return result, nil
}

// PageSize returns the page size for all subsequent calls to NextPage.
func (it *LogMetricIterator) PageSize() int {
	return int(it.pageSize)
}

// SetPageSize sets the page size for all subsequent calls to NextPage.
func (it *LogMetricIterator) SetPageSize(pageSize int) {
	if pageSize > math.MaxInt32 {
		pageSize = math.MaxInt32
	}
	it.pageSize = int32(pageSize)
}

// SetPageToken sets the page token for the next call to NextPage, to resume the iteration from
// a previous point.
func (it *LogMetricIterator) SetPageToken(token string) {
	it.nextPageToken = token
}

// NextPageToken returns a page token that can be used with SetPageToken to resume
// iteration from the next page. It returns the empty string if there are no more pages.
func (it *LogMetricIterator) NextPageToken() string {
	return it.nextPageToken
}
