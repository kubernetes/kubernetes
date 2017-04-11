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
	configParentPathTemplate = gax.MustCompilePathTemplate("projects/{project}")
	configSinkPathTemplate   = gax.MustCompilePathTemplate("projects/{project}/sinks/{sink}")
)

// ConfigCallOptions contains the retry settings for each method of this client.
type ConfigCallOptions struct {
	ListSinks  []gax.CallOption
	GetSink    []gax.CallOption
	CreateSink []gax.CallOption
	UpdateSink []gax.CallOption
	DeleteSink []gax.CallOption
}

func defaultConfigClientOptions() []option.ClientOption {
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

func defaultConfigCallOptions() *ConfigCallOptions {
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

	return &ConfigCallOptions{
		ListSinks:  retry[[2]string{"default", "idempotent"}],
		GetSink:    retry[[2]string{"default", "idempotent"}],
		CreateSink: retry[[2]string{"default", "non_idempotent"}],
		UpdateSink: retry[[2]string{"default", "non_idempotent"}],
		DeleteSink: retry[[2]string{"default", "idempotent"}],
	}
}

// ConfigClient is a client for interacting with ConfigServiceV2.
type ConfigClient struct {
	// The connection to the service.
	conn *grpc.ClientConn

	// The gRPC API client.
	client loggingpb.ConfigServiceV2Client

	// The call options for this service.
	CallOptions *ConfigCallOptions

	// The metadata to be sent with each request.
	metadata map[string][]string
}

// NewConfigClient creates a new config service client.
//
// Service for configuring sinks used to export log entries outside Stackdriver
// Logging.
func NewConfigClient(ctx context.Context, opts ...option.ClientOption) (*ConfigClient, error) {
	conn, err := transport.DialGRPC(ctx, append(defaultConfigClientOptions(), opts...)...)
	if err != nil {
		return nil, err
	}
	c := &ConfigClient{
		conn:        conn,
		client:      loggingpb.NewConfigServiceV2Client(conn),
		CallOptions: defaultConfigCallOptions(),
	}
	c.SetGoogleClientInfo("gax", gax.Version)
	return c, nil
}

// Connection returns the client's connection to the API service.
func (c *ConfigClient) Connection() *grpc.ClientConn {
	return c.conn
}

// Close closes the connection to the API service. The user should invoke this when
// the client is no longer required.
func (c *ConfigClient) Close() error {
	return c.conn.Close()
}

// SetGoogleClientInfo sets the name and version of the application in
// the `x-goog-api-client` header passed on each request. Intended for
// use by Google-written clients.
func (c *ConfigClient) SetGoogleClientInfo(name, version string) {
	c.metadata = map[string][]string{
		"x-goog-api-client": {fmt.Sprintf("%s/%s %s gax/%s go/%s", name, version, gapicNameVersion, gax.Version, runtime.Version())},
	}
}

// ParentPath returns the path for the parent resource.
func ConfigParentPath(project string) string {
	path, err := configParentPathTemplate.Render(map[string]string{
		"project": project,
	})
	if err != nil {
		panic(err)
	}
	return path
}

// SinkPath returns the path for the sink resource.
func ConfigSinkPath(project string, sink string) string {
	path, err := configSinkPathTemplate.Render(map[string]string{
		"project": project,
		"sink":    sink,
	})
	if err != nil {
		panic(err)
	}
	return path
}

// ListSinks lists sinks.
func (c *ConfigClient) ListSinks(ctx context.Context, req *loggingpb.ListSinksRequest) *LogSinkIterator {
	ctx = metadata.NewContext(ctx, c.metadata)
	it := &LogSinkIterator{}
	it.apiCall = func() error {
		var resp *loggingpb.ListSinksResponse
		err := gax.Invoke(ctx, func(ctx context.Context) error {
			var err error
			req.PageToken = it.nextPageToken
			req.PageSize = it.pageSize
			resp, err = c.client.ListSinks(ctx, req)
			return err
		}, c.CallOptions.ListSinks...)
		if err != nil {
			return err
		}
		if resp.NextPageToken == "" {
			it.atLastPage = true
		}
		it.nextPageToken = resp.NextPageToken
		it.items = resp.Sinks
		return nil
	}
	return it
}

// GetSink gets a sink.
func (c *ConfigClient) GetSink(ctx context.Context, req *loggingpb.GetSinkRequest) (*loggingpb.LogSink, error) {
	ctx = metadata.NewContext(ctx, c.metadata)
	var resp *loggingpb.LogSink
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		resp, err = c.client.GetSink(ctx, req)
		return err
	}, c.CallOptions.GetSink...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// CreateSink creates a sink.
func (c *ConfigClient) CreateSink(ctx context.Context, req *loggingpb.CreateSinkRequest) (*loggingpb.LogSink, error) {
	ctx = metadata.NewContext(ctx, c.metadata)
	var resp *loggingpb.LogSink
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		resp, err = c.client.CreateSink(ctx, req)
		return err
	}, c.CallOptions.CreateSink...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// UpdateSink creates or updates a sink.
func (c *ConfigClient) UpdateSink(ctx context.Context, req *loggingpb.UpdateSinkRequest) (*loggingpb.LogSink, error) {
	ctx = metadata.NewContext(ctx, c.metadata)
	var resp *loggingpb.LogSink
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		resp, err = c.client.UpdateSink(ctx, req)
		return err
	}, c.CallOptions.UpdateSink...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// DeleteSink deletes a sink.
func (c *ConfigClient) DeleteSink(ctx context.Context, req *loggingpb.DeleteSinkRequest) error {
	ctx = metadata.NewContext(ctx, c.metadata)
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		_, err = c.client.DeleteSink(ctx, req)
		return err
	}, c.CallOptions.DeleteSink...)
	return err
}

// LogSinkIterator manages a stream of *loggingpb.LogSink.
type LogSinkIterator struct {
	// The current page data.
	items         []*loggingpb.LogSink
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
func (it *LogSinkIterator) NextPage() ([]*loggingpb.LogSink, error) {
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
func (it *LogSinkIterator) Next() (*loggingpb.LogSink, error) {
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
func (it *LogSinkIterator) PageSize() int {
	return int(it.pageSize)
}

// SetPageSize sets the page size for all subsequent calls to NextPage.
func (it *LogSinkIterator) SetPageSize(pageSize int) {
	if pageSize > math.MaxInt32 {
		pageSize = math.MaxInt32
	}
	it.pageSize = int32(pageSize)
}

// SetPageToken sets the page token for the next call to NextPage, to resume the iteration from
// a previous point.
func (it *LogSinkIterator) SetPageToken(token string) {
	it.nextPageToken = token
}

// NextPageToken returns a page token that can be used with SetPageToken to resume
// iteration from the next page. It returns the empty string if there are no more pages.
func (it *LogSinkIterator) NextPageToken() string {
	return it.nextPageToken
}
