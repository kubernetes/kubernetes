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

// Sink describes a sink used to export log entries outside Stackdriver
// Logging. Incoming log entries matching a filter are exported to a
// destination (a Cloud Storage bucket, BigQuery dataset or Cloud Pub/Sub
// topic).
//
// For more information, see https://cloud.google.com/logging/docs/export/using_exported_logs.
// (The Sinks in this package are what the documentation refers to as "project sinks".)
type Sink struct {
	// ID is a client-assigned sink identifier. Example:
	// "my-severe-errors-to-pubsub".
	// Sink identifiers are limited to 1000 characters
	// and can include only the following characters: A-Z, a-z,
	// 0-9, and the special characters "_-.".
	ID string

	// Destination is the export destination. See
	// https://cloud.google.com/logging/docs/api/tasks/exporting-logs.
	// Examples: "storage.googleapis.com/a-bucket",
	// "bigquery.googleapis.com/projects/a-project-id/datasets/a-dataset".
	Destination string

	// Filter optionally specifies an advanced logs filter (see
	// https://cloud.google.com/logging/docs/view/advanced_filters) that
	// defines the log entries to be exported. Example: "logName:syslog AND
	// severity>=ERROR". If omitted, all entries are returned.
	Filter string
}

// CreateSink creates a Sink. It returns an error if the Sink already exists.
// Requires AdminScope.
func (c *Client) CreateSink(ctx context.Context, sink *Sink) (*Sink, error) {
	ls, err := c.sClient.CreateSink(ctx, &logpb.CreateSinkRequest{
		Parent: c.parent(),
		Sink:   toLogSink(sink),
	})
	if err != nil {
		fmt.Printf("Sink: %+v\n", toLogSink(sink))
		return nil, err
	}
	return fromLogSink(ls), nil
}

// DeleteSink deletes a sink. The provided sinkID is the sink's identifier, such as
// "my-severe-errors-to-pubsub".
// Requires AdminScope.
func (c *Client) DeleteSink(ctx context.Context, sinkID string) error {
	return c.sClient.DeleteSink(ctx, &logpb.DeleteSinkRequest{
		SinkName: c.sinkPath(sinkID),
	})
}

// Sink gets a sink. The provided sinkID is the sink's identifier, such as
// "my-severe-errors-to-pubsub".
// Requires ReadScope or AdminScope.
func (c *Client) Sink(ctx context.Context, sinkID string) (*Sink, error) {
	ls, err := c.sClient.GetSink(ctx, &logpb.GetSinkRequest{
		SinkName: c.sinkPath(sinkID),
	})
	if err != nil {
		return nil, err
	}
	return fromLogSink(ls), nil
}

// UpdateSink updates an existing Sink, or creates a new one if the Sink doesn't exist.
// Requires AdminScope.
func (c *Client) UpdateSink(ctx context.Context, sink *Sink) (*Sink, error) {
	ls, err := c.sClient.UpdateSink(ctx, &logpb.UpdateSinkRequest{
		SinkName: c.sinkPath(sink.ID),
		Sink:     toLogSink(sink),
	})
	if err != nil {
		return nil, err
	}
	return fromLogSink(ls), err
}

func (c *Client) sinkPath(sinkID string) string {
	return fmt.Sprintf("%s/sinks/%s", c.parent(), sinkID)
}

// Sinks returns a SinkIterator for iterating over all Sinks in the Client's project.
// Requires ReadScope or AdminScope.
func (c *Client) Sinks(ctx context.Context) *SinkIterator {
	it := &SinkIterator{
		ctx:    ctx,
		client: c.sClient,
		req:    &logpb.ListSinksRequest{Parent: c.parent()},
	}
	it.pageInfo, it.nextFunc = iterator.NewPageInfo(
		it.fetch,
		func() int { return len(it.items) },
		func() interface{} { b := it.items; it.items = nil; return b })
	return it
}

// A SinkIterator iterates over Sinks.
type SinkIterator struct {
	ctx      context.Context
	client   *vkit.ConfigClient
	pageInfo *iterator.PageInfo
	nextFunc func() error
	req      *logpb.ListSinksRequest
	items    []*Sink
}

// PageInfo supports pagination. See the google.golang.org/api/iterator package for details.
func (it *SinkIterator) PageInfo() *iterator.PageInfo { return it.pageInfo }

// Next returns the next result. Its second return value is Done if there are
// no more results. Once Next returns Done, all subsequent calls will return
// Done.
func (it *SinkIterator) Next() (*Sink, error) {
	if err := it.nextFunc(); err != nil {
		return nil, err
	}
	item := it.items[0]
	it.items = it.items[1:]
	return item, nil
}

func (it *SinkIterator) fetch(pageSize int, pageToken string) (string, error) {
	// TODO(jba): Do this a nicer way if the generated code supports one.
	// TODO(jba): If the above TODO can't be done, find a way to pass metadata in the call.
	client := logpb.NewConfigServiceV2Client(it.client.Connection())
	var res *logpb.ListSinksResponse
	err := gax.Invoke(it.ctx, func(ctx context.Context) error {
		it.req.PageSize = trunc32(pageSize)
		it.req.PageToken = pageToken
		var err error
		res, err = client.ListSinks(ctx, it.req)
		return err
	}, it.client.CallOptions.ListSinks...)
	if err != nil {
		return "", err
	}
	for _, sp := range res.Sinks {
		it.items = append(it.items, fromLogSink(sp))
	}
	return res.NextPageToken, nil
}

func toLogSink(s *Sink) *logpb.LogSink {
	return &logpb.LogSink{
		Name:                s.ID,
		Destination:         s.Destination,
		Filter:              s.Filter,
		OutputVersionFormat: logpb.LogSink_V2,
	}
}

func fromLogSink(ls *logpb.LogSink) *Sink {
	return &Sink{
		ID:          ls.Name,
		Destination: ls.Destination,
		Filter:      ls.Filter,
	}
}
