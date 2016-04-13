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

// TODO(mcgreevy): support dry-run mode when creating jobs.

import (
	"fmt"
	"net/http"

	"golang.org/x/net/context"
	bq "google.golang.org/api/bigquery/v2"
)

// A Source is a source of data for the Copy function.
type Source interface {
	implementsSource()
}

// A Destination is a destination of data for the Copy function.
type Destination interface {
	implementsDestination()
}

// An Option is an optional argument to Copy.
type Option interface {
	implementsOption()
}

// A ReadSource is a source of data for the Read function.
type ReadSource interface {
	implementsReadSource()
}

// A ReadOption is an optional argument to Read.
type ReadOption interface {
	customizeRead(conf *pagingConf)
}

const Scope = "https://www.googleapis.com/auth/bigquery"

// Client may be used to perform BigQuery operations.
type Client struct {
	service   service
	projectID string
}

// Note: many of the methods on *Client appear in the various *_op.go source files.

// NewClient constructs a new Client which can perform BigQuery operations.
// Operations performed via the client are billed to the specified GCP project.
// The supplied http.Client is used for making requests to the BigQuery server and must be capable of
// authenticating requests with Scope.
func NewClient(client *http.Client, projectID string) (*Client, error) {
	bqService, err := newBigqueryService(client)
	if err != nil {
		return nil, fmt.Errorf("constructing bigquery client: %v", err)
	}

	c := &Client{
		service:   bqService,
		projectID: projectID,
	}
	return c, nil
}

// initJobProto creates and returns a bigquery Job proto.
// The proto is customized using any jobOptions in options.
// The list of Options is returned with the jobOptions removed.
func initJobProto(projectID string, options []Option) (*bq.Job, []Option) {
	job := &bq.Job{}

	var other []Option
	for _, opt := range options {
		if o, ok := opt.(jobOption); ok {
			o.customizeJob(job, projectID)
		} else {
			other = append(other, opt)
		}
	}
	return job, other
}

// Copy starts a BigQuery operation to copy data from a Source to a Destination.
func (c *Client) Copy(ctx context.Context, dst Destination, src Source, options ...Option) (*Job, error) {
	switch dst := dst.(type) {
	case *Table:
		switch src := src.(type) {
		case *GCSReference:
			return c.load(ctx, dst, src, options)
		case *Table:
			return c.cp(ctx, dst, Tables{src}, options)
		case Tables:
			return c.cp(ctx, dst, src, options)
		case *Query:
			return c.query(ctx, dst, src, options)
		}
	case *GCSReference:
		if src, ok := src.(*Table); ok {
			return c.extract(ctx, dst, src, options)
		}
	}
	return nil, fmt.Errorf("no Copy operation matches dst/src pair: dst: %T ; src: %T", dst, src)
}

// Read fetches data from a ReadSource and returns the data via an Iterator.
func (c *Client) Read(ctx context.Context, src ReadSource, options ...ReadOption) (*Iterator, error) {
	switch src := src.(type) {
	case *Job:
		return c.readQueryResults(src, options)
	case *Query:
		return c.executeQuery(ctx, src, options...)
	case *Table:
		return c.readTable(src, options)
	}
	return nil, fmt.Errorf("src (%T) does not support the Read operation", src)
}

// executeQuery submits a query for execution and returns the results via an Iterator.
func (c *Client) executeQuery(ctx context.Context, q *Query, options ...ReadOption) (*Iterator, error) {
	dest := &Table{}
	job, err := c.Copy(ctx, dest, q, WriteTruncate)
	if err != nil {
		return nil, err
	}

	return c.Read(ctx, job, options...)
}

func (c *Client) Dataset(id string) *Dataset {
	return &Dataset{
		id:     id,
		client: c,
	}
}
