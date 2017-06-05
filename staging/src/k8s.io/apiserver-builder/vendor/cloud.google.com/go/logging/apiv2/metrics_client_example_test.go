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

package logging_test

import (
	"cloud.google.com/go/logging/apiv2"
	"golang.org/x/net/context"
	loggingpb "google.golang.org/genproto/googleapis/logging/v2"
)

func ExampleNewMetricsClient() {
	ctx := context.Background()
	c, err := logging.NewMetricsClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use client.
	_ = c
}

func ExampleMetricsClient_ListLogMetrics() {
	ctx := context.Background()
	c, err := logging.NewMetricsClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &loggingpb.ListLogMetricsRequest{
	// TODO: Fill request struct fields.
	}
	it := c.ListLogMetrics(ctx, req)
	for {
		resp, err := it.Next()
		if err != nil {
			// TODO: Handle error.
			break
		}
		// TODO: Use resp.
		_ = resp
	}
}

func ExampleMetricsClient_GetLogMetric() {
	ctx := context.Background()
	c, err := logging.NewMetricsClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &loggingpb.GetLogMetricRequest{
	// TODO: Fill request struct fields.
	}
	resp, err := c.GetLogMetric(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleMetricsClient_CreateLogMetric() {
	ctx := context.Background()
	c, err := logging.NewMetricsClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &loggingpb.CreateLogMetricRequest{
	// TODO: Fill request struct fields.
	}
	resp, err := c.CreateLogMetric(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleMetricsClient_UpdateLogMetric() {
	ctx := context.Background()
	c, err := logging.NewMetricsClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &loggingpb.UpdateLogMetricRequest{
	// TODO: Fill request struct fields.
	}
	resp, err := c.UpdateLogMetric(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleMetricsClient_DeleteLogMetric() {
	ctx := context.Background()
	c, err := logging.NewMetricsClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &loggingpb.DeleteLogMetricRequest{
	// TODO: Fill request struct fields.
	}
	err = c.DeleteLogMetric(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
}
