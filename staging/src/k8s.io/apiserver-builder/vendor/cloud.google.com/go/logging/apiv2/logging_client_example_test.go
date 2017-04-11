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

func ExampleNewClient() {
	ctx := context.Background()
	c, err := logging.NewClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use client.
	_ = c
}

func ExampleClient_DeleteLog() {
	ctx := context.Background()
	c, err := logging.NewClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &loggingpb.DeleteLogRequest{
	// TODO: Fill request struct fields.
	}
	err = c.DeleteLog(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
}

func ExampleClient_WriteLogEntries() {
	ctx := context.Background()
	c, err := logging.NewClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &loggingpb.WriteLogEntriesRequest{
	// TODO: Fill request struct fields.
	}
	resp, err := c.WriteLogEntries(ctx, req)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp
}

func ExampleClient_ListLogEntries() {
	ctx := context.Background()
	c, err := logging.NewClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &loggingpb.ListLogEntriesRequest{
	// TODO: Fill request struct fields.
	}
	it := c.ListLogEntries(ctx, req)
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

func ExampleClient_ListMonitoredResourceDescriptors() {
	ctx := context.Background()
	c, err := logging.NewClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	req := &loggingpb.ListMonitoredResourceDescriptorsRequest{
	// TODO: Fill request struct fields.
	}
	it := c.ListMonitoredResourceDescriptors(ctx, req)
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
