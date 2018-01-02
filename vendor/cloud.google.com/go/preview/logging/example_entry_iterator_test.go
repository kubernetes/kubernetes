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

package logging_test

import (
	"fmt"
	"time"

	"cloud.google.com/go/preview/logging"
	"golang.org/x/net/context"
	"google.golang.org/api/iterator"
)

func ExampleClient_Entries() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	it := client.Entries(ctx, logging.Filter(`logName = "projects/my-project/logs/my-log"`))
	_ = it // TODO: iterate using Next or iterator.Pager.
}

func ExampleFilter_timestamp() {
	// This example demonstrates how to list the last 24 hours of log entries.
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	oneDayAgo := time.Now().Add(-24 * time.Hour)
	t := oneDayAgo.Format(time.RFC3339) // Logging API wants timestamps in RFC 3339 format.
	it := client.Entries(ctx, logging.Filter(fmt.Sprintf(`timestamp > "%s"`, t)))
	_ = it // TODO: iterate using Next or iterator.Pager.
}

func ExampleEntryIterator_Next() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	it := client.Entries(ctx)
	for {
		entry, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			// TODO: Handle error.
		}
		fmt.Println(entry)
	}
}
