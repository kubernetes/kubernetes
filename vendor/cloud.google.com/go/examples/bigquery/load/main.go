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

// load is an example client of the bigquery client library.
// It loads a file from Google Cloud Storage into a BigQuery table.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"cloud.google.com/go/bigquery"
	"golang.org/x/net/context"
)

var (
	project  = flag.String("project", "", "The ID of a Google Cloud Platform project")
	dataset  = flag.String("dataset", "", "The ID of a BigQuery dataset")
	table    = flag.String("table", "", "The ID of a BigQuery table to load data into")
	bucket   = flag.String("bucket", "", "The name of a Google Cloud Storage bucket to load data from")
	object   = flag.String("object", "", "The name of a Google Cloud Storage object to load data from. Must exist within the bucket specified by --bucket")
	skiprows = flag.Int64("skiprows", 0, "The number of rows of the source data to skip when loading")
	pollint  = flag.Duration("pollint", 10*time.Second, "Polling interval for checking job status")
)

func main() {
	flag.Parse()

	flagsOk := true
	for _, f := range []string{"project", "dataset", "table", "bucket", "object"} {
		if flag.Lookup(f).Value.String() == "" {
			fmt.Fprintf(os.Stderr, "Flag --%s is required\n", f)
			flagsOk = false
		}
	}
	if !flagsOk {
		os.Exit(1)
	}

	ctx := context.Background()
	client, err := bigquery.NewClient(ctx, *project)
	if err != nil {
		log.Fatalf("Creating bigquery client: %v", err)
	}

	table := client.Dataset(*dataset).Table(*table)

	gcs := client.NewGCSReference(fmt.Sprintf("gs://%s/%s", *bucket, *object))
	gcs.SkipLeadingRows = *skiprows

	// Load data from Google Cloud Storage into a BigQuery table.
	job, err := client.Copy(
		ctx, table, gcs,
		bigquery.MaxBadRecords(1),
		bigquery.AllowQuotedNewlines(),
		bigquery.WriteTruncate)

	if err != nil {
		log.Fatalf("Loading data: %v", err)
	}

	fmt.Printf("Job for data load operation: %+v\n", job)
	fmt.Printf("Waiting for job to complete.\n")

	for range time.Tick(*pollint) {
		status, err := job.Status(ctx)
		if err != nil {
			fmt.Printf("Failure determining status: %v", err)
			break
		}
		if !status.Done() {
			continue
		}
		if err := status.Err(); err == nil {
			fmt.Printf("Success\n")
		} else {
			fmt.Printf("Failure: %+v\n", err)
		}
		break
	}
}
