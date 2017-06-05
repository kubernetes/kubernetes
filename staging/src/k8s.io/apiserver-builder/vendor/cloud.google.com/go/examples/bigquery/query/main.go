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

// query is an example client of the bigquery client library.
// It submits a query and writes the result to a table.
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
	project = flag.String("project", "", "The ID of a Google Cloud Platform project")
	dataset = flag.String("dataset", "", "The ID of a BigQuery dataset")
	q       = flag.String("q", "", "The query string")
	dest    = flag.String("dest", "", "The ID of the BigQuery table to write the result to.  If unset, an ephemeral table ID will be generated.")
	pollint = flag.Duration("pollint", 10*time.Second, "Polling interval for checking job status")
	wait    = flag.Bool("wait", false, "Whether to wait for the query job to complete.")
)

func main() {
	flag.Parse()

	flagsOk := true
	for _, f := range []string{"project", "dataset", "q"} {
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

	d := &bigquery.Table{}
	if *dest != "" {
		d = client.Dataset(*dataset).Table(*dest)
	}

	query := &bigquery.Query{
		Q:                *q,
		DefaultProjectID: *project,
		DefaultDatasetID: *dataset,
	}

	// Query data.
	job, err := client.Copy(ctx, d, query, bigquery.WriteTruncate)

	if err != nil {
		log.Fatalf("Querying: %v", err)
	}

	fmt.Printf("Submitted query. Job ID: %s\n", job.ID())
	if !*wait {
		return
	}

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
