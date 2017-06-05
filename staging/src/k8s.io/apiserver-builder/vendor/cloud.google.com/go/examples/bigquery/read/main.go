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

// read is an example client of the bigquery client library.
// It reads from a table, returning the data via an Iterator.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
	"text/tabwriter"

	"cloud.google.com/go/bigquery"
	"golang.org/x/net/context"
)

var (
	project = flag.String("project", "", "The ID of a Google Cloud Platform project")
	dataset = flag.String("dataset", "", "The ID of a BigQuery dataset")
	table   = flag.String("table", ".*", "A regular expression to match the IDs of tables to read.")
	jobID   = flag.String("jobid", "", "The ID of a query job that has already been submitted."+
		" If set, --dataset, --table will be ignored, and results will be read from the specified job.")
)

func printValues(ctx context.Context, it *bigquery.Iterator) {
	// one-space padding.
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 1, ' ', 0)

	for it.Next(ctx) {
		var vals bigquery.ValueList
		if err := it.Get(&vals); err != nil {
			fmt.Printf("err calling get: %v\n", err)
		} else {
			sep := ""
			for _, v := range vals {
				fmt.Fprintf(tw, "%s%v", sep, v)
				sep = "\t"
			}
			fmt.Fprintf(tw, "\n")
		}
	}
	tw.Flush()

	fmt.Printf("\n")
	if err := it.Err(); err != nil {
		fmt.Printf("err reading: %v\n", err)
	}
}

func printTable(ctx context.Context, client *bigquery.Client, t *bigquery.Table) {
	it, err := client.Read(ctx, t)
	if err != nil {
		log.Fatalf("Reading: %v", err)
	}

	id := t.FullyQualifiedName()
	fmt.Printf("%s\n%s\n", id, strings.Repeat("-", len(id)))
	printValues(ctx, it)
}

func printQueryResults(ctx context.Context, client *bigquery.Client, queryJobID string) {
	job, err := client.JobFromID(ctx, queryJobID)
	if err != nil {
		log.Fatalf("Loading job: %v", err)
	}

	it, err := client.Read(ctx, job)
	if err != nil {
		log.Fatalf("Reading: %v", err)
	}

	// TODO: print schema.
	printValues(ctx, it)
}

func main() {
	flag.Parse()

	flagsOk := true
	if flag.Lookup("project").Value.String() == "" {
		fmt.Fprintf(os.Stderr, "Flag --project is required\n")
		flagsOk = false
	}

	var sourceFlagCount int
	if flag.Lookup("dataset").Value.String() != "" {
		sourceFlagCount++
	}
	if flag.Lookup("jobid").Value.String() != "" {
		sourceFlagCount++
	}
	if sourceFlagCount != 1 {
		fmt.Fprintf(os.Stderr, "Exactly one of --dataset or --jobid must be set\n")
		flagsOk = false
	}

	if !flagsOk {
		os.Exit(1)
	}

	ctx := context.Background()
	tableRE, err := regexp.Compile(*table)
	if err != nil {
		fmt.Fprintf(os.Stderr, "--table is not a valid regular expression: %q\n", *table)
		os.Exit(1)
	}

	client, err := bigquery.NewClient(ctx, *project)
	if err != nil {
		log.Fatalf("Creating bigquery client: %v", err)
	}

	if *jobID != "" {
		printQueryResults(ctx, client, *jobID)
		return
	}
	ds := client.Dataset(*dataset)
	var tables []*bigquery.Table
	tables, err = ds.ListTables(context.Background())
	if err != nil {
		log.Fatalf("Listing tables: %v", err)
	}
	for _, t := range tables {
		if tableRE.MatchString(t.TableID) {
			printTable(ctx, client, t)
		}
	}
}
