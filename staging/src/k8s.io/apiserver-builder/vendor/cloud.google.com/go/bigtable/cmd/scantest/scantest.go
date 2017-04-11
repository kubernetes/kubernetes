/*
Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
Scantest does scan-related load testing against Cloud Bigtable. The logic here
mimics a similar test written using the Java client.
*/
package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"sync"
	"sync/atomic"
	"text/tabwriter"
	"time"

	"cloud.google.com/go/bigtable"
	"cloud.google.com/go/bigtable/internal/cbtrc"
	"cloud.google.com/go/bigtable/internal/stat"
	"golang.org/x/net/context"
)

var (
	runFor   = flag.Duration("run_for", 5*time.Second, "how long to run the load test for")
	numScans = flag.Int("concurrent_scans", 1, "number of concurrent scans")
	rowLimit = flag.Int("row_limit", 10000, "max number of records per scan")

	config *cbtrc.Config
	client *bigtable.Client
)

func main() {
	flag.Usage = func() {
		fmt.Printf("Usage: scantest [options] <table_name>\n\n")
		flag.PrintDefaults()
	}

	var err error
	config, err = cbtrc.Load()
	if err != nil {
		log.Fatal(err)
	}
	config.RegisterFlags()

	flag.Parse()
	if err := config.CheckFlags(); err != nil {
		log.Fatal(err)
	}
	if config.Creds != "" {
		os.Setenv("GOOGLE_APPLICATION_CREDENTIALS", config.Creds)
	}
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(1)
	}

	table := flag.Arg(0)

	log.Printf("Dialing connections...")
	client, err = bigtable.NewClient(context.Background(), config.Project, config.Instance)
	if err != nil {
		log.Fatalf("Making bigtable.Client: %v", err)
	}
	defer client.Close()

	log.Printf("Starting scan test... (run for %v)", *runFor)
	tbl := client.Open(table)
	sem := make(chan int, *numScans) // limit the number of requests happening at once
	var scans stats

	stopTime := time.Now().Add(*runFor)
	var wg sync.WaitGroup
	for time.Now().Before(stopTime) {
		sem <- 1
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()

			ok := true
			opStart := time.Now()
			defer func() {
				scans.Record(ok, time.Since(opStart))
			}()

			// Start at a random row key
			key := fmt.Sprintf("user%d", rand.Int63())
			limit := bigtable.LimitRows(int64(*rowLimit))
			noop := func(bigtable.Row) bool { return true }
			if err := tbl.ReadRows(context.Background(), bigtable.NewRange(key, ""), noop, limit); err != nil {
				log.Printf("Error during scan: %v", err)
				ok = false
			}
		}()
	}
	wg.Wait()

	agg := stat.NewAggregate("scans", scans.ds, scans.tries-scans.ok)
	log.Printf("Scans (%d ok / %d tries):\nscan times:\n%v\nthroughput (rows/second):\n%v",
		scans.ok, scans.tries, agg, throughputString(agg))
}

func throughputString(agg *stat.Aggregate) string {
	var buf bytes.Buffer
	tw := tabwriter.NewWriter(&buf, 0, 0, 1, ' ', 0) // one-space padding
	rowLimitF := float64(*rowLimit)
	fmt.Fprintf(
		tw,
		"min:\t%.2f\nmedian:\t%.2f\nmax:\t%.2f\n",
		rowLimitF/agg.Max.Seconds(),
		rowLimitF/agg.Median.Seconds(),
		rowLimitF/agg.Min.Seconds())
	tw.Flush()
	return buf.String()
}

var allStats int64 // atomic

type stats struct {
	mu        sync.Mutex
	tries, ok int
	ds        []time.Duration
}

func (s *stats) Record(ok bool, d time.Duration) {
	s.mu.Lock()
	s.tries++
	if ok {
		s.ok++
	}
	s.ds = append(s.ds, d)
	s.mu.Unlock()

	if n := atomic.AddInt64(&allStats, 1); n%1000 == 0 {
		log.Printf("Progress: done %d ops", n)
	}
}
