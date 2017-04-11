/*
Copyright 2015 Google Inc. All Rights Reserved.

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
Loadtest does some load testing through the Go client library for Cloud Bigtable.
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
	"time"

	"cloud.google.com/go/bigtable"
	"cloud.google.com/go/bigtable/internal/cbtrc"
	"cloud.google.com/go/bigtable/internal/stat"
	"golang.org/x/net/context"
	"google.golang.org/api/option"
)

var (
	runFor       = flag.Duration("run_for", 5*time.Second, "how long to run the load test for")
	scratchTable = flag.String("scratch_table", "loadtest-scratch", "name of table to use; should not already exist")
	csvOutput    = flag.String("csv_output", "",
		"output path for statistics in .csv format. If this file already exists it will be overwritten.")
	poolSize = flag.Int("pool_size", 1, "size of the gRPC connection pool to use for the data client")
	reqCount = flag.Int("req_count", 100, "number of concurrent requests")

	config      *cbtrc.Config
	client      *bigtable.Client
	adminClient *bigtable.AdminClient
)

func main() {
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
	if flag.NArg() != 0 {
		flag.Usage()
		os.Exit(1)
	}

	var options []option.ClientOption
	if *poolSize > 1 {
		options = append(options, option.WithGRPCConnectionPool(*poolSize))
	}

	var csvFile *os.File
	if *csvOutput != "" {
		csvFile, err = os.Create(*csvOutput)
		if err != nil {
			log.Fatalf("creating csv output file: %v", err)
		}
		defer csvFile.Close()
		log.Printf("Writing statistics to %q ...", *csvOutput)
	}

	log.Printf("Dialing connections...")
	client, err = bigtable.NewClient(context.Background(), config.Project, config.Instance, options...)
	if err != nil {
		log.Fatalf("Making bigtable.Client: %v", err)
	}
	defer client.Close()
	adminClient, err = bigtable.NewAdminClient(context.Background(), config.Project, config.Instance)
	if err != nil {
		log.Fatalf("Making bigtable.AdminClient: %v", err)
	}
	defer adminClient.Close()

	// Create a scratch table.
	log.Printf("Setting up scratch table...")
	if err := adminClient.CreateTable(context.Background(), *scratchTable); err != nil {
		log.Fatalf("Making scratch table %q: %v", *scratchTable, err)
	}
	if err := adminClient.CreateColumnFamily(context.Background(), *scratchTable, "f"); err != nil {
		log.Fatalf("Making scratch table column family: %v", err)
	}
	// Upon a successful run, delete the table. Don't bother checking for errors.
	defer adminClient.DeleteTable(context.Background(), *scratchTable)

	log.Printf("Starting load test... (run for %v)", *runFor)
	tbl := client.Open(*scratchTable)
	sem := make(chan int, *reqCount) // limit the number of requests happening at once
	var reads, writes stats
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
			var stats *stats
			defer func() {
				stats.Record(ok, time.Since(opStart))
			}()

			row := fmt.Sprintf("row%d", rand.Intn(100)) // operate on 1 of 100 rows

			switch rand.Intn(10) {
			default:
				// read
				stats = &reads
				_, err := tbl.ReadRow(context.Background(), row, bigtable.RowFilter(bigtable.LatestNFilter(1)))
				if err != nil {
					log.Printf("Error doing read: %v", err)
					ok = false
				}
			case 0, 1, 2, 3, 4:
				// write
				stats = &writes
				mut := bigtable.NewMutation()
				mut.Set("f", "col", bigtable.Now(), bytes.Repeat([]byte("0"), 1<<10)) // 1 KB write
				if err := tbl.Apply(context.Background(), row, mut); err != nil {
					log.Printf("Error doing mutation: %v", err)
					ok = false
				}
			}
		}()
	}
	wg.Wait()

	readsAgg := stat.NewAggregate("reads", reads.ds, reads.tries-reads.ok)
	writesAgg := stat.NewAggregate("writes", writes.ds, writes.tries-writes.ok)
	log.Printf("Reads (%d ok / %d tries):\n%v", reads.ok, reads.tries, readsAgg)
	log.Printf("Writes (%d ok / %d tries):\n%v", writes.ok, writes.tries, writesAgg)

	if csvFile != nil {
		stat.WriteCSV([]*stat.Aggregate{readsAgg, writesAgg}, csvFile)
	}
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
