// Copyright 2017, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Command helloworld is an example program that collects data for
// video size.
package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"time"

	"go.opencensus.io/examples/exporter"
	"go.opencensus.io/stats"
	"go.opencensus.io/stats/view"
	"go.opencensus.io/tag"
	"go.opencensus.io/trace"
)

var (
	// frontendKey allows us to breakdown the recorded data
	// by the frontend used when uploading the video.
	frontendKey tag.Key

	// videoSize will measure the size of processed videos.
	videoSize *stats.Int64Measure
)

func main() {
	ctx := context.Background()

	// Register an exporter to be able to retrieve
	// the data from the subscribed views.
	e, err := exporter.NewLogExporter(exporter.Options{ReportingInterval: time.Second})
	if err != nil {
		log.Fatal(err)
	}
	e.Start()
	defer e.Stop()
	defer e.Close()

	trace.ApplyConfig(trace.Config{DefaultSampler: trace.AlwaysSample()})

	frontendKey = tag.MustNewKey("example.com/keys/frontend")
	videoSize = stats.Int64("example.com/measure/video_size", "size of processed videos", stats.UnitBytes)
	view.SetReportingPeriod(2 * time.Second)

	// Create view to see the processed video size
	// distribution broken down by frontend.
	// Register will allow view data to be exported.
	if err := view.Register(&view.View{
		Name:        "example.com/views/video_size",
		Description: "processed video size over time",
		TagKeys:     []tag.Key{frontendKey},
		Measure:     videoSize,
		Aggregation: view.Distribution(1<<16, 1<<32),
	}); err != nil {
		log.Fatalf("Cannot register view: %v", err)
	}

	// Process the video.
	process(ctx)

	// Wait for a duration longer than reporting duration to ensure the stats
	// library reports the collected data.
	fmt.Println("Wait longer than the reporting duration...")
	time.Sleep(4 * time.Second)
}

// process processes the video and instruments the processing
// by creating a span and collecting metrics about the operation.
func process(ctx context.Context) {
	ctx, err := tag.New(ctx,
		tag.Insert(frontendKey, "mobile-ios9.3.5"),
	)
	if err != nil {
		log.Fatal(err)
	}
	ctx, span := trace.StartSpan(ctx, "example.com/ProcessVideo")
	defer span.End()
	// Process video.
	// Record the processed video size.

	// Sleep for [1,10] milliseconds to fake work.
	time.Sleep(time.Duration(rand.Intn(10)+1) * time.Millisecond)

	stats.Record(ctx, videoSize.M(25648))
}
