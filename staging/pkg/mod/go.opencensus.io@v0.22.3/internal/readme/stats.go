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

// Package readme generates the README.
package readme // import "go.opencensus.io/internal/readme"

import (
	"context"
	"log"

	"go.opencensus.io/stats"
	"go.opencensus.io/stats/view"
)

// README.md is generated with the examples here by using embedmd.
// For more details, see https://github.com/rakyll/embedmd.

func statsExamples() {
	ctx := context.Background()

	videoSize := stats.Int64("example.com/video_size", "processed video size", "MB")

	// START aggs
	distAgg := view.Distribution(1<<32, 2<<32, 3<<32)
	countAgg := view.Count()
	sumAgg := view.Sum()
	// END aggs

	_, _, _ = distAgg, countAgg, sumAgg

	// START view
	if err := view.Register(&view.View{
		Name:        "example.com/video_size_distribution",
		Description: "distribution of processed video size over time",
		Measure:     videoSize,
		Aggregation: view.Distribution(1<<32, 2<<32, 3<<32),
	}); err != nil {
		log.Fatalf("Failed to register view: %v", err)
	}
	// END view

	// START record
	stats.Record(ctx, videoSize.M(102478))
	// END record
}
