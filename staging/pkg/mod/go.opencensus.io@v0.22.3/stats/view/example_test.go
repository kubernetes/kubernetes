// Copyright 2018, OpenCensus Authors
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

package view_test

import (
	"log"

	"go.opencensus.io/stats"
	"go.opencensus.io/stats/view"
)

func Example() {
	// Measures are usually declared and used by instrumented packages.
	m := stats.Int64("example.com/measure/openconns", "open connections", stats.UnitDimensionless)

	// Views are usually registered in your application main function.
	if err := view.Register(&view.View{
		Name:        "example.com/views/openconns",
		Description: "open connections",
		Measure:     m,
		Aggregation: view.Distribution(0, 1000, 2000),
	}); err != nil {
		log.Fatal(err)
	}

	// Use view.RegisterExporter to export collected data.
}
