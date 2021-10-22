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

package stats_test

import (
	"context"

	"go.opencensus.io/stats"
)

func ExampleRecord() {
	ctx := context.Background()

	// Measures are usually declared as package-private global variables.
	openConns := stats.Int64("example.com/measure/openconns", "open connections", stats.UnitDimensionless)

	// Instrumented packages call stats.Record() to record measuremens.
	stats.Record(ctx, openConns.M(124)) // Record 124 open connections.

	// Without any views or exporters registered, this statement has no observable effects.
}
