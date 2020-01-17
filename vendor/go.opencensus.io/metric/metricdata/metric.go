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

package metricdata

import (
	"time"

	"go.opencensus.io/resource"
)

// Descriptor holds metadata about a metric.
type Descriptor struct {
	Name        string     // full name of the metric
	Description string     // human-readable description
	Unit        Unit       // units for the measure
	Type        Type       // type of measure
	LabelKeys   []LabelKey // label keys
}

// Metric represents a quantity measured against a resource with different
// label value combinations.
type Metric struct {
	Descriptor Descriptor         // metric descriptor
	Resource   *resource.Resource // resource against which this was measured
	TimeSeries []*TimeSeries      // one time series for each combination of label values
}

// TimeSeries is a sequence of points associated with a combination of label
// values.
type TimeSeries struct {
	LabelValues []LabelValue // label values, same order as keys in the metric descriptor
	Points      []Point      // points sequence
	StartTime   time.Time    // time we started recording this time series
}
