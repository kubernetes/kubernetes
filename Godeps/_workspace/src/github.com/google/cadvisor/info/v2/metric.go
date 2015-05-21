// Copyright 2015 Google Inc. All Rights Reserved.
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

package v2

import (
	"time"
)

// Type of metric being exported.
type MetricType string

const (
	// Instantaneous value. May increase or decrease.
	MetricGauge MetricType = "gauge"

	// A counter-like value that is only expected to increase.
	MetricCumulative = "cumulative"

	// Rate over a time period.
	MetricDelta = "delta"
)

// An exported metric.
type Metric struct {
	// The name of the metric.
	Name string `json:"name"`

	// Type of the metric.
	Type MetricType `json:"type"`

	// Metadata associated with this metric.
	Labels map[string]string

	// Value of the metric. Only one of these values will be
	// available according to the output type of the metric.
	// If no values are available, there are no data points.
	IntPoints   []IntPoint   `json:"int_points,omitempty"`
	FloatPoints []FloatPoint `json:"float_points,omitempty"`
}

// An integer metric data point.
type IntPoint struct {
	// Time at which the metric was queried
	Timestamp time.Time `json:"timestamp"`

	// The value of the metric at this point.
	Value int64 `json:"value"`
}

// A float metric data point.
type FloatPoint struct {
	// Time at which the metric was queried
	Timestamp time.Time `json:"timestamp"`

	// The value of the metric at this point.
	Value float64 `json:"value"`
}
