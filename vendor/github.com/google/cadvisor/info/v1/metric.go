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

package v1

import (
	"time"
)

// Type of metric being exported.
type MetricType string

const (
	// Instantaneous value. May increase or decrease.
	MetricGauge MetricType = "gauge"

	// A counter-like value that is only expected to increase.
	MetricCumulative MetricType = "cumulative"
)

// DataType for metric being exported.
type DataType string

const (
	IntType   DataType = "int"
	FloatType DataType = "float"
)

// Spec for custom metric.
type MetricSpec struct {
	// The name of the metric.
	Name string `json:"name"`

	// Type of the metric.
	Type MetricType `json:"type"`

	// Data Type for the stats.
	Format DataType `json:"format"`

	// Display Units for the stats.
	Units string `json:"units"`
}

// An exported metric.
type MetricValBasic struct {
	// Time at which the metric was queried
	Timestamp time.Time `json:"timestamp"`

	// The value of the metric at this point.
	IntValue   int64   `json:"int_value,omitempty"`
	FloatValue float64 `json:"float_value,omitempty"`
}

// An exported metric.
type MetricVal struct {
	// Label associated with a metric
	Label  string            `json:"label,omitempty"`
	Labels map[string]string `json:"labels,omitempty"`

	// Time at which the metric was queried
	Timestamp time.Time `json:"timestamp"`

	// The value of the metric at this point.
	IntValue   int64   `json:"int_value,omitempty"`
	FloatValue float64 `json:"float_value,omitempty"`
}
