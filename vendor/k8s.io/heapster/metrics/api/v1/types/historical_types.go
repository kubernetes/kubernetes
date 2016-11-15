// Copyright 2016 Google Inc. All Rights Reserved.
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

package types

import (
	"time"
)

// MetricValue is either a floating point value or an unsigned integer value
type MetricValue struct {
	IntValue   *int64   `json:"intValue,omitempty"`
	FloatValue *float64 `json:"floatValue,omitempty"`
}

// MetricAggregationBucket holds information about various aggregations across a single bucket of time
type MetricAggregationBucket struct {
	Timestamp time.Time `json:"timestamp"`
	Count     *uint64   `json:"count,omitempty"`

	Average *MetricValue `json:"average,omitempty"`
	Maximum *MetricValue `json:"maximum,omitempty"`
	Minimum *MetricValue `json:"minimum,omitempty"`
	Median  *MetricValue `json:"median,omitempty"`

	Percentiles map[string]MetricValue `json:"percentiles,omitempty"`
}

// MetricAggregationResult holds a series of MetricAggregationBuckets of a particular size
type MetricAggregationResult struct {
	Buckets    []MetricAggregationBucket `json:"buckets"`
	BucketSize time.Duration             `json:"bucketSize"`
}

// MetricAggregationResultList is a list of MetricAggregationResults, each for a different object
type MetricAggregationResultList struct {
	Items []MetricAggregationResult `json:"items"`
}
