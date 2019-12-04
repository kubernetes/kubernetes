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
)

// Point is a single data point of a time series.
type Point struct {
	// Time is the point in time that this point represents in a time series.
	Time time.Time
	// Value is the value of this point. Prefer using ReadValue to switching on
	// the value type, since new value types might be added.
	Value interface{}
}

//go:generate stringer -type ValueType

// NewFloat64Point creates a new Point holding a float64 value.
func NewFloat64Point(t time.Time, val float64) Point {
	return Point{
		Value: val,
		Time:  t,
	}
}

// NewInt64Point creates a new Point holding an int64 value.
func NewInt64Point(t time.Time, val int64) Point {
	return Point{
		Value: val,
		Time:  t,
	}
}

// NewDistributionPoint creates a new Point holding a Distribution value.
func NewDistributionPoint(t time.Time, val *Distribution) Point {
	return Point{
		Value: val,
		Time:  t,
	}
}

// NewSummaryPoint creates a new Point holding a Summary value.
func NewSummaryPoint(t time.Time, val *Summary) Point {
	return Point{
		Value: val,
		Time:  t,
	}
}

// ValueVisitor allows reading the value of a point.
type ValueVisitor interface {
	VisitFloat64Value(float64)
	VisitInt64Value(int64)
	VisitDistributionValue(*Distribution)
	VisitSummaryValue(*Summary)
}

// ReadValue accepts a ValueVisitor and calls the appropriate method with the
// value of this point.
// Consumers of Point should use this in preference to switching on the type
// of the value directly, since new value types may be added.
func (p Point) ReadValue(vv ValueVisitor) {
	switch v := p.Value.(type) {
	case int64:
		vv.VisitInt64Value(v)
	case float64:
		vv.VisitFloat64Value(v)
	case *Distribution:
		vv.VisitDistributionValue(v)
	case *Summary:
		vv.VisitSummaryValue(v)
	default:
		panic("unexpected value type")
	}
}

// Distribution contains summary statistics for a population of values. It
// optionally contains a histogram representing the distribution of those
// values across a set of buckets.
type Distribution struct {
	// Count is the number of values in the population. Must be non-negative. This value
	// must equal the sum of the values in bucket_counts if a histogram is
	// provided.
	Count int64
	// Sum is the sum of the values in the population. If count is zero then this field
	// must be zero.
	Sum float64
	// SumOfSquaredDeviation is the sum of squared deviations from the mean of the values in the
	// population. For values x_i this is:
	//
	//     Sum[i=1..n]((x_i - mean)^2)
	//
	// Knuth, "The Art of Computer Programming", Vol. 2, page 323, 3rd edition
	// describes Welford's method for accumulating this sum in one pass.
	//
	// If count is zero then this field must be zero.
	SumOfSquaredDeviation float64
	// BucketOptions describes the bounds of the histogram buckets in this
	// distribution.
	//
	// A Distribution may optionally contain a histogram of the values in the
	// population.
	//
	// If nil, there is no associated histogram.
	BucketOptions *BucketOptions
	// Bucket If the distribution does not have a histogram, then omit this field.
	// If there is a histogram, then the sum of the values in the Bucket counts
	// must equal the value in the count field of the distribution.
	Buckets []Bucket
}

// BucketOptions describes the bounds of the histogram buckets in this
// distribution.
type BucketOptions struct {
	// Bounds specifies a set of bucket upper bounds.
	// This defines len(bounds) + 1 (= N) buckets. The boundaries for bucket
	// index i are:
	//
	// [0, Bounds[i]) for i == 0
	// [Bounds[i-1], Bounds[i]) for 0 < i < N-1
	// [Bounds[i-1], +infinity) for i == N-1
	Bounds []float64
}

// Bucket represents a single bucket (value range) in a distribution.
type Bucket struct {
	// Count is the number of values in each bucket of the histogram, as described in
	// bucket_bounds.
	Count int64
	// Exemplar associated with this bucket (if any).
	Exemplar *Exemplar
}

// Summary is a representation of percentiles.
type Summary struct {
	// Count is the cumulative count (if available).
	Count int64
	// Sum is the cumulative sum of values  (if available).
	Sum float64
	// HasCountAndSum is true if Count and Sum are available.
	HasCountAndSum bool
	// Snapshot represents percentiles calculated over an arbitrary time window.
	// The values in this struct can be reset at arbitrary unknown times, with
	// the requirement that all of them are reset at the same time.
	Snapshot Snapshot
}

// Snapshot represents percentiles over an arbitrary time.
// The values in this struct can be reset at arbitrary unknown times, with
// the requirement that all of them are reset at the same time.
type Snapshot struct {
	// Count is the number of values in the snapshot. Optional since some systems don't
	// expose this. Set to 0 if not available.
	Count int64
	// Sum is the sum of values in the snapshot. Optional since some systems don't
	// expose this. If count is 0 then this field must be zero.
	Sum float64
	// Percentiles is a map from percentile (range (0-100.0]) to the value of
	// the percentile.
	Percentiles map[float64]float64
}

//go:generate stringer -type Type

// Type is the overall type of metric, including its value type and whether it
// represents a cumulative total (since the start time) or if it represents a
// gauge value.
type Type int

// Metric types.
const (
	TypeGaugeInt64 Type = iota
	TypeGaugeFloat64
	TypeGaugeDistribution
	TypeCumulativeInt64
	TypeCumulativeFloat64
	TypeCumulativeDistribution
	TypeSummary
)
