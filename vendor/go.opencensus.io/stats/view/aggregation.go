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
//

package view

import "time"

// AggType represents the type of aggregation function used on a View.
type AggType int

// All available aggregation types.
const (
	AggTypeNone         AggType = iota // no aggregation; reserved for future use.
	AggTypeCount                       // the count aggregation, see Count.
	AggTypeSum                         // the sum aggregation, see Sum.
	AggTypeDistribution                // the distribution aggregation, see Distribution.
	AggTypeLastValue                   // the last value aggregation, see LastValue.
)

func (t AggType) String() string {
	return aggTypeName[t]
}

var aggTypeName = map[AggType]string{
	AggTypeNone:         "None",
	AggTypeCount:        "Count",
	AggTypeSum:          "Sum",
	AggTypeDistribution: "Distribution",
	AggTypeLastValue:    "LastValue",
}

// Aggregation represents a data aggregation method. Use one of the functions:
// Count, Sum, or Distribution to construct an Aggregation.
type Aggregation struct {
	Type    AggType   // Type is the AggType of this Aggregation.
	Buckets []float64 // Buckets are the bucket endpoints if this Aggregation represents a distribution, see Distribution.

	newData func(time.Time) AggregationData
}

var (
	aggCount = &Aggregation{
		Type: AggTypeCount,
		newData: func(t time.Time) AggregationData {
			return &CountData{Start: t}
		},
	}
	aggSum = &Aggregation{
		Type: AggTypeSum,
		newData: func(t time.Time) AggregationData {
			return &SumData{Start: t}
		},
	}
)

// Count indicates that data collected and aggregated
// with this method will be turned into a count value.
// For example, total number of accepted requests can be
// aggregated by using Count.
func Count() *Aggregation {
	return aggCount
}

// Sum indicates that data collected and aggregated
// with this method will be summed up.
// For example, accumulated request bytes can be aggregated by using
// Sum.
func Sum() *Aggregation {
	return aggSum
}

// Distribution indicates that the desired aggregation is
// a histogram distribution.
//
// A distribution aggregation may contain a histogram of the values in the
// population. The bucket boundaries for that histogram are described
// by the bounds. This defines len(bounds)+1 buckets.
//
// If len(bounds) >= 2 then the boundaries for bucket index i are:
//
//     [-infinity, bounds[i]) for i = 0
//     [bounds[i-1], bounds[i]) for 0 < i < length
//     [bounds[i-1], +infinity) for i = length
//
// If len(bounds) is 0 then there is no histogram associated with the
// distribution. There will be a single bucket with boundaries
// (-infinity, +infinity).
//
// If len(bounds) is 1 then there is no finite buckets, and that single
// element is the common boundary of the overflow and underflow buckets.
func Distribution(bounds ...float64) *Aggregation {
	agg := &Aggregation{
		Type:    AggTypeDistribution,
		Buckets: bounds,
	}
	agg.newData = func(t time.Time) AggregationData {
		return newDistributionData(agg, t)
	}
	return agg
}

// LastValue only reports the last value recorded using this
// aggregation. All other measurements will be dropped.
func LastValue() *Aggregation {
	return &Aggregation{
		Type: AggTypeLastValue,
		newData: func(_ time.Time) AggregationData {
			return &LastValueData{}
		},
	}
}
