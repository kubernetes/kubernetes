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

import (
	"math"
	"time"

	"go.opencensus.io/metric/metricdata"
)

// AggregationData represents an aggregated value from a collection.
// They are reported on the view data during exporting.
// Mosts users won't directly access aggregration data.
type AggregationData interface {
	isAggregationData() bool
	addSample(v float64, attachments map[string]interface{}, t time.Time)
	clone() AggregationData
	equal(other AggregationData) bool
	toPoint(t metricdata.Type, time time.Time) metricdata.Point
}

const epsilon = 1e-9

// CountData is the aggregated data for the Count aggregation.
// A count aggregation processes data and counts the recordings.
//
// Most users won't directly access count data.
type CountData struct {
	Value int64
}

func (a *CountData) isAggregationData() bool { return true }

func (a *CountData) addSample(_ float64, _ map[string]interface{}, _ time.Time) {
	a.Value = a.Value + 1
}

func (a *CountData) clone() AggregationData {
	return &CountData{Value: a.Value}
}

func (a *CountData) equal(other AggregationData) bool {
	a2, ok := other.(*CountData)
	if !ok {
		return false
	}

	return a.Value == a2.Value
}

func (a *CountData) toPoint(metricType metricdata.Type, t time.Time) metricdata.Point {
	switch metricType {
	case metricdata.TypeCumulativeInt64:
		return metricdata.NewInt64Point(t, a.Value)
	default:
		panic("unsupported metricdata.Type")
	}
}

// SumData is the aggregated data for the Sum aggregation.
// A sum aggregation processes data and sums up the recordings.
//
// Most users won't directly access sum data.
type SumData struct {
	Value float64
}

func (a *SumData) isAggregationData() bool { return true }

func (a *SumData) addSample(v float64, _ map[string]interface{}, _ time.Time) {
	a.Value += v
}

func (a *SumData) clone() AggregationData {
	return &SumData{Value: a.Value}
}

func (a *SumData) equal(other AggregationData) bool {
	a2, ok := other.(*SumData)
	if !ok {
		return false
	}
	return math.Pow(a.Value-a2.Value, 2) < epsilon
}

func (a *SumData) toPoint(metricType metricdata.Type, t time.Time) metricdata.Point {
	switch metricType {
	case metricdata.TypeCumulativeInt64:
		return metricdata.NewInt64Point(t, int64(a.Value))
	case metricdata.TypeCumulativeFloat64:
		return metricdata.NewFloat64Point(t, a.Value)
	default:
		panic("unsupported metricdata.Type")
	}
}

// DistributionData is the aggregated data for the
// Distribution aggregation.
//
// Most users won't directly access distribution data.
//
// For a distribution with N bounds, the associated DistributionData will have
// N+1 buckets.
type DistributionData struct {
	Count           int64   // number of data points aggregated
	Min             float64 // minimum value in the distribution
	Max             float64 // max value in the distribution
	Mean            float64 // mean of the distribution
	SumOfSquaredDev float64 // sum of the squared deviation from the mean
	CountPerBucket  []int64 // number of occurrences per bucket
	// ExemplarsPerBucket is slice the same length as CountPerBucket containing
	// an exemplar for the associated bucket, or nil.
	ExemplarsPerBucket []*metricdata.Exemplar
	bounds             []float64 // histogram distribution of the values
}

func newDistributionData(agg *Aggregation) *DistributionData {
	bucketCount := len(agg.Buckets) + 1
	return &DistributionData{
		CountPerBucket:     make([]int64, bucketCount),
		ExemplarsPerBucket: make([]*metricdata.Exemplar, bucketCount),
		bounds:             agg.Buckets,
		Min:                math.MaxFloat64,
		Max:                math.SmallestNonzeroFloat64,
	}
}

// Sum returns the sum of all samples collected.
func (a *DistributionData) Sum() float64 { return a.Mean * float64(a.Count) }

func (a *DistributionData) variance() float64 {
	if a.Count <= 1 {
		return 0
	}
	return a.SumOfSquaredDev / float64(a.Count-1)
}

func (a *DistributionData) isAggregationData() bool { return true }

// TODO(songy23): support exemplar attachments.
func (a *DistributionData) addSample(v float64, attachments map[string]interface{}, t time.Time) {
	if v < a.Min {
		a.Min = v
	}
	if v > a.Max {
		a.Max = v
	}
	a.Count++
	a.addToBucket(v, attachments, t)

	if a.Count == 1 {
		a.Mean = v
		return
	}

	oldMean := a.Mean
	a.Mean = a.Mean + (v-a.Mean)/float64(a.Count)
	a.SumOfSquaredDev = a.SumOfSquaredDev + (v-oldMean)*(v-a.Mean)
}

func (a *DistributionData) addToBucket(v float64, attachments map[string]interface{}, t time.Time) {
	var count *int64
	var i int
	var b float64
	for i, b = range a.bounds {
		if v < b {
			count = &a.CountPerBucket[i]
			break
		}
	}
	if count == nil { // Last bucket.
		i = len(a.bounds)
		count = &a.CountPerBucket[i]
	}
	*count++
	if exemplar := getExemplar(v, attachments, t); exemplar != nil {
		a.ExemplarsPerBucket[i] = exemplar
	}
}

func getExemplar(v float64, attachments map[string]interface{}, t time.Time) *metricdata.Exemplar {
	if len(attachments) == 0 {
		return nil
	}
	return &metricdata.Exemplar{
		Value:       v,
		Timestamp:   t,
		Attachments: attachments,
	}
}

func (a *DistributionData) clone() AggregationData {
	c := *a
	c.CountPerBucket = append([]int64(nil), a.CountPerBucket...)
	c.ExemplarsPerBucket = append([]*metricdata.Exemplar(nil), a.ExemplarsPerBucket...)
	return &c
}

func (a *DistributionData) equal(other AggregationData) bool {
	a2, ok := other.(*DistributionData)
	if !ok {
		return false
	}
	if a2 == nil {
		return false
	}
	if len(a.CountPerBucket) != len(a2.CountPerBucket) {
		return false
	}
	for i := range a.CountPerBucket {
		if a.CountPerBucket[i] != a2.CountPerBucket[i] {
			return false
		}
	}
	return a.Count == a2.Count && a.Min == a2.Min && a.Max == a2.Max && math.Pow(a.Mean-a2.Mean, 2) < epsilon && math.Pow(a.variance()-a2.variance(), 2) < epsilon
}

func (a *DistributionData) toPoint(metricType metricdata.Type, t time.Time) metricdata.Point {
	switch metricType {
	case metricdata.TypeCumulativeDistribution:
		buckets := []metricdata.Bucket{}
		for i := 0; i < len(a.CountPerBucket); i++ {
			buckets = append(buckets, metricdata.Bucket{
				Count:    a.CountPerBucket[i],
				Exemplar: a.ExemplarsPerBucket[i],
			})
		}
		bucketOptions := &metricdata.BucketOptions{Bounds: a.bounds}

		val := &metricdata.Distribution{
			Count:                 a.Count,
			Sum:                   a.Sum(),
			SumOfSquaredDeviation: a.SumOfSquaredDev,
			BucketOptions:         bucketOptions,
			Buckets:               buckets,
		}
		return metricdata.NewDistributionPoint(t, val)

	default:
		// TODO: [rghetia] when we have a use case for TypeGaugeDistribution.
		panic("unsupported metricdata.Type")
	}
}

// LastValueData returns the last value recorded for LastValue aggregation.
type LastValueData struct {
	Value float64
}

func (l *LastValueData) isAggregationData() bool {
	return true
}

func (l *LastValueData) addSample(v float64, _ map[string]interface{}, _ time.Time) {
	l.Value = v
}

func (l *LastValueData) clone() AggregationData {
	return &LastValueData{l.Value}
}

func (l *LastValueData) equal(other AggregationData) bool {
	a2, ok := other.(*LastValueData)
	if !ok {
		return false
	}
	return l.Value == a2.Value
}

func (l *LastValueData) toPoint(metricType metricdata.Type, t time.Time) metricdata.Point {
	switch metricType {
	case metricdata.TypeGaugeInt64:
		return metricdata.NewInt64Point(t, int64(l.Value))
	case metricdata.TypeGaugeFloat64:
		return metricdata.NewFloat64Point(t, l.Value)
	default:
		panic("unsupported metricdata.Type")
	}
}
