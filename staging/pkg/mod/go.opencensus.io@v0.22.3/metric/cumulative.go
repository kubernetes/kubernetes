// Copyright 2019, OpenCensus Authors
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

package metric

import (
	"math"
	"sync/atomic"
	"time"

	"go.opencensus.io/metric/metricdata"
)

// Float64Cumulative represents a float64 value that can only go up.
//
// Float64Cumulative maintains a float64 value for each combination of label values
// passed to the Set or Inc methods.
type Float64Cumulative struct {
	bm baseMetric
}

// Float64CumulativeEntry represents a single value of the cumulative corresponding to a set
// of label values.
type Float64CumulativeEntry struct {
	val uint64 // needs to be uint64 for atomic access, interpret with math.Float64frombits
}

func (e *Float64CumulativeEntry) read(t time.Time) metricdata.Point {
	v := math.Float64frombits(atomic.LoadUint64(&e.val))
	if v < 0 {
		v = 0
	}
	return metricdata.NewFloat64Point(t, v)
}

// GetEntry returns a cumulative entry where each key for this cumulative has the value
// given.
//
// The number of label values supplied must be exactly the same as the number
// of keys supplied when this cumulative was created.
func (c *Float64Cumulative) GetEntry(labelVals ...metricdata.LabelValue) (*Float64CumulativeEntry, error) {
	entry, err := c.bm.entryForValues(labelVals, func() baseEntry {
		return &Float64CumulativeEntry{}
	})
	if err != nil {
		return nil, err
	}
	return entry.(*Float64CumulativeEntry), nil
}

// Inc increments the cumulative entry value by val. It returns without incrementing if the val
// is negative.
func (e *Float64CumulativeEntry) Inc(val float64) {
	var swapped bool
	if val <= 0.0 {
		return
	}
	for !swapped {
		oldVal := atomic.LoadUint64(&e.val)
		newVal := math.Float64bits(math.Float64frombits(oldVal) + val)
		swapped = atomic.CompareAndSwapUint64(&e.val, oldVal, newVal)
	}
}

// Int64Cumulative represents a int64 cumulative value that can only go up.
//
// Int64Cumulative maintains an int64 value for each combination of label values passed to the
// Set or Inc methods.
type Int64Cumulative struct {
	bm baseMetric
}

// Int64CumulativeEntry represents a single value of the cumulative corresponding to a set
// of label values.
type Int64CumulativeEntry struct {
	val int64
}

func (e *Int64CumulativeEntry) read(t time.Time) metricdata.Point {
	v := atomic.LoadInt64(&e.val)
	if v < 0 {
		v = 0.0
	}
	return metricdata.NewInt64Point(t, v)
}

// GetEntry returns a cumulative entry where each key for this cumulative has the value
// given.
//
// The number of label values supplied must be exactly the same as the number
// of keys supplied when this cumulative was created.
func (c *Int64Cumulative) GetEntry(labelVals ...metricdata.LabelValue) (*Int64CumulativeEntry, error) {
	entry, err := c.bm.entryForValues(labelVals, func() baseEntry {
		return &Int64CumulativeEntry{}
	})
	if err != nil {
		return nil, err
	}
	return entry.(*Int64CumulativeEntry), nil
}

// Inc increments the current cumulative entry value by val. It returns without incrementing if
// the val is negative.
func (e *Int64CumulativeEntry) Inc(val int64) {
	if val <= 0 {
		return
	}
	atomic.AddInt64(&e.val, val)
}

// Int64DerivedCumulative represents int64 cumulative value that is derived from an object.
//
// Int64DerivedCumulative maintains objects for each combination of label values.
// These objects implement Int64DerivedCumulativeInterface to read instantaneous value
// representing the object.
type Int64DerivedCumulative struct {
	bm baseMetric
}

type int64DerivedCumulativeEntry struct {
	fn func() int64
}

func (e *int64DerivedCumulativeEntry) read(t time.Time) metricdata.Point {
	// TODO: [rghetia] handle a condition where new value return by fn is lower than previous call.
	// It requires that we maintain the old values.
	return metricdata.NewInt64Point(t, e.fn())
}

// UpsertEntry inserts or updates a derived cumulative entry for the given set of label values.
// The object for which this cumulative entry is inserted or updated, must implement func() int64
//
// It returns an error if
// 1. The number of label values supplied are not the same as the number
// of keys supplied when this cumulative was created.
// 2. fn func() int64 is nil.
func (c *Int64DerivedCumulative) UpsertEntry(fn func() int64, labelVals ...metricdata.LabelValue) error {
	if fn == nil {
		return errInvalidParam
	}
	return c.bm.upsertEntry(labelVals, func() baseEntry {
		return &int64DerivedCumulativeEntry{fn}
	})
}

// Float64DerivedCumulative represents float64 cumulative value that is derived from an object.
//
// Float64DerivedCumulative maintains objects for each combination of label values.
// These objects implement Float64DerivedCumulativeInterface to read instantaneous value
// representing the object.
type Float64DerivedCumulative struct {
	bm baseMetric
}

type float64DerivedCumulativeEntry struct {
	fn func() float64
}

func (e *float64DerivedCumulativeEntry) read(t time.Time) metricdata.Point {
	return metricdata.NewFloat64Point(t, e.fn())
}

// UpsertEntry inserts or updates a derived cumulative entry for the given set of label values.
// The object for which this cumulative entry is inserted or updated, must implement func() float64
//
// It returns an error if
// 1. The number of label values supplied are not the same as the number
// of keys supplied when this cumulative was created.
// 2. fn func() float64 is nil.
func (c *Float64DerivedCumulative) UpsertEntry(fn func() float64, labelVals ...metricdata.LabelValue) error {
	if fn == nil {
		return errInvalidParam
	}
	return c.bm.upsertEntry(labelVals, func() baseEntry {
		return &float64DerivedCumulativeEntry{fn}
	})
}
