// Copyright The OpenTelemetry Authors
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

package exact // import "go.opentelemetry.io/otel/sdk/metric/aggregator/exact"

import (
	"context"
	"sync"
	"time"

	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/number"
	export "go.opentelemetry.io/otel/sdk/export/metric"
	"go.opentelemetry.io/otel/sdk/export/metric/aggregation"
	"go.opentelemetry.io/otel/sdk/metric/aggregator"
)

type (
	// Aggregator aggregates events that form a distribution, keeping
	// an array with the exact set of values.
	Aggregator struct {
		lock    sync.Mutex
		samples []aggregation.Point
	}
)

var _ export.Aggregator = &Aggregator{}
var _ aggregation.Points = &Aggregator{}
var _ aggregation.Count = &Aggregator{}

// New returns cnt many new exact aggregators, which aggregate recorded
// measurements by storing them in an array.  This type uses a mutex
// for Update() and SynchronizedMove() concurrency.
func New(cnt int) []Aggregator {
	return make([]Aggregator, cnt)
}

// Aggregation returns an interface for reading the state of this aggregator.
func (c *Aggregator) Aggregation() aggregation.Aggregation {
	return c
}

// Kind returns aggregation.ExactKind.
func (c *Aggregator) Kind() aggregation.Kind {
	return aggregation.ExactKind
}

// Count returns the number of values in the checkpoint.
func (c *Aggregator) Count() (uint64, error) {
	return uint64(len(c.samples)), nil
}

// Points returns access to the raw data set.
func (c *Aggregator) Points() ([]aggregation.Point, error) {
	return c.samples, nil
}

// SynchronizedMove saves the current state to oa and resets the current state to
// the empty set, taking a lock to prevent concurrent Update() calls.
func (c *Aggregator) SynchronizedMove(oa export.Aggregator, desc *metric.Descriptor) error {
	o, _ := oa.(*Aggregator)

	if oa != nil && o == nil {
		return aggregator.NewInconsistentAggregatorError(c, oa)
	}

	c.lock.Lock()
	defer c.lock.Unlock()

	if o != nil {
		o.samples = c.samples
	}
	c.samples = nil

	return nil
}

// Update adds the recorded measurement to the current data set.
// Update takes a lock to prevent concurrent Update() and SynchronizedMove()
// calls.
func (c *Aggregator) Update(_ context.Context, number number.Number, desc *metric.Descriptor) error {
	now := time.Now()
	c.lock.Lock()
	defer c.lock.Unlock()
	c.samples = append(c.samples, aggregation.Point{
		Number: number,
		Time:   now,
	})

	return nil
}

// Merge combines two data sets into one.
func (c *Aggregator) Merge(oa export.Aggregator, desc *metric.Descriptor) error {
	o, _ := oa.(*Aggregator)
	if o == nil {
		return aggregator.NewInconsistentAggregatorError(c, oa)
	}

	c.samples = combine(c.samples, o.samples)
	return nil
}

func combine(a, b []aggregation.Point) []aggregation.Point {
	result := make([]aggregation.Point, 0, len(a)+len(b))

	for len(a) != 0 && len(b) != 0 {
		if a[0].Time.Before(b[0].Time) {
			result = append(result, a[0])
			a = a[1:]
		} else {
			result = append(result, b[0])
			b = b[1:]
		}
	}
	result = append(result, a...)
	result = append(result, b...)
	return result
}
