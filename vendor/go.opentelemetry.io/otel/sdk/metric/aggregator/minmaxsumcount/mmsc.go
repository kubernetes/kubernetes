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

package minmaxsumcount // import "go.opentelemetry.io/otel/sdk/metric/aggregator/minmaxsumcount"

import (
	"context"
	"sync"

	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/number"
	export "go.opentelemetry.io/otel/sdk/export/metric"
	"go.opentelemetry.io/otel/sdk/export/metric/aggregation"
	"go.opentelemetry.io/otel/sdk/metric/aggregator"
)

type (
	// Aggregator aggregates events that form a distribution,
	// keeping only the min, max, sum, and count.
	Aggregator struct {
		lock sync.Mutex
		kind number.Kind
		state
	}

	state struct {
		sum   number.Number
		min   number.Number
		max   number.Number
		count uint64
	}
)

var _ export.Aggregator = &Aggregator{}
var _ aggregation.MinMaxSumCount = &Aggregator{}

// New returns a new aggregator for computing the min, max, sum, and
// count.
//
// This type uses a mutex for Update() and SynchronizedMove() concurrency.
func New(cnt int, desc *metric.Descriptor) []Aggregator {
	kind := desc.NumberKind()
	aggs := make([]Aggregator, cnt)
	for i := range aggs {
		aggs[i] = Aggregator{
			kind:  kind,
			state: emptyState(kind),
		}
	}
	return aggs
}

// Aggregation returns an interface for reading the state of this aggregator.
func (c *Aggregator) Aggregation() aggregation.Aggregation {
	return c
}

// Kind returns aggregation.MinMaxSumCountKind.
func (c *Aggregator) Kind() aggregation.Kind {
	return aggregation.MinMaxSumCountKind
}

// Sum returns the sum of values in the checkpoint.
func (c *Aggregator) Sum() (number.Number, error) {
	return c.sum, nil
}

// Count returns the number of values in the checkpoint.
func (c *Aggregator) Count() (uint64, error) {
	return c.count, nil
}

// Min returns the minimum value in the checkpoint.
// The error value aggregation.ErrNoData will be returned
// if there were no measurements recorded during the checkpoint.
func (c *Aggregator) Min() (number.Number, error) {
	if c.count == 0 {
		return 0, aggregation.ErrNoData
	}
	return c.min, nil
}

// Max returns the maximum value in the checkpoint.
// The error value aggregation.ErrNoData will be returned
// if there were no measurements recorded during the checkpoint.
func (c *Aggregator) Max() (number.Number, error) {
	if c.count == 0 {
		return 0, aggregation.ErrNoData
	}
	return c.max, nil
}

// SynchronizedMove saves the current state into oa and resets the current state to
// the empty set.
func (c *Aggregator) SynchronizedMove(oa export.Aggregator, desc *metric.Descriptor) error {
	o, _ := oa.(*Aggregator)

	if oa != nil && o == nil {
		return aggregator.NewInconsistentAggregatorError(c, oa)
	}
	c.lock.Lock()
	if o != nil {
		o.state = c.state
	}
	c.state = emptyState(c.kind)
	c.lock.Unlock()

	return nil
}

func emptyState(kind number.Kind) state {
	return state{
		count: 0,
		sum:   0,
		min:   kind.Maximum(),
		max:   kind.Minimum(),
	}
}

// Update adds the recorded measurement to the current data set.
func (c *Aggregator) Update(_ context.Context, number number.Number, desc *metric.Descriptor) error {
	kind := desc.NumberKind()

	c.lock.Lock()
	defer c.lock.Unlock()
	c.count++
	c.sum.AddNumber(kind, number)
	if number.CompareNumber(kind, c.min) < 0 {
		c.min = number
	}
	if number.CompareNumber(kind, c.max) > 0 {
		c.max = number
	}
	return nil
}

// Merge combines two data sets into one.
func (c *Aggregator) Merge(oa export.Aggregator, desc *metric.Descriptor) error {
	o, _ := oa.(*Aggregator)
	if o == nil {
		return aggregator.NewInconsistentAggregatorError(c, oa)
	}

	c.count += o.count
	c.sum.AddNumber(desc.NumberKind(), o.sum)

	if c.min.CompareNumber(desc.NumberKind(), o.min) > 0 {
		c.min.SetNumber(o.min)
	}
	if c.max.CompareNumber(desc.NumberKind(), o.max) < 0 {
		c.max.SetNumber(o.max)
	}
	return nil
}
