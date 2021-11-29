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

package sum // import "go.opentelemetry.io/otel/sdk/metric/aggregator/sum"

import (
	"context"

	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/number"
	export "go.opentelemetry.io/otel/sdk/export/metric"
	"go.opentelemetry.io/otel/sdk/export/metric/aggregation"
	"go.opentelemetry.io/otel/sdk/metric/aggregator"
)

// Aggregator aggregates counter events.
type Aggregator struct {
	// current holds current increments to this counter record
	// current needs to be aligned for 64-bit atomic operations.
	value number.Number
}

var _ export.Aggregator = &Aggregator{}
var _ export.Subtractor = &Aggregator{}
var _ aggregation.Sum = &Aggregator{}

// New returns a new counter aggregator implemented by atomic
// operations.  This aggregator implements the aggregation.Sum
// export interface.
func New(cnt int) []Aggregator {
	return make([]Aggregator, cnt)
}

// Aggregation returns an interface for reading the state of this aggregator.
func (c *Aggregator) Aggregation() aggregation.Aggregation {
	return c
}

// Kind returns aggregation.SumKind.
func (c *Aggregator) Kind() aggregation.Kind {
	return aggregation.SumKind
}

// Sum returns the last-checkpointed sum.  This will never return an
// error.
func (c *Aggregator) Sum() (number.Number, error) {
	return c.value, nil
}

// SynchronizedMove atomically saves the current value into oa and resets the
// current sum to zero.
func (c *Aggregator) SynchronizedMove(oa export.Aggregator, _ *metric.Descriptor) error {
	if oa == nil {
		c.value.SetRawAtomic(0)
		return nil
	}
	o, _ := oa.(*Aggregator)
	if o == nil {
		return aggregator.NewInconsistentAggregatorError(c, oa)
	}
	o.value = c.value.SwapNumberAtomic(number.Number(0))
	return nil
}

// Update atomically adds to the current value.
func (c *Aggregator) Update(_ context.Context, num number.Number, desc *metric.Descriptor) error {
	c.value.AddNumberAtomic(desc.NumberKind(), num)
	return nil
}

// Merge combines two counters by adding their sums.
func (c *Aggregator) Merge(oa export.Aggregator, desc *metric.Descriptor) error {
	o, _ := oa.(*Aggregator)
	if o == nil {
		return aggregator.NewInconsistentAggregatorError(c, oa)
	}
	c.value.AddNumber(desc.NumberKind(), o.value)
	return nil
}

func (c *Aggregator) Subtract(opAgg, resAgg export.Aggregator, descriptor *metric.Descriptor) error {
	op, _ := opAgg.(*Aggregator)
	if op == nil {
		return aggregator.NewInconsistentAggregatorError(c, opAgg)
	}

	res, _ := resAgg.(*Aggregator)
	if res == nil {
		return aggregator.NewInconsistentAggregatorError(c, resAgg)
	}

	res.value = c.value
	res.value.AddNumber(descriptor.NumberKind(), number.NewNumberSignChange(descriptor.NumberKind(), op.value))
	return nil
}
