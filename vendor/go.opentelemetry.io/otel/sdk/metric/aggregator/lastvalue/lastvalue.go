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

package lastvalue // import "go.opentelemetry.io/otel/sdk/metric/aggregator/lastvalue"

import (
	"context"
	"sync/atomic"
	"time"
	"unsafe"

	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/number"
	export "go.opentelemetry.io/otel/sdk/export/metric"
	"go.opentelemetry.io/otel/sdk/export/metric/aggregation"
	"go.opentelemetry.io/otel/sdk/metric/aggregator"
)

type (

	// Aggregator aggregates lastValue events.
	Aggregator struct {
		// value is an atomic pointer to *lastValueData.  It is never nil.
		value unsafe.Pointer
	}

	// lastValueData stores the current value of a lastValue along with
	// a sequence number to determine the winner of a race.
	lastValueData struct {
		// value is the int64- or float64-encoded Set() data
		//
		// value needs to be aligned for 64-bit atomic operations.
		value number.Number

		// timestamp indicates when this record was submitted.
		// this can be used to pick a winner when multiple
		// records contain lastValue data for the same labels due
		// to races.
		timestamp time.Time
	}
)

var _ export.Aggregator = &Aggregator{}
var _ aggregation.LastValue = &Aggregator{}

// An unset lastValue has zero timestamp and zero value.
var unsetLastValue = &lastValueData{}

// New returns a new lastValue aggregator.  This aggregator retains the
// last value and timestamp that were recorded.
func New(cnt int) []Aggregator {
	aggs := make([]Aggregator, cnt)
	for i := range aggs {
		aggs[i] = Aggregator{
			value: unsafe.Pointer(unsetLastValue),
		}
	}
	return aggs
}

// Aggregation returns an interface for reading the state of this aggregator.
func (g *Aggregator) Aggregation() aggregation.Aggregation {
	return g
}

// Kind returns aggregation.LastValueKind.
func (g *Aggregator) Kind() aggregation.Kind {
	return aggregation.LastValueKind
}

// LastValue returns the last-recorded lastValue value and the
// corresponding timestamp.  The error value aggregation.ErrNoData
// will be returned if (due to a race condition) the checkpoint was
// computed before the first value was set.
func (g *Aggregator) LastValue() (number.Number, time.Time, error) {
	gd := (*lastValueData)(g.value)
	if gd == unsetLastValue {
		return 0, time.Time{}, aggregation.ErrNoData
	}
	return gd.value.AsNumber(), gd.timestamp, nil
}

// SynchronizedMove atomically saves the current value.
func (g *Aggregator) SynchronizedMove(oa export.Aggregator, _ *metric.Descriptor) error {
	if oa == nil {
		atomic.StorePointer(&g.value, unsafe.Pointer(unsetLastValue))
		return nil
	}
	o, _ := oa.(*Aggregator)
	if o == nil {
		return aggregator.NewInconsistentAggregatorError(g, oa)
	}
	o.value = atomic.SwapPointer(&g.value, unsafe.Pointer(unsetLastValue))
	return nil
}

// Update atomically sets the current "last" value.
func (g *Aggregator) Update(_ context.Context, number number.Number, desc *metric.Descriptor) error {
	ngd := &lastValueData{
		value:     number,
		timestamp: time.Now(),
	}
	atomic.StorePointer(&g.value, unsafe.Pointer(ngd))
	return nil
}

// Merge combines state from two aggregators.  The most-recently set
// value is chosen.
func (g *Aggregator) Merge(oa export.Aggregator, desc *metric.Descriptor) error {
	o, _ := oa.(*Aggregator)
	if o == nil {
		return aggregator.NewInconsistentAggregatorError(g, oa)
	}

	ggd := (*lastValueData)(atomic.LoadPointer(&g.value))
	ogd := (*lastValueData)(atomic.LoadPointer(&o.value))

	if ggd.timestamp.After(ogd.timestamp) {
		return nil
	}

	g.value = unsafe.Pointer(ogd)
	return nil
}
