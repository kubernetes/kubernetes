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

package metric // import "go.opentelemetry.io/otel/metric"

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric/number"
)

// MeterImpl is the interface an SDK must implement to supply a Meter
// implementation.
type MeterImpl interface {
	// RecordBatch atomically records a batch of measurements.
	RecordBatch(ctx context.Context, labels []attribute.KeyValue, measurement ...Measurement)

	// NewSyncInstrument returns a newly constructed
	// synchronous instrument implementation or an error, should
	// one occur.
	NewSyncInstrument(descriptor Descriptor) (SyncImpl, error)

	// NewAsyncInstrument returns a newly constructed
	// asynchronous instrument implementation or an error, should
	// one occur.
	NewAsyncInstrument(
		descriptor Descriptor,
		runner AsyncRunner,
	) (AsyncImpl, error)
}

// InstrumentImpl is a common interface for synchronous and
// asynchronous instruments.
type InstrumentImpl interface {
	// Implementation returns the underlying implementation of the
	// instrument, which allows the implementation to gain access
	// to its own representation especially from a `Measurement`.
	Implementation() interface{}

	// Descriptor returns a copy of the instrument's Descriptor.
	Descriptor() Descriptor
}

// SyncImpl is the implementation-level interface to a generic
// synchronous instrument (e.g., ValueRecorder and Counter instruments).
type SyncImpl interface {
	InstrumentImpl

	// Bind creates an implementation-level bound instrument,
	// binding a label set with this instrument implementation.
	Bind(labels []attribute.KeyValue) BoundSyncImpl

	// RecordOne captures a single synchronous metric event.
	RecordOne(ctx context.Context, number number.Number, labels []attribute.KeyValue)
}

// BoundSyncImpl is the implementation-level interface to a
// generic bound synchronous instrument
type BoundSyncImpl interface {

	// RecordOne captures a single synchronous metric event.
	RecordOne(ctx context.Context, number number.Number)

	// Unbind frees the resources associated with this bound instrument. It
	// does not affect the metric this bound instrument was created through.
	Unbind()
}

// AsyncImpl is an implementation-level interface to an
// asynchronous instrument (e.g., Observer instruments).
type AsyncImpl interface {
	InstrumentImpl
}

// WrapMeterImpl constructs a `Meter` implementation from a
// `MeterImpl` implementation.
func WrapMeterImpl(impl MeterImpl, instrumentationName string, opts ...MeterOption) Meter {
	return Meter{
		impl:    impl,
		name:    instrumentationName,
		version: NewMeterConfig(opts...).InstrumentationVersion,
	}
}
