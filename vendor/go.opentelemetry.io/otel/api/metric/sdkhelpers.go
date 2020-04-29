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

package metric

import (
	"context"

	"go.opentelemetry.io/otel/api/core"
)

// MeterImpl is a convenient interface for SDK and test
// implementations that would provide a `Meter` but do not wish to
// re-implement the API's type-safe interfaces.  Helpers provided in
// this package will construct a `Meter` given a `MeterImpl`.
type MeterImpl interface {
	// RecordBatch atomically records a batch of measurements.
	RecordBatch(context.Context, []core.KeyValue, ...Measurement)

	// NewSyncInstrument returns a newly constructed
	// synchronous instrument implementation or an error, should
	// one occur.
	NewSyncInstrument(descriptor Descriptor) (SyncImpl, error)

	// NewAsyncInstrument returns a newly constructed
	// asynchronous instrument implementation or an error, should
	// one occur.
	NewAsyncInstrument(
		descriptor Descriptor,
		callback func(func(core.Number, []core.KeyValue)),
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
// synchronous instrument (e.g., Measure and Counter instruments).
type SyncImpl interface {
	InstrumentImpl

	// Bind creates an implementation-level bound instrument,
	// binding a label set with this instrument implementation.
	Bind(labels []core.KeyValue) BoundSyncImpl

	// RecordOne captures a single synchronous metric event.
	RecordOne(ctx context.Context, number core.Number, labels []core.KeyValue)
}

// BoundSyncImpl is the implementation-level interface to a
// generic bound synchronous instrument
type BoundSyncImpl interface {

	// RecordOne captures a single synchronous metric event.
	RecordOne(ctx context.Context, number core.Number)

	// Unbind frees the resources associated with this bound instrument. It
	// does not affect the metric this bound instrument was created through.
	Unbind()
}

// AsyncImpl is an implementation-level interface to an
// asynchronous instrument (e.g., Observer instruments).
type AsyncImpl interface {
	InstrumentImpl

	// Note: An `Unregister()` API could be supported here.
}

// wrappedMeterImpl implements the `Meter` interface given a
// `MeterImpl` implementation.
type wrappedMeterImpl struct {
	impl        MeterImpl
	libraryName string
}

// int64ObserverResult is an adapter for int64-valued asynchronous
// callbacks.
type int64ObserverResult struct {
	observe func(core.Number, []core.KeyValue)
}

// float64ObserverResult is an adapter for float64-valued asynchronous
// callbacks.
type float64ObserverResult struct {
	observe func(core.Number, []core.KeyValue)
}

var (
	_ Meter                 = (*wrappedMeterImpl)(nil)
	_ Int64ObserverResult   = int64ObserverResult{}
	_ Float64ObserverResult = float64ObserverResult{}
)

// Configure is a helper that applies all the options to a Config.
func Configure(opts []Option) Config {
	var config Config
	for _, o := range opts {
		o.Apply(&config)
	}
	return config
}

// WrapMeterImpl constructs a `Meter` implementation from a
// `MeterImpl` implementation.
func WrapMeterImpl(impl MeterImpl, libraryName string) Meter {
	return &wrappedMeterImpl{
		impl:        impl,
		libraryName: libraryName,
	}
}

func (m *wrappedMeterImpl) RecordBatch(ctx context.Context, ls []core.KeyValue, ms ...Measurement) {
	m.impl.RecordBatch(ctx, ls, ms...)
}

func (m *wrappedMeterImpl) newSync(name string, metricKind Kind, numberKind core.NumberKind, opts []Option) (SyncImpl, error) {
	desc := NewDescriptor(name, metricKind, numberKind, opts...)
	desc.config.LibraryName = m.libraryName
	return m.impl.NewSyncInstrument(desc)
}

func (m *wrappedMeterImpl) NewInt64Counter(name string, opts ...Option) (Int64Counter, error) {
	return WrapInt64CounterInstrument(
		m.newSync(name, CounterKind, core.Int64NumberKind, opts))
}

// WrapInt64CounterInstrument returns an `Int64Counter` from a
// `SyncImpl`.  An error will be generated if the
// `SyncImpl` is nil (in which case a No-op is substituted),
// otherwise the error passes through.
func WrapInt64CounterInstrument(syncInst SyncImpl, err error) (Int64Counter, error) {
	common, err := checkNewSync(syncInst, err)
	return Int64Counter{syncInstrument: common}, err
}

func (m *wrappedMeterImpl) NewFloat64Counter(name string, opts ...Option) (Float64Counter, error) {
	return WrapFloat64CounterInstrument(
		m.newSync(name, CounterKind, core.Float64NumberKind, opts))
}

// WrapFloat64CounterInstrument returns an `Float64Counter` from a
// `SyncImpl`.  An error will be generated if the
// `SyncImpl` is nil (in which case a No-op is substituted),
// otherwise the error passes through.
func WrapFloat64CounterInstrument(syncInst SyncImpl, err error) (Float64Counter, error) {
	common, err := checkNewSync(syncInst, err)
	return Float64Counter{syncInstrument: common}, err
}

func (m *wrappedMeterImpl) NewInt64Measure(name string, opts ...Option) (Int64Measure, error) {
	return WrapInt64MeasureInstrument(
		m.newSync(name, MeasureKind, core.Int64NumberKind, opts))
}

// WrapInt64MeasureInstrument returns an `Int64Measure` from a
// `SyncImpl`.  An error will be generated if the
// `SyncImpl` is nil (in which case a No-op is substituted),
// otherwise the error passes through.
func WrapInt64MeasureInstrument(syncInst SyncImpl, err error) (Int64Measure, error) {
	common, err := checkNewSync(syncInst, err)
	return Int64Measure{syncInstrument: common}, err
}

func (m *wrappedMeterImpl) NewFloat64Measure(name string, opts ...Option) (Float64Measure, error) {
	return WrapFloat64MeasureInstrument(
		m.newSync(name, MeasureKind, core.Float64NumberKind, opts))
}

// WrapFloat64MeasureInstrument returns an `Float64Measure` from a
// `SyncImpl`.  An error will be generated if the
// `SyncImpl` is nil (in which case a No-op is substituted),
// otherwise the error passes through.
func WrapFloat64MeasureInstrument(syncInst SyncImpl, err error) (Float64Measure, error) {
	common, err := checkNewSync(syncInst, err)
	return Float64Measure{syncInstrument: common}, err
}

func (m *wrappedMeterImpl) newAsync(name string, mkind Kind, nkind core.NumberKind, opts []Option, callback func(func(core.Number, []core.KeyValue))) (AsyncImpl, error) {
	desc := NewDescriptor(name, mkind, nkind, opts...)
	desc.config.LibraryName = m.libraryName
	return m.impl.NewAsyncInstrument(desc, callback)
}

func (m *wrappedMeterImpl) RegisterInt64Observer(name string, callback Int64ObserverCallback, opts ...Option) (Int64Observer, error) {
	if callback == nil {
		return NoopMeter{}.RegisterInt64Observer("", nil)
	}
	return WrapInt64ObserverInstrument(
		m.newAsync(name, ObserverKind, core.Int64NumberKind, opts,
			func(observe func(core.Number, []core.KeyValue)) {
				// Note: this memory allocation could be avoided by
				// using a pointer to this object and mutating it
				// on each collection interval.
				callback(int64ObserverResult{observe})
			}))
}

// WrapInt64ObserverInstrument returns an `Int64Observer` from a
// `AsyncImpl`.  An error will be generated if the
// `AsyncImpl` is nil (in which case a No-op is substituted),
// otherwise the error passes through.
func WrapInt64ObserverInstrument(asyncInst AsyncImpl, err error) (Int64Observer, error) {
	common, err := checkNewAsync(asyncInst, err)
	return Int64Observer{asyncInstrument: common}, err
}

func (m *wrappedMeterImpl) RegisterFloat64Observer(name string, callback Float64ObserverCallback, opts ...Option) (Float64Observer, error) {
	if callback == nil {
		return NoopMeter{}.RegisterFloat64Observer("", nil)
	}
	return WrapFloat64ObserverInstrument(
		m.newAsync(name, ObserverKind, core.Float64NumberKind, opts,
			func(observe func(core.Number, []core.KeyValue)) {
				callback(float64ObserverResult{observe})
			}))
}

// WrapFloat64ObserverInstrument returns an `Float64Observer` from a
// `AsyncImpl`.  An error will be generated if the
// `AsyncImpl` is nil (in which case a No-op is substituted),
// otherwise the error passes through.
func WrapFloat64ObserverInstrument(asyncInst AsyncImpl, err error) (Float64Observer, error) {
	common, err := checkNewAsync(asyncInst, err)
	return Float64Observer{asyncInstrument: common}, err
}

func (io int64ObserverResult) Observe(value int64, labels ...core.KeyValue) {
	io.observe(core.NewInt64Number(value), labels)
}

func (fo float64ObserverResult) Observe(value float64, labels ...core.KeyValue) {
	fo.observe(core.NewFloat64Number(value), labels)
}
