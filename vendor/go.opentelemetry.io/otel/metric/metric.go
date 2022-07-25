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
	"go.opentelemetry.io/otel/unit"
)

// MeterProvider supports named Meter instances.
type MeterProvider interface {
	// Meter creates an implementation of the Meter interface.
	// The instrumentationName must be the name of the library providing
	// instrumentation. This name may be the same as the instrumented code
	// only if that code provides built-in instrumentation. If the
	// instrumentationName is empty, then a implementation defined default
	// name will be used instead.
	Meter(instrumentationName string, opts ...MeterOption) Meter
}

// Meter is the creator of metric instruments.
//
// An uninitialized Meter is a no-op implementation.
type Meter struct {
	impl          MeterImpl
	name, version string
}

// RecordBatch atomically records a batch of measurements.
func (m Meter) RecordBatch(ctx context.Context, ls []attribute.KeyValue, ms ...Measurement) {
	if m.impl == nil {
		return
	}
	m.impl.RecordBatch(ctx, ls, ms...)
}

// NewBatchObserver creates a new BatchObserver that supports
// making batches of observations for multiple instruments.
func (m Meter) NewBatchObserver(callback BatchObserverFunc) BatchObserver {
	return BatchObserver{
		meter:  m,
		runner: newBatchAsyncRunner(callback),
	}
}

// NewInt64Counter creates a new integer Counter instrument with the
// given name, customized with options.  May return an error if the
// name is invalid (e.g., empty) or improperly registered (e.g.,
// duplicate registration).
func (m Meter) NewInt64Counter(name string, options ...InstrumentOption) (Int64Counter, error) {
	return wrapInt64CounterInstrument(
		m.newSync(name, CounterInstrumentKind, number.Int64Kind, options))
}

// NewFloat64Counter creates a new floating point Counter with the
// given name, customized with options.  May return an error if the
// name is invalid (e.g., empty) or improperly registered (e.g.,
// duplicate registration).
func (m Meter) NewFloat64Counter(name string, options ...InstrumentOption) (Float64Counter, error) {
	return wrapFloat64CounterInstrument(
		m.newSync(name, CounterInstrumentKind, number.Float64Kind, options))
}

// NewInt64UpDownCounter creates a new integer UpDownCounter instrument with the
// given name, customized with options.  May return an error if the
// name is invalid (e.g., empty) or improperly registered (e.g.,
// duplicate registration).
func (m Meter) NewInt64UpDownCounter(name string, options ...InstrumentOption) (Int64UpDownCounter, error) {
	return wrapInt64UpDownCounterInstrument(
		m.newSync(name, UpDownCounterInstrumentKind, number.Int64Kind, options))
}

// NewFloat64UpDownCounter creates a new floating point UpDownCounter with the
// given name, customized with options.  May return an error if the
// name is invalid (e.g., empty) or improperly registered (e.g.,
// duplicate registration).
func (m Meter) NewFloat64UpDownCounter(name string, options ...InstrumentOption) (Float64UpDownCounter, error) {
	return wrapFloat64UpDownCounterInstrument(
		m.newSync(name, UpDownCounterInstrumentKind, number.Float64Kind, options))
}

// NewInt64ValueRecorder creates a new integer ValueRecorder instrument with the
// given name, customized with options.  May return an error if the
// name is invalid (e.g., empty) or improperly registered (e.g.,
// duplicate registration).
func (m Meter) NewInt64ValueRecorder(name string, opts ...InstrumentOption) (Int64ValueRecorder, error) {
	return wrapInt64ValueRecorderInstrument(
		m.newSync(name, ValueRecorderInstrumentKind, number.Int64Kind, opts))
}

// NewFloat64ValueRecorder creates a new floating point ValueRecorder with the
// given name, customized with options.  May return an error if the
// name is invalid (e.g., empty) or improperly registered (e.g.,
// duplicate registration).
func (m Meter) NewFloat64ValueRecorder(name string, opts ...InstrumentOption) (Float64ValueRecorder, error) {
	return wrapFloat64ValueRecorderInstrument(
		m.newSync(name, ValueRecorderInstrumentKind, number.Float64Kind, opts))
}

// NewInt64ValueObserver creates a new integer ValueObserver instrument
// with the given name, running a given callback, and customized with
// options.  May return an error if the name is invalid (e.g., empty)
// or improperly registered (e.g., duplicate registration).
func (m Meter) NewInt64ValueObserver(name string, callback Int64ObserverFunc, opts ...InstrumentOption) (Int64ValueObserver, error) {
	if callback == nil {
		return wrapInt64ValueObserverInstrument(NoopAsync{}, nil)
	}
	return wrapInt64ValueObserverInstrument(
		m.newAsync(name, ValueObserverInstrumentKind, number.Int64Kind, opts,
			newInt64AsyncRunner(callback)))
}

// NewFloat64ValueObserver creates a new floating point ValueObserver with
// the given name, running a given callback, and customized with
// options.  May return an error if the name is invalid (e.g., empty)
// or improperly registered (e.g., duplicate registration).
func (m Meter) NewFloat64ValueObserver(name string, callback Float64ObserverFunc, opts ...InstrumentOption) (Float64ValueObserver, error) {
	if callback == nil {
		return wrapFloat64ValueObserverInstrument(NoopAsync{}, nil)
	}
	return wrapFloat64ValueObserverInstrument(
		m.newAsync(name, ValueObserverInstrumentKind, number.Float64Kind, opts,
			newFloat64AsyncRunner(callback)))
}

// NewInt64SumObserver creates a new integer SumObserver instrument
// with the given name, running a given callback, and customized with
// options.  May return an error if the name is invalid (e.g., empty)
// or improperly registered (e.g., duplicate registration).
func (m Meter) NewInt64SumObserver(name string, callback Int64ObserverFunc, opts ...InstrumentOption) (Int64SumObserver, error) {
	if callback == nil {
		return wrapInt64SumObserverInstrument(NoopAsync{}, nil)
	}
	return wrapInt64SumObserverInstrument(
		m.newAsync(name, SumObserverInstrumentKind, number.Int64Kind, opts,
			newInt64AsyncRunner(callback)))
}

// NewFloat64SumObserver creates a new floating point SumObserver with
// the given name, running a given callback, and customized with
// options.  May return an error if the name is invalid (e.g., empty)
// or improperly registered (e.g., duplicate registration).
func (m Meter) NewFloat64SumObserver(name string, callback Float64ObserverFunc, opts ...InstrumentOption) (Float64SumObserver, error) {
	if callback == nil {
		return wrapFloat64SumObserverInstrument(NoopAsync{}, nil)
	}
	return wrapFloat64SumObserverInstrument(
		m.newAsync(name, SumObserverInstrumentKind, number.Float64Kind, opts,
			newFloat64AsyncRunner(callback)))
}

// NewInt64UpDownSumObserver creates a new integer UpDownSumObserver instrument
// with the given name, running a given callback, and customized with
// options.  May return an error if the name is invalid (e.g., empty)
// or improperly registered (e.g., duplicate registration).
func (m Meter) NewInt64UpDownSumObserver(name string, callback Int64ObserverFunc, opts ...InstrumentOption) (Int64UpDownSumObserver, error) {
	if callback == nil {
		return wrapInt64UpDownSumObserverInstrument(NoopAsync{}, nil)
	}
	return wrapInt64UpDownSumObserverInstrument(
		m.newAsync(name, UpDownSumObserverInstrumentKind, number.Int64Kind, opts,
			newInt64AsyncRunner(callback)))
}

// NewFloat64UpDownSumObserver creates a new floating point UpDownSumObserver with
// the given name, running a given callback, and customized with
// options.  May return an error if the name is invalid (e.g., empty)
// or improperly registered (e.g., duplicate registration).
func (m Meter) NewFloat64UpDownSumObserver(name string, callback Float64ObserverFunc, opts ...InstrumentOption) (Float64UpDownSumObserver, error) {
	if callback == nil {
		return wrapFloat64UpDownSumObserverInstrument(NoopAsync{}, nil)
	}
	return wrapFloat64UpDownSumObserverInstrument(
		m.newAsync(name, UpDownSumObserverInstrumentKind, number.Float64Kind, opts,
			newFloat64AsyncRunner(callback)))
}

// NewInt64ValueObserver creates a new integer ValueObserver instrument
// with the given name, running in a batch callback, and customized with
// options.  May return an error if the name is invalid (e.g., empty)
// or improperly registered (e.g., duplicate registration).
func (b BatchObserver) NewInt64ValueObserver(name string, opts ...InstrumentOption) (Int64ValueObserver, error) {
	if b.runner == nil {
		return wrapInt64ValueObserverInstrument(NoopAsync{}, nil)
	}
	return wrapInt64ValueObserverInstrument(
		b.meter.newAsync(name, ValueObserverInstrumentKind, number.Int64Kind, opts, b.runner))
}

// NewFloat64ValueObserver creates a new floating point ValueObserver with
// the given name, running in a batch callback, and customized with
// options.  May return an error if the name is invalid (e.g., empty)
// or improperly registered (e.g., duplicate registration).
func (b BatchObserver) NewFloat64ValueObserver(name string, opts ...InstrumentOption) (Float64ValueObserver, error) {
	if b.runner == nil {
		return wrapFloat64ValueObserverInstrument(NoopAsync{}, nil)
	}
	return wrapFloat64ValueObserverInstrument(
		b.meter.newAsync(name, ValueObserverInstrumentKind, number.Float64Kind, opts,
			b.runner))
}

// NewInt64SumObserver creates a new integer SumObserver instrument
// with the given name, running in a batch callback, and customized with
// options.  May return an error if the name is invalid (e.g., empty)
// or improperly registered (e.g., duplicate registration).
func (b BatchObserver) NewInt64SumObserver(name string, opts ...InstrumentOption) (Int64SumObserver, error) {
	if b.runner == nil {
		return wrapInt64SumObserverInstrument(NoopAsync{}, nil)
	}
	return wrapInt64SumObserverInstrument(
		b.meter.newAsync(name, SumObserverInstrumentKind, number.Int64Kind, opts, b.runner))
}

// NewFloat64SumObserver creates a new floating point SumObserver with
// the given name, running in a batch callback, and customized with
// options.  May return an error if the name is invalid (e.g., empty)
// or improperly registered (e.g., duplicate registration).
func (b BatchObserver) NewFloat64SumObserver(name string, opts ...InstrumentOption) (Float64SumObserver, error) {
	if b.runner == nil {
		return wrapFloat64SumObserverInstrument(NoopAsync{}, nil)
	}
	return wrapFloat64SumObserverInstrument(
		b.meter.newAsync(name, SumObserverInstrumentKind, number.Float64Kind, opts,
			b.runner))
}

// NewInt64UpDownSumObserver creates a new integer UpDownSumObserver instrument
// with the given name, running in a batch callback, and customized with
// options.  May return an error if the name is invalid (e.g., empty)
// or improperly registered (e.g., duplicate registration).
func (b BatchObserver) NewInt64UpDownSumObserver(name string, opts ...InstrumentOption) (Int64UpDownSumObserver, error) {
	if b.runner == nil {
		return wrapInt64UpDownSumObserverInstrument(NoopAsync{}, nil)
	}
	return wrapInt64UpDownSumObserverInstrument(
		b.meter.newAsync(name, UpDownSumObserverInstrumentKind, number.Int64Kind, opts, b.runner))
}

// NewFloat64UpDownSumObserver creates a new floating point UpDownSumObserver with
// the given name, running in a batch callback, and customized with
// options.  May return an error if the name is invalid (e.g., empty)
// or improperly registered (e.g., duplicate registration).
func (b BatchObserver) NewFloat64UpDownSumObserver(name string, opts ...InstrumentOption) (Float64UpDownSumObserver, error) {
	if b.runner == nil {
		return wrapFloat64UpDownSumObserverInstrument(NoopAsync{}, nil)
	}
	return wrapFloat64UpDownSumObserverInstrument(
		b.meter.newAsync(name, UpDownSumObserverInstrumentKind, number.Float64Kind, opts,
			b.runner))
}

// MeterImpl returns the underlying MeterImpl of this Meter.
func (m Meter) MeterImpl() MeterImpl {
	return m.impl
}

// newAsync constructs one new asynchronous instrument.
func (m Meter) newAsync(
	name string,
	mkind InstrumentKind,
	nkind number.Kind,
	opts []InstrumentOption,
	runner AsyncRunner,
) (
	AsyncImpl,
	error,
) {
	if m.impl == nil {
		return NoopAsync{}, nil
	}
	desc := NewDescriptor(name, mkind, nkind, opts...)
	desc.config.InstrumentationName = m.name
	desc.config.InstrumentationVersion = m.version
	return m.impl.NewAsyncInstrument(desc, runner)
}

// newSync constructs one new synchronous instrument.
func (m Meter) newSync(
	name string,
	metricKind InstrumentKind,
	numberKind number.Kind,
	opts []InstrumentOption,
) (
	SyncImpl,
	error,
) {
	if m.impl == nil {
		return NoopSync{}, nil
	}
	desc := NewDescriptor(name, metricKind, numberKind, opts...)
	desc.config.InstrumentationName = m.name
	desc.config.InstrumentationVersion = m.version
	return m.impl.NewSyncInstrument(desc)
}

// MeterMust is a wrapper for Meter interfaces that panics when any
// instrument constructor encounters an error.
type MeterMust struct {
	meter Meter
}

// BatchObserverMust is a wrapper for BatchObserver that panics when
// any instrument constructor encounters an error.
type BatchObserverMust struct {
	batch BatchObserver
}

// Must constructs a MeterMust implementation from a Meter, allowing
// the application to panic when any instrument constructor yields an
// error.
func Must(meter Meter) MeterMust {
	return MeterMust{meter: meter}
}

// NewInt64Counter calls `Meter.NewInt64Counter` and returns the
// instrument, panicking if it encounters an error.
func (mm MeterMust) NewInt64Counter(name string, cos ...InstrumentOption) Int64Counter {
	if inst, err := mm.meter.NewInt64Counter(name, cos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewFloat64Counter calls `Meter.NewFloat64Counter` and returns the
// instrument, panicking if it encounters an error.
func (mm MeterMust) NewFloat64Counter(name string, cos ...InstrumentOption) Float64Counter {
	if inst, err := mm.meter.NewFloat64Counter(name, cos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewInt64UpDownCounter calls `Meter.NewInt64UpDownCounter` and returns the
// instrument, panicking if it encounters an error.
func (mm MeterMust) NewInt64UpDownCounter(name string, cos ...InstrumentOption) Int64UpDownCounter {
	if inst, err := mm.meter.NewInt64UpDownCounter(name, cos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewFloat64UpDownCounter calls `Meter.NewFloat64UpDownCounter` and returns the
// instrument, panicking if it encounters an error.
func (mm MeterMust) NewFloat64UpDownCounter(name string, cos ...InstrumentOption) Float64UpDownCounter {
	if inst, err := mm.meter.NewFloat64UpDownCounter(name, cos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewInt64ValueRecorder calls `Meter.NewInt64ValueRecorder` and returns the
// instrument, panicking if it encounters an error.
func (mm MeterMust) NewInt64ValueRecorder(name string, mos ...InstrumentOption) Int64ValueRecorder {
	if inst, err := mm.meter.NewInt64ValueRecorder(name, mos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewFloat64ValueRecorder calls `Meter.NewFloat64ValueRecorder` and returns the
// instrument, panicking if it encounters an error.
func (mm MeterMust) NewFloat64ValueRecorder(name string, mos ...InstrumentOption) Float64ValueRecorder {
	if inst, err := mm.meter.NewFloat64ValueRecorder(name, mos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewInt64ValueObserver calls `Meter.NewInt64ValueObserver` and
// returns the instrument, panicking if it encounters an error.
func (mm MeterMust) NewInt64ValueObserver(name string, callback Int64ObserverFunc, oos ...InstrumentOption) Int64ValueObserver {
	if inst, err := mm.meter.NewInt64ValueObserver(name, callback, oos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewFloat64ValueObserver calls `Meter.NewFloat64ValueObserver` and
// returns the instrument, panicking if it encounters an error.
func (mm MeterMust) NewFloat64ValueObserver(name string, callback Float64ObserverFunc, oos ...InstrumentOption) Float64ValueObserver {
	if inst, err := mm.meter.NewFloat64ValueObserver(name, callback, oos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewInt64SumObserver calls `Meter.NewInt64SumObserver` and
// returns the instrument, panicking if it encounters an error.
func (mm MeterMust) NewInt64SumObserver(name string, callback Int64ObserverFunc, oos ...InstrumentOption) Int64SumObserver {
	if inst, err := mm.meter.NewInt64SumObserver(name, callback, oos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewFloat64SumObserver calls `Meter.NewFloat64SumObserver` and
// returns the instrument, panicking if it encounters an error.
func (mm MeterMust) NewFloat64SumObserver(name string, callback Float64ObserverFunc, oos ...InstrumentOption) Float64SumObserver {
	if inst, err := mm.meter.NewFloat64SumObserver(name, callback, oos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewInt64UpDownSumObserver calls `Meter.NewInt64UpDownSumObserver` and
// returns the instrument, panicking if it encounters an error.
func (mm MeterMust) NewInt64UpDownSumObserver(name string, callback Int64ObserverFunc, oos ...InstrumentOption) Int64UpDownSumObserver {
	if inst, err := mm.meter.NewInt64UpDownSumObserver(name, callback, oos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewFloat64UpDownSumObserver calls `Meter.NewFloat64UpDownSumObserver` and
// returns the instrument, panicking if it encounters an error.
func (mm MeterMust) NewFloat64UpDownSumObserver(name string, callback Float64ObserverFunc, oos ...InstrumentOption) Float64UpDownSumObserver {
	if inst, err := mm.meter.NewFloat64UpDownSumObserver(name, callback, oos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewBatchObserver returns a wrapper around BatchObserver that panics
// when any instrument constructor returns an error.
func (mm MeterMust) NewBatchObserver(callback BatchObserverFunc) BatchObserverMust {
	return BatchObserverMust{
		batch: mm.meter.NewBatchObserver(callback),
	}
}

// NewInt64ValueObserver calls `BatchObserver.NewInt64ValueObserver` and
// returns the instrument, panicking if it encounters an error.
func (bm BatchObserverMust) NewInt64ValueObserver(name string, oos ...InstrumentOption) Int64ValueObserver {
	if inst, err := bm.batch.NewInt64ValueObserver(name, oos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewFloat64ValueObserver calls `BatchObserver.NewFloat64ValueObserver` and
// returns the instrument, panicking if it encounters an error.
func (bm BatchObserverMust) NewFloat64ValueObserver(name string, oos ...InstrumentOption) Float64ValueObserver {
	if inst, err := bm.batch.NewFloat64ValueObserver(name, oos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewInt64SumObserver calls `BatchObserver.NewInt64SumObserver` and
// returns the instrument, panicking if it encounters an error.
func (bm BatchObserverMust) NewInt64SumObserver(name string, oos ...InstrumentOption) Int64SumObserver {
	if inst, err := bm.batch.NewInt64SumObserver(name, oos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewFloat64SumObserver calls `BatchObserver.NewFloat64SumObserver` and
// returns the instrument, panicking if it encounters an error.
func (bm BatchObserverMust) NewFloat64SumObserver(name string, oos ...InstrumentOption) Float64SumObserver {
	if inst, err := bm.batch.NewFloat64SumObserver(name, oos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewInt64UpDownSumObserver calls `BatchObserver.NewInt64UpDownSumObserver` and
// returns the instrument, panicking if it encounters an error.
func (bm BatchObserverMust) NewInt64UpDownSumObserver(name string, oos ...InstrumentOption) Int64UpDownSumObserver {
	if inst, err := bm.batch.NewInt64UpDownSumObserver(name, oos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// NewFloat64UpDownSumObserver calls `BatchObserver.NewFloat64UpDownSumObserver` and
// returns the instrument, panicking if it encounters an error.
func (bm BatchObserverMust) NewFloat64UpDownSumObserver(name string, oos ...InstrumentOption) Float64UpDownSumObserver {
	if inst, err := bm.batch.NewFloat64UpDownSumObserver(name, oos...); err != nil {
		panic(err)
	} else {
		return inst
	}
}

// Descriptor contains all the settings that describe an instrument,
// including its name, metric kind, number kind, and the configurable
// options.
type Descriptor struct {
	name           string
	instrumentKind InstrumentKind
	numberKind     number.Kind
	config         InstrumentConfig
}

// NewDescriptor returns a Descriptor with the given contents.
func NewDescriptor(name string, ikind InstrumentKind, nkind number.Kind, opts ...InstrumentOption) Descriptor {
	return Descriptor{
		name:           name,
		instrumentKind: ikind,
		numberKind:     nkind,
		config:         NewInstrumentConfig(opts...),
	}
}

// Name returns the metric instrument's name.
func (d Descriptor) Name() string {
	return d.name
}

// InstrumentKind returns the specific kind of instrument.
func (d Descriptor) InstrumentKind() InstrumentKind {
	return d.instrumentKind
}

// Description provides a human-readable description of the metric
// instrument.
func (d Descriptor) Description() string {
	return d.config.Description
}

// Unit describes the units of the metric instrument.  Unitless
// metrics return the empty string.
func (d Descriptor) Unit() unit.Unit {
	return d.config.Unit
}

// NumberKind returns whether this instrument is declared over int64,
// float64, or uint64 values.
func (d Descriptor) NumberKind() number.Kind {
	return d.numberKind
}

// InstrumentationName returns the name of the library that provided
// instrumentation for this instrument.
func (d Descriptor) InstrumentationName() string {
	return d.config.InstrumentationName
}

// InstrumentationVersion returns the version of the library that provided
// instrumentation for this instrument.
func (d Descriptor) InstrumentationVersion() string {
	return d.config.InstrumentationVersion
}
