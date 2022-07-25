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

//go:generate stringer -type=InstrumentKind

package metric // import "go.opentelemetry.io/otel/metric"

import (
	"context"
	"errors"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric/number"
)

// ErrSDKReturnedNilImpl is returned when a new `MeterImpl` returns nil.
var ErrSDKReturnedNilImpl = errors.New("SDK returned a nil implementation")

// InstrumentKind describes the kind of instrument.
type InstrumentKind int8

const (
	// ValueRecorderInstrumentKind indicates a ValueRecorder instrument.
	ValueRecorderInstrumentKind InstrumentKind = iota
	// ValueObserverInstrumentKind indicates an ValueObserver instrument.
	ValueObserverInstrumentKind

	// CounterInstrumentKind indicates a Counter instrument.
	CounterInstrumentKind
	// UpDownCounterInstrumentKind indicates a UpDownCounter instrument.
	UpDownCounterInstrumentKind

	// SumObserverInstrumentKind indicates a SumObserver instrument.
	SumObserverInstrumentKind
	// UpDownSumObserverInstrumentKind indicates a UpDownSumObserver
	// instrument.
	UpDownSumObserverInstrumentKind
)

// Synchronous returns whether this is a synchronous kind of instrument.
func (k InstrumentKind) Synchronous() bool {
	switch k {
	case CounterInstrumentKind, UpDownCounterInstrumentKind, ValueRecorderInstrumentKind:
		return true
	}
	return false
}

// Asynchronous returns whether this is an asynchronous kind of instrument.
func (k InstrumentKind) Asynchronous() bool {
	return !k.Synchronous()
}

// Adding returns whether this kind of instrument adds its inputs (as opposed to Grouping).
func (k InstrumentKind) Adding() bool {
	switch k {
	case CounterInstrumentKind, UpDownCounterInstrumentKind, SumObserverInstrumentKind, UpDownSumObserverInstrumentKind:
		return true
	}
	return false
}

// Grouping returns whether this kind of instrument groups its inputs (as opposed to Adding).
func (k InstrumentKind) Grouping() bool {
	return !k.Adding()
}

// Monotonic returns whether this kind of instrument exposes a non-decreasing sum.
func (k InstrumentKind) Monotonic() bool {
	switch k {
	case CounterInstrumentKind, SumObserverInstrumentKind:
		return true
	}
	return false
}

// PrecomputedSum returns whether this kind of instrument receives precomputed sums.
func (k InstrumentKind) PrecomputedSum() bool {
	return k.Adding() && k.Asynchronous()
}

// Observation is used for reporting an asynchronous  batch of metric
// values. Instances of this type should be created by asynchronous
// instruments (e.g., Int64ValueObserver.Observation()).
type Observation struct {
	// number needs to be aligned for 64-bit atomic operations.
	number     number.Number
	instrument AsyncImpl
}

// Int64ObserverFunc is a type of callback that integral
// observers run.
type Int64ObserverFunc func(context.Context, Int64ObserverResult)

// Float64ObserverFunc is a type of callback that floating point
// observers run.
type Float64ObserverFunc func(context.Context, Float64ObserverResult)

// BatchObserverFunc is a callback argument for use with any
// Observer instrument that will be reported as a batch of
// observations.
type BatchObserverFunc func(context.Context, BatchObserverResult)

// Int64ObserverResult is passed to an observer callback to capture
// observations for one asynchronous integer metric instrument.
type Int64ObserverResult struct {
	instrument AsyncImpl
	function   func([]attribute.KeyValue, ...Observation)
}

// Float64ObserverResult is passed to an observer callback to capture
// observations for one asynchronous floating point metric instrument.
type Float64ObserverResult struct {
	instrument AsyncImpl
	function   func([]attribute.KeyValue, ...Observation)
}

// BatchObserverResult is passed to a batch observer callback to
// capture observations for multiple asynchronous instruments.
type BatchObserverResult struct {
	function func([]attribute.KeyValue, ...Observation)
}

// Observe captures a single integer value from the associated
// instrument callback, with the given labels.
func (ir Int64ObserverResult) Observe(value int64, labels ...attribute.KeyValue) {
	ir.function(labels, Observation{
		instrument: ir.instrument,
		number:     number.NewInt64Number(value),
	})
}

// Observe captures a single floating point value from the associated
// instrument callback, with the given labels.
func (fr Float64ObserverResult) Observe(value float64, labels ...attribute.KeyValue) {
	fr.function(labels, Observation{
		instrument: fr.instrument,
		number:     number.NewFloat64Number(value),
	})
}

// Observe captures a multiple observations from the associated batch
// instrument callback, with the given labels.
func (br BatchObserverResult) Observe(labels []attribute.KeyValue, obs ...Observation) {
	br.function(labels, obs...)
}

// AsyncRunner is expected to convert into an AsyncSingleRunner or an
// AsyncBatchRunner.  SDKs will encounter an error if the AsyncRunner
// does not satisfy one of these interfaces.
type AsyncRunner interface {
	// AnyRunner() is a non-exported method with no functional use
	// other than to make this a non-empty interface.
	AnyRunner()
}

// AsyncSingleRunner is an interface implemented by single-observer
// callbacks.
type AsyncSingleRunner interface {
	// Run accepts a single instrument and function for capturing
	// observations of that instrument.  Each call to the function
	// receives one captured observation.  (The function accepts
	// multiple observations so the same implementation can be
	// used for batch runners.)
	Run(ctx context.Context, single AsyncImpl, capture func([]attribute.KeyValue, ...Observation))

	AsyncRunner
}

// AsyncBatchRunner is an interface implemented by batch-observer
// callbacks.
type AsyncBatchRunner interface {
	// Run accepts a function for capturing observations of
	// multiple instruments.
	Run(ctx context.Context, capture func([]attribute.KeyValue, ...Observation))

	AsyncRunner
}

var _ AsyncSingleRunner = (*Int64ObserverFunc)(nil)
var _ AsyncSingleRunner = (*Float64ObserverFunc)(nil)
var _ AsyncBatchRunner = (*BatchObserverFunc)(nil)

// newInt64AsyncRunner returns a single-observer callback for integer Observer instruments.
func newInt64AsyncRunner(c Int64ObserverFunc) AsyncSingleRunner {
	return &c
}

// newFloat64AsyncRunner returns a single-observer callback for floating point Observer instruments.
func newFloat64AsyncRunner(c Float64ObserverFunc) AsyncSingleRunner {
	return &c
}

// newBatchAsyncRunner returns a batch-observer callback use with multiple Observer instruments.
func newBatchAsyncRunner(c BatchObserverFunc) AsyncBatchRunner {
	return &c
}

// AnyRunner implements AsyncRunner.
func (*Int64ObserverFunc) AnyRunner() {}

// AnyRunner implements AsyncRunner.
func (*Float64ObserverFunc) AnyRunner() {}

// AnyRunner implements AsyncRunner.
func (*BatchObserverFunc) AnyRunner() {}

// Run implements AsyncSingleRunner.
func (i *Int64ObserverFunc) Run(ctx context.Context, impl AsyncImpl, function func([]attribute.KeyValue, ...Observation)) {
	(*i)(ctx, Int64ObserverResult{
		instrument: impl,
		function:   function,
	})
}

// Run implements AsyncSingleRunner.
func (f *Float64ObserverFunc) Run(ctx context.Context, impl AsyncImpl, function func([]attribute.KeyValue, ...Observation)) {
	(*f)(ctx, Float64ObserverResult{
		instrument: impl,
		function:   function,
	})
}

// Run implements AsyncBatchRunner.
func (b *BatchObserverFunc) Run(ctx context.Context, function func([]attribute.KeyValue, ...Observation)) {
	(*b)(ctx, BatchObserverResult{
		function: function,
	})
}

// wrapInt64ValueObserverInstrument converts an AsyncImpl into Int64ValueObserver.
func wrapInt64ValueObserverInstrument(asyncInst AsyncImpl, err error) (Int64ValueObserver, error) {
	common, err := checkNewAsync(asyncInst, err)
	return Int64ValueObserver{asyncInstrument: common}, err
}

// wrapFloat64ValueObserverInstrument converts an AsyncImpl into Float64ValueObserver.
func wrapFloat64ValueObserverInstrument(asyncInst AsyncImpl, err error) (Float64ValueObserver, error) {
	common, err := checkNewAsync(asyncInst, err)
	return Float64ValueObserver{asyncInstrument: common}, err
}

// wrapInt64SumObserverInstrument converts an AsyncImpl into Int64SumObserver.
func wrapInt64SumObserverInstrument(asyncInst AsyncImpl, err error) (Int64SumObserver, error) {
	common, err := checkNewAsync(asyncInst, err)
	return Int64SumObserver{asyncInstrument: common}, err
}

// wrapFloat64SumObserverInstrument converts an AsyncImpl into Float64SumObserver.
func wrapFloat64SumObserverInstrument(asyncInst AsyncImpl, err error) (Float64SumObserver, error) {
	common, err := checkNewAsync(asyncInst, err)
	return Float64SumObserver{asyncInstrument: common}, err
}

// wrapInt64UpDownSumObserverInstrument converts an AsyncImpl into Int64UpDownSumObserver.
func wrapInt64UpDownSumObserverInstrument(asyncInst AsyncImpl, err error) (Int64UpDownSumObserver, error) {
	common, err := checkNewAsync(asyncInst, err)
	return Int64UpDownSumObserver{asyncInstrument: common}, err
}

// wrapFloat64UpDownSumObserverInstrument converts an AsyncImpl into Float64UpDownSumObserver.
func wrapFloat64UpDownSumObserverInstrument(asyncInst AsyncImpl, err error) (Float64UpDownSumObserver, error) {
	common, err := checkNewAsync(asyncInst, err)
	return Float64UpDownSumObserver{asyncInstrument: common}, err
}

// BatchObserver represents an Observer callback that can report
// observations for multiple instruments.
type BatchObserver struct {
	meter  Meter
	runner AsyncBatchRunner
}

// Int64ValueObserver is a metric that captures a set of int64 values at a
// point in time.
type Int64ValueObserver struct {
	asyncInstrument
}

// Float64ValueObserver is a metric that captures a set of float64 values
// at a point in time.
type Float64ValueObserver struct {
	asyncInstrument
}

// Int64SumObserver is a metric that captures a precomputed sum of
// int64 values at a point in time.
type Int64SumObserver struct {
	asyncInstrument
}

// Float64SumObserver is a metric that captures a precomputed sum of
// float64 values at a point in time.
type Float64SumObserver struct {
	asyncInstrument
}

// Int64UpDownSumObserver is a metric that captures a precomputed sum of
// int64 values at a point in time.
type Int64UpDownSumObserver struct {
	asyncInstrument
}

// Float64UpDownSumObserver is a metric that captures a precomputed sum of
// float64 values at a point in time.
type Float64UpDownSumObserver struct {
	asyncInstrument
}

// Observation returns an Observation, a BatchObserverFunc
// argument, for an asynchronous integer instrument.
// This returns an implementation-level object for use by the SDK,
// users should not refer to this.
func (i Int64ValueObserver) Observation(v int64) Observation {
	return Observation{
		number:     number.NewInt64Number(v),
		instrument: i.instrument,
	}
}

// Observation returns an Observation, a BatchObserverFunc
// argument, for an asynchronous integer instrument.
// This returns an implementation-level object for use by the SDK,
// users should not refer to this.
func (f Float64ValueObserver) Observation(v float64) Observation {
	return Observation{
		number:     number.NewFloat64Number(v),
		instrument: f.instrument,
	}
}

// Observation returns an Observation, a BatchObserverFunc
// argument, for an asynchronous integer instrument.
// This returns an implementation-level object for use by the SDK,
// users should not refer to this.
func (i Int64SumObserver) Observation(v int64) Observation {
	return Observation{
		number:     number.NewInt64Number(v),
		instrument: i.instrument,
	}
}

// Observation returns an Observation, a BatchObserverFunc
// argument, for an asynchronous integer instrument.
// This returns an implementation-level object for use by the SDK,
// users should not refer to this.
func (f Float64SumObserver) Observation(v float64) Observation {
	return Observation{
		number:     number.NewFloat64Number(v),
		instrument: f.instrument,
	}
}

// Observation returns an Observation, a BatchObserverFunc
// argument, for an asynchronous integer instrument.
// This returns an implementation-level object for use by the SDK,
// users should not refer to this.
func (i Int64UpDownSumObserver) Observation(v int64) Observation {
	return Observation{
		number:     number.NewInt64Number(v),
		instrument: i.instrument,
	}
}

// Observation returns an Observation, a BatchObserverFunc
// argument, for an asynchronous integer instrument.
// This returns an implementation-level object for use by the SDK,
// users should not refer to this.
func (f Float64UpDownSumObserver) Observation(v float64) Observation {
	return Observation{
		number:     number.NewFloat64Number(v),
		instrument: f.instrument,
	}
}

// Measurement is used for reporting a synchronous batch of metric
// values. Instances of this type should be created by synchronous
// instruments (e.g., Int64Counter.Measurement()).
type Measurement struct {
	// number needs to be aligned for 64-bit atomic operations.
	number     number.Number
	instrument SyncImpl
}

// syncInstrument contains a SyncImpl.
type syncInstrument struct {
	instrument SyncImpl
}

// syncBoundInstrument contains a BoundSyncImpl.
type syncBoundInstrument struct {
	boundInstrument BoundSyncImpl
}

// asyncInstrument contains a AsyncImpl.
type asyncInstrument struct {
	instrument AsyncImpl
}

// SyncImpl returns the instrument that created this measurement.
// This returns an implementation-level object for use by the SDK,
// users should not refer to this.
func (m Measurement) SyncImpl() SyncImpl {
	return m.instrument
}

// Number returns a number recorded in this measurement.
func (m Measurement) Number() number.Number {
	return m.number
}

// AsyncImpl returns the instrument that created this observation.
// This returns an implementation-level object for use by the SDK,
// users should not refer to this.
func (m Observation) AsyncImpl() AsyncImpl {
	return m.instrument
}

// Number returns a number recorded in this observation.
func (m Observation) Number() number.Number {
	return m.number
}

// AsyncImpl implements AsyncImpl.
func (a asyncInstrument) AsyncImpl() AsyncImpl {
	return a.instrument
}

// SyncImpl returns the implementation object for synchronous instruments.
func (s syncInstrument) SyncImpl() SyncImpl {
	return s.instrument
}

func (s syncInstrument) bind(labels []attribute.KeyValue) syncBoundInstrument {
	return newSyncBoundInstrument(s.instrument.Bind(labels))
}

func (s syncInstrument) float64Measurement(value float64) Measurement {
	return newMeasurement(s.instrument, number.NewFloat64Number(value))
}

func (s syncInstrument) int64Measurement(value int64) Measurement {
	return newMeasurement(s.instrument, number.NewInt64Number(value))
}

func (s syncInstrument) directRecord(ctx context.Context, number number.Number, labels []attribute.KeyValue) {
	s.instrument.RecordOne(ctx, number, labels)
}

func (h syncBoundInstrument) directRecord(ctx context.Context, number number.Number) {
	h.boundInstrument.RecordOne(ctx, number)
}

// Unbind calls SyncImpl.Unbind.
func (h syncBoundInstrument) Unbind() {
	h.boundInstrument.Unbind()
}

// checkNewAsync receives an AsyncImpl and potential
// error, and returns the same types, checking for and ensuring that
// the returned interface is not nil.
func checkNewAsync(instrument AsyncImpl, err error) (asyncInstrument, error) {
	if instrument == nil {
		if err == nil {
			err = ErrSDKReturnedNilImpl
		}
		instrument = NoopAsync{}
	}
	return asyncInstrument{
		instrument: instrument,
	}, err
}

// checkNewSync receives an SyncImpl and potential
// error, and returns the same types, checking for and ensuring that
// the returned interface is not nil.
func checkNewSync(instrument SyncImpl, err error) (syncInstrument, error) {
	if instrument == nil {
		if err == nil {
			err = ErrSDKReturnedNilImpl
		}
		// Note: an alternate behavior would be to synthesize a new name
		// or group all duplicately-named instruments of a certain type
		// together and use a tag for the original name, e.g.,
		//   name = 'invalid.counter.int64'
		//   label = 'original-name=duplicate-counter-name'
		instrument = NoopSync{}
	}
	return syncInstrument{
		instrument: instrument,
	}, err
}

func newSyncBoundInstrument(boundInstrument BoundSyncImpl) syncBoundInstrument {
	return syncBoundInstrument{
		boundInstrument: boundInstrument,
	}
}

func newMeasurement(instrument SyncImpl, number number.Number) Measurement {
	return Measurement{
		instrument: instrument,
		number:     number,
	}
}

// wrapInt64CounterInstrument converts a SyncImpl into Int64Counter.
func wrapInt64CounterInstrument(syncInst SyncImpl, err error) (Int64Counter, error) {
	common, err := checkNewSync(syncInst, err)
	return Int64Counter{syncInstrument: common}, err
}

// wrapFloat64CounterInstrument converts a SyncImpl into Float64Counter.
func wrapFloat64CounterInstrument(syncInst SyncImpl, err error) (Float64Counter, error) {
	common, err := checkNewSync(syncInst, err)
	return Float64Counter{syncInstrument: common}, err
}

// wrapInt64UpDownCounterInstrument converts a SyncImpl into Int64UpDownCounter.
func wrapInt64UpDownCounterInstrument(syncInst SyncImpl, err error) (Int64UpDownCounter, error) {
	common, err := checkNewSync(syncInst, err)
	return Int64UpDownCounter{syncInstrument: common}, err
}

// wrapFloat64UpDownCounterInstrument converts a SyncImpl into Float64UpDownCounter.
func wrapFloat64UpDownCounterInstrument(syncInst SyncImpl, err error) (Float64UpDownCounter, error) {
	common, err := checkNewSync(syncInst, err)
	return Float64UpDownCounter{syncInstrument: common}, err
}

// wrapInt64ValueRecorderInstrument converts a SyncImpl into Int64ValueRecorder.
func wrapInt64ValueRecorderInstrument(syncInst SyncImpl, err error) (Int64ValueRecorder, error) {
	common, err := checkNewSync(syncInst, err)
	return Int64ValueRecorder{syncInstrument: common}, err
}

// wrapFloat64ValueRecorderInstrument converts a SyncImpl into Float64ValueRecorder.
func wrapFloat64ValueRecorderInstrument(syncInst SyncImpl, err error) (Float64ValueRecorder, error) {
	common, err := checkNewSync(syncInst, err)
	return Float64ValueRecorder{syncInstrument: common}, err
}

// Float64Counter is a metric that accumulates float64 values.
type Float64Counter struct {
	syncInstrument
}

// Int64Counter is a metric that accumulates int64 values.
type Int64Counter struct {
	syncInstrument
}

// BoundFloat64Counter is a bound instrument for Float64Counter.
//
// It inherits the Unbind function from syncBoundInstrument.
type BoundFloat64Counter struct {
	syncBoundInstrument
}

// BoundInt64Counter is a boundInstrument for Int64Counter.
//
// It inherits the Unbind function from syncBoundInstrument.
type BoundInt64Counter struct {
	syncBoundInstrument
}

// Bind creates a bound instrument for this counter. The labels are
// associated with values recorded via subsequent calls to Record.
func (c Float64Counter) Bind(labels ...attribute.KeyValue) (h BoundFloat64Counter) {
	h.syncBoundInstrument = c.bind(labels)
	return
}

// Bind creates a bound instrument for this counter. The labels are
// associated with values recorded via subsequent calls to Record.
func (c Int64Counter) Bind(labels ...attribute.KeyValue) (h BoundInt64Counter) {
	h.syncBoundInstrument = c.bind(labels)
	return
}

// Measurement creates a Measurement object to use with batch
// recording.
func (c Float64Counter) Measurement(value float64) Measurement {
	return c.float64Measurement(value)
}

// Measurement creates a Measurement object to use with batch
// recording.
func (c Int64Counter) Measurement(value int64) Measurement {
	return c.int64Measurement(value)
}

// Add adds the value to the counter's sum. The labels should contain
// the keys and values to be associated with this value.
func (c Float64Counter) Add(ctx context.Context, value float64, labels ...attribute.KeyValue) {
	c.directRecord(ctx, number.NewFloat64Number(value), labels)
}

// Add adds the value to the counter's sum. The labels should contain
// the keys and values to be associated with this value.
func (c Int64Counter) Add(ctx context.Context, value int64, labels ...attribute.KeyValue) {
	c.directRecord(ctx, number.NewInt64Number(value), labels)
}

// Add adds the value to the counter's sum using the labels
// previously bound to this counter via Bind()
func (b BoundFloat64Counter) Add(ctx context.Context, value float64) {
	b.directRecord(ctx, number.NewFloat64Number(value))
}

// Add adds the value to the counter's sum using the labels
// previously bound to this counter via Bind()
func (b BoundInt64Counter) Add(ctx context.Context, value int64) {
	b.directRecord(ctx, number.NewInt64Number(value))
}

// Float64UpDownCounter is a metric instrument that sums floating
// point values.
type Float64UpDownCounter struct {
	syncInstrument
}

// Int64UpDownCounter is a metric instrument that sums integer values.
type Int64UpDownCounter struct {
	syncInstrument
}

// BoundFloat64UpDownCounter is a bound instrument for Float64UpDownCounter.
//
// It inherits the Unbind function from syncBoundInstrument.
type BoundFloat64UpDownCounter struct {
	syncBoundInstrument
}

// BoundInt64UpDownCounter is a boundInstrument for Int64UpDownCounter.
//
// It inherits the Unbind function from syncBoundInstrument.
type BoundInt64UpDownCounter struct {
	syncBoundInstrument
}

// Bind creates a bound instrument for this counter. The labels are
// associated with values recorded via subsequent calls to Record.
func (c Float64UpDownCounter) Bind(labels ...attribute.KeyValue) (h BoundFloat64UpDownCounter) {
	h.syncBoundInstrument = c.bind(labels)
	return
}

// Bind creates a bound instrument for this counter. The labels are
// associated with values recorded via subsequent calls to Record.
func (c Int64UpDownCounter) Bind(labels ...attribute.KeyValue) (h BoundInt64UpDownCounter) {
	h.syncBoundInstrument = c.bind(labels)
	return
}

// Measurement creates a Measurement object to use with batch
// recording.
func (c Float64UpDownCounter) Measurement(value float64) Measurement {
	return c.float64Measurement(value)
}

// Measurement creates a Measurement object to use with batch
// recording.
func (c Int64UpDownCounter) Measurement(value int64) Measurement {
	return c.int64Measurement(value)
}

// Add adds the value to the counter's sum. The labels should contain
// the keys and values to be associated with this value.
func (c Float64UpDownCounter) Add(ctx context.Context, value float64, labels ...attribute.KeyValue) {
	c.directRecord(ctx, number.NewFloat64Number(value), labels)
}

// Add adds the value to the counter's sum. The labels should contain
// the keys and values to be associated with this value.
func (c Int64UpDownCounter) Add(ctx context.Context, value int64, labels ...attribute.KeyValue) {
	c.directRecord(ctx, number.NewInt64Number(value), labels)
}

// Add adds the value to the counter's sum using the labels
// previously bound to this counter via Bind()
func (b BoundFloat64UpDownCounter) Add(ctx context.Context, value float64) {
	b.directRecord(ctx, number.NewFloat64Number(value))
}

// Add adds the value to the counter's sum using the labels
// previously bound to this counter via Bind()
func (b BoundInt64UpDownCounter) Add(ctx context.Context, value int64) {
	b.directRecord(ctx, number.NewInt64Number(value))
}

// Float64ValueRecorder is a metric that records float64 values.
type Float64ValueRecorder struct {
	syncInstrument
}

// Int64ValueRecorder is a metric that records int64 values.
type Int64ValueRecorder struct {
	syncInstrument
}

// BoundFloat64ValueRecorder is a bound instrument for Float64ValueRecorder.
//
// It inherits the Unbind function from syncBoundInstrument.
type BoundFloat64ValueRecorder struct {
	syncBoundInstrument
}

// BoundInt64ValueRecorder is a bound instrument for Int64ValueRecorder.
//
// It inherits the Unbind function from syncBoundInstrument.
type BoundInt64ValueRecorder struct {
	syncBoundInstrument
}

// Bind creates a bound instrument for this ValueRecorder. The labels are
// associated with values recorded via subsequent calls to Record.
func (c Float64ValueRecorder) Bind(labels ...attribute.KeyValue) (h BoundFloat64ValueRecorder) {
	h.syncBoundInstrument = c.bind(labels)
	return
}

// Bind creates a bound instrument for this ValueRecorder. The labels are
// associated with values recorded via subsequent calls to Record.
func (c Int64ValueRecorder) Bind(labels ...attribute.KeyValue) (h BoundInt64ValueRecorder) {
	h.syncBoundInstrument = c.bind(labels)
	return
}

// Measurement creates a Measurement object to use with batch
// recording.
func (c Float64ValueRecorder) Measurement(value float64) Measurement {
	return c.float64Measurement(value)
}

// Measurement creates a Measurement object to use with batch
// recording.
func (c Int64ValueRecorder) Measurement(value int64) Measurement {
	return c.int64Measurement(value)
}

// Record adds a new value to the list of ValueRecorder's records. The
// labels should contain the keys and values to be associated with
// this value.
func (c Float64ValueRecorder) Record(ctx context.Context, value float64, labels ...attribute.KeyValue) {
	c.directRecord(ctx, number.NewFloat64Number(value), labels)
}

// Record adds a new value to the ValueRecorder's distribution. The
// labels should contain the keys and values to be associated with
// this value.
func (c Int64ValueRecorder) Record(ctx context.Context, value int64, labels ...attribute.KeyValue) {
	c.directRecord(ctx, number.NewInt64Number(value), labels)
}

// Record adds a new value to the ValueRecorder's distribution using the labels
// previously bound to the ValueRecorder via Bind().
func (b BoundFloat64ValueRecorder) Record(ctx context.Context, value float64) {
	b.directRecord(ctx, number.NewFloat64Number(value))
}

// Record adds a new value to the ValueRecorder's distribution using the labels
// previously bound to the ValueRecorder via Bind().
func (b BoundInt64ValueRecorder) Record(ctx context.Context, value int64) {
	b.directRecord(ctx, number.NewInt64Number(value))
}
