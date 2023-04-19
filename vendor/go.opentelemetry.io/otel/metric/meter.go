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
	"go.opentelemetry.io/otel/metric/instrument"
)

// MeterProvider provides access to named Meter instances, for instrumenting
// an application or library.
//
// Warning: methods may be added to this interface in minor releases.
type MeterProvider interface {
	// Meter creates an instance of a `Meter` interface. The name must be the
	// name of the library providing instrumentation. This name may be the same
	// as the instrumented code only if that code provides built-in
	// instrumentation. If the name is empty, then a implementation defined
	// default name will be used instead.
	Meter(name string, opts ...MeterOption) Meter
}

// Meter provides access to instrument instances for recording metrics.
//
// Warning: methods may be added to this interface in minor releases.
type Meter interface {
	// Int64Counter returns a new instrument identified by name and configured
	// with options. The instrument is used to synchronously record increasing
	// int64 measurements during a computational operation.
	Int64Counter(name string, options ...instrument.Int64Option) (instrument.Int64Counter, error)
	// Int64UpDownCounter returns a new instrument identified by name and
	// configured with options. The instrument is used to synchronously record
	// int64 measurements during a computational operation.
	Int64UpDownCounter(name string, options ...instrument.Int64Option) (instrument.Int64UpDownCounter, error)
	// Int64Histogram returns a new instrument identified by name and
	// configured with options. The instrument is used to synchronously record
	// the distribution of int64 measurements during a computational operation.
	Int64Histogram(name string, options ...instrument.Int64Option) (instrument.Int64Histogram, error)
	// Int64ObservableCounter returns a new instrument identified by name and
	// configured with options. The instrument is used to asynchronously record
	// increasing int64 measurements once per a measurement collection cycle.
	Int64ObservableCounter(name string, options ...instrument.Int64ObserverOption) (instrument.Int64ObservableCounter, error)
	// Int64ObservableUpDownCounter returns a new instrument identified by name
	// and configured with options. The instrument is used to asynchronously
	// record int64 measurements once per a measurement collection cycle.
	Int64ObservableUpDownCounter(name string, options ...instrument.Int64ObserverOption) (instrument.Int64ObservableUpDownCounter, error)
	// Int64ObservableGauge returns a new instrument identified by name and
	// configured with options. The instrument is used to asynchronously record
	// instantaneous int64 measurements once per a measurement collection
	// cycle.
	Int64ObservableGauge(name string, options ...instrument.Int64ObserverOption) (instrument.Int64ObservableGauge, error)

	// Float64Counter returns a new instrument identified by name and
	// configured with options. The instrument is used to synchronously record
	// increasing float64 measurements during a computational operation.
	Float64Counter(name string, options ...instrument.Float64Option) (instrument.Float64Counter, error)
	// Float64UpDownCounter returns a new instrument identified by name and
	// configured with options. The instrument is used to synchronously record
	// float64 measurements during a computational operation.
	Float64UpDownCounter(name string, options ...instrument.Float64Option) (instrument.Float64UpDownCounter, error)
	// Float64Histogram returns a new instrument identified by name and
	// configured with options. The instrument is used to synchronously record
	// the distribution of float64 measurements during a computational
	// operation.
	Float64Histogram(name string, options ...instrument.Float64Option) (instrument.Float64Histogram, error)
	// Float64ObservableCounter returns a new instrument identified by name and
	// configured with options. The instrument is used to asynchronously record
	// increasing float64 measurements once per a measurement collection cycle.
	Float64ObservableCounter(name string, options ...instrument.Float64ObserverOption) (instrument.Float64ObservableCounter, error)
	// Float64ObservableUpDownCounter returns a new instrument identified by
	// name and configured with options. The instrument is used to
	// asynchronously record float64 measurements once per a measurement
	// collection cycle.
	Float64ObservableUpDownCounter(name string, options ...instrument.Float64ObserverOption) (instrument.Float64ObservableUpDownCounter, error)
	// Float64ObservableGauge returns a new instrument identified by name and
	// configured with options. The instrument is used to asynchronously record
	// instantaneous float64 measurements once per a measurement collection
	// cycle.
	Float64ObservableGauge(name string, options ...instrument.Float64ObserverOption) (instrument.Float64ObservableGauge, error)

	// RegisterCallback registers f to be called during the collection of a
	// measurement cycle.
	//
	// If Unregister of the returned Registration is called, f needs to be
	// unregistered and not called during collection.
	//
	// The instruments f is registered with are the only instruments that f may
	// observe values for.
	//
	// If no instruments are passed, f should not be registered nor called
	// during collection.
	RegisterCallback(f Callback, instruments ...instrument.Asynchronous) (Registration, error)
}

// Callback is a function registered with a Meter that makes observations for
// the set of instruments it is registered with. The Observer parameter is used
// to record measurment observations for these instruments.
//
// The function needs to complete in a finite amount of time and the deadline
// of the passed context is expected to be honored.
//
// The function needs to make unique observations across all registered
// Callbacks. Meaning, it should not report measurements for an instrument with
// the same attributes as another Callback will report.
//
// The function needs to be concurrent safe.
type Callback func(context.Context, Observer) error

// Observer records measurements for multiple instruments in a Callback.
type Observer interface {
	// ObserveFloat64 records the float64 value with attributes for obsrv.
	ObserveFloat64(obsrv instrument.Float64Observable, value float64, attributes ...attribute.KeyValue)
	// ObserveInt64 records the int64 value with attributes for obsrv.
	ObserveInt64(obsrv instrument.Int64Observable, value int64, attributes ...attribute.KeyValue)
}

// Registration is an token representing the unique registration of a callback
// for a set of instruments with a Meter.
type Registration interface {
	// Unregister removes the callback registration from a Meter.
	//
	// This method needs to be idempotent and concurrent safe.
	Unregister() error
}
