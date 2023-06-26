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

	"go.opentelemetry.io/otel/metric/embedded"
)

// MeterProvider provides access to named Meter instances, for instrumenting
// an application or package.
//
// Warning: Methods may be added to this interface in minor releases. See
// package documentation on API implementation for information on how to set
// default behavior for unimplemented methods.
type MeterProvider interface {
	// Users of the interface can ignore this. This embedded type is only used
	// by implementations of this interface. See the "API Implementations"
	// section of the package documentation for more information.
	embedded.MeterProvider

	// Meter returns a new Meter with the provided name and configuration.
	//
	// A Meter should be scoped at most to a single package. The name needs to
	// be unique so it does not collide with other names used by
	// an application, nor other applications. To achieve this, the import path
	// of the instrumentation package is recommended to be used as name.
	//
	// If the name is empty, then an implementation defined default name will
	// be used instead.
	Meter(name string, opts ...MeterOption) Meter
}

// Meter provides access to instrument instances for recording metrics.
//
// Warning: Methods may be added to this interface in minor releases. See
// package documentation on API implementation for information on how to set
// default behavior for unimplemented methods.
type Meter interface {
	// Users of the interface can ignore this. This embedded type is only used
	// by implementations of this interface. See the "API Implementations"
	// section of the package documentation for more information.
	embedded.Meter

	// Int64Counter returns a new Int64Counter instrument identified by name
	// and configured with options. The instrument is used to synchronously
	// record increasing int64 measurements during a computational operation.
	Int64Counter(name string, options ...Int64CounterOption) (Int64Counter, error)
	// Int64UpDownCounter returns a new Int64UpDownCounter instrument
	// identified by name and configured with options. The instrument is used
	// to synchronously record int64 measurements during a computational
	// operation.
	Int64UpDownCounter(name string, options ...Int64UpDownCounterOption) (Int64UpDownCounter, error)
	// Int64Histogram returns a new Int64Histogram instrument identified by
	// name and configured with options. The instrument is used to
	// synchronously record the distribution of int64 measurements during a
	// computational operation.
	Int64Histogram(name string, options ...Int64HistogramOption) (Int64Histogram, error)
	// Int64ObservableCounter returns a new Int64ObservableCounter identified
	// by name and configured with options. The instrument is used to
	// asynchronously record increasing int64 measurements once per a
	// measurement collection cycle.
	//
	// Measurements for the returned instrument are made via a callback. Use
	// the WithInt64Callback option to register the callback here, or use the
	// RegisterCallback method of this Meter to register one later. See the
	// Measurements section of the package documentation for more information.
	Int64ObservableCounter(name string, options ...Int64ObservableCounterOption) (Int64ObservableCounter, error)
	// Int64ObservableUpDownCounter returns a new Int64ObservableUpDownCounter
	// instrument identified by name and configured with options. The
	// instrument is used to asynchronously record int64 measurements once per
	// a measurement collection cycle.
	//
	// Measurements for the returned instrument are made via a callback. Use
	// the WithInt64Callback option to register the callback here, or use the
	// RegisterCallback method of this Meter to register one later. See the
	// Measurements section of the package documentation for more information.
	Int64ObservableUpDownCounter(name string, options ...Int64ObservableUpDownCounterOption) (Int64ObservableUpDownCounter, error)
	// Int64ObservableGauge returns a new Int64ObservableGauge instrument
	// identified by name and configured with options. The instrument is used
	// to asynchronously record instantaneous int64 measurements once per a
	// measurement collection cycle.
	//
	// Measurements for the returned instrument are made via a callback. Use
	// the WithInt64Callback option to register the callback here, or use the
	// RegisterCallback method of this Meter to register one later. See the
	// Measurements section of the package documentation for more information.
	Int64ObservableGauge(name string, options ...Int64ObservableGaugeOption) (Int64ObservableGauge, error)

	// Float64Counter returns a new Float64Counter instrument identified by
	// name and configured with options. The instrument is used to
	// synchronously record increasing float64 measurements during a
	// computational operation.
	Float64Counter(name string, options ...Float64CounterOption) (Float64Counter, error)
	// Float64UpDownCounter returns a new Float64UpDownCounter instrument
	// identified by name and configured with options. The instrument is used
	// to synchronously record float64 measurements during a computational
	// operation.
	Float64UpDownCounter(name string, options ...Float64UpDownCounterOption) (Float64UpDownCounter, error)
	// Float64Histogram returns a new Float64Histogram instrument identified by
	// name and configured with options. The instrument is used to
	// synchronously record the distribution of float64 measurements during a
	// computational operation.
	Float64Histogram(name string, options ...Float64HistogramOption) (Float64Histogram, error)
	// Float64ObservableCounter returns a new Float64ObservableCounter
	// instrument identified by name and configured with options. The
	// instrument is used to asynchronously record increasing float64
	// measurements once per a measurement collection cycle.
	//
	// Measurements for the returned instrument are made via a callback. Use
	// the WithFloat64Callback option to register the callback here, or use the
	// RegisterCallback method of this Meter to register one later. See the
	// Measurements section of the package documentation for more information.
	Float64ObservableCounter(name string, options ...Float64ObservableCounterOption) (Float64ObservableCounter, error)
	// Float64ObservableUpDownCounter returns a new
	// Float64ObservableUpDownCounter instrument identified by name and
	// configured with options. The instrument is used to asynchronously record
	// float64 measurements once per a measurement collection cycle.
	//
	// Measurements for the returned instrument are made via a callback. Use
	// the WithFloat64Callback option to register the callback here, or use the
	// RegisterCallback method of this Meter to register one later. See the
	// Measurements section of the package documentation for more information.
	Float64ObservableUpDownCounter(name string, options ...Float64ObservableUpDownCounterOption) (Float64ObservableUpDownCounter, error)
	// Float64ObservableGauge returns a new Float64ObservableGauge instrument
	// identified by name and configured with options. The instrument is used
	// to asynchronously record instantaneous float64 measurements once per a
	// measurement collection cycle.
	//
	// Measurements for the returned instrument are made via a callback. Use
	// the WithFloat64Callback option to register the callback here, or use the
	// RegisterCallback method of this Meter to register one later. See the
	// Measurements section of the package documentation for more information.
	Float64ObservableGauge(name string, options ...Float64ObservableGaugeOption) (Float64ObservableGauge, error)

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
	RegisterCallback(f Callback, instruments ...Observable) (Registration, error)
}

// Callback is a function registered with a Meter that makes observations for
// the set of instruments it is registered with. The Observer parameter is used
// to record measurement observations for these instruments.
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
//
// Warning: Methods may be added to this interface in minor releases. See
// package documentation on API implementation for information on how to set
// default behavior for unimplemented methods.
type Observer interface {
	// Users of the interface can ignore this. This embedded type is only used
	// by implementations of this interface. See the "API Implementations"
	// section of the package documentation for more information.
	embedded.Observer

	// ObserveFloat64 records the float64 value for obsrv.
	ObserveFloat64(obsrv Float64Observable, value float64, opts ...ObserveOption)
	// ObserveInt64 records the int64 value for obsrv.
	ObserveInt64(obsrv Int64Observable, value int64, opts ...ObserveOption)
}

// Registration is an token representing the unique registration of a callback
// for a set of instruments with a Meter.
//
// Warning: Methods may be added to this interface in minor releases. See
// package documentation on API implementation for information on how to set
// default behavior for unimplemented methods.
type Registration interface {
	// Users of the interface can ignore this. This embedded type is only used
	// by implementations of this interface. See the "API Implementations"
	// section of the package documentation for more information.
	embedded.Registration

	// Unregister removes the callback registration from a Meter.
	//
	// This method needs to be idempotent and concurrent safe.
	Unregister() error
}
