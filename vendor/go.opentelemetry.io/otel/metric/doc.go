// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

/*
Package metric provides the OpenTelemetry API used to measure metrics about
source code operation.

This API is separate from its implementation so the instrumentation built from
it is reusable. See [go.opentelemetry.io/otel/sdk/metric] for the official
OpenTelemetry implementation of this API.

All measurements made with this package are made via instruments. These
instruments are created by a [Meter] which itself is created by a
[MeterProvider]. Applications need to accept a [MeterProvider] implementation
as a starting point when instrumenting. This can be done directly, or by using
the OpenTelemetry global MeterProvider via [GetMeterProvider]. Using an
appropriately named [Meter] from the accepted [MeterProvider], instrumentation
can then be built from the [Meter]'s instruments.

# Instruments

Each instrument is designed to make measurements of a particular type. Broadly,
all instruments fall into two overlapping logical categories: asynchronous or
synchronous, and int64 or float64.

All synchronous instruments ([Int64Counter], [Int64UpDownCounter],
[Int64Histogram], [Float64Counter], [Float64UpDownCounter], and
[Float64Histogram]) are used to measure the operation and performance of source
code during the source code execution. These instruments only make measurements
when the source code they instrument is run.

All asynchronous instruments ([Int64ObservableCounter],
[Int64ObservableUpDownCounter], [Int64ObservableGauge],
[Float64ObservableCounter], [Float64ObservableUpDownCounter], and
[Float64ObservableGauge]) are used to measure metrics outside of the execution
of source code. They are said to make "observations" via a callback function
called once every measurement collection cycle.

Each instrument is also grouped by the value type it measures. Either int64 or
float64. The value being measured will dictate which instrument in these
categories to use.

Outside of these two broad categories, instruments are described by the
function they are designed to serve. All Counters ([Int64Counter],
[Float64Counter], [Int64ObservableCounter], and [Float64ObservableCounter]) are
designed to measure values that never decrease in value, but instead only
incrementally increase in value. UpDownCounters ([Int64UpDownCounter],
[Float64UpDownCounter], [Int64ObservableUpDownCounter], and
[Float64ObservableUpDownCounter]) on the other hand, are designed to measure
values that can increase and decrease. When more information needs to be
conveyed about all the synchronous measurements made during a collection cycle,
a Histogram ([Int64Histogram] and [Float64Histogram]) should be used. Finally,
when just the most recent measurement needs to be conveyed about an
asynchronous measurement, a Gauge ([Int64ObservableGauge] and
[Float64ObservableGauge]) should be used.

See the [OpenTelemetry documentation] for more information about instruments
and their intended use.

# Instrument Name

OpenTelemetry defines an [instrument name syntax] that restricts what
instrument names are allowed.

Instrument names should ...

  - Not be empty.
  - Have an alphabetic character as their first letter.
  - Have any letter after the first be an alphanumeric character, ‘_’, ‘.’,
    ‘-’, or ‘/’.
  - Have a maximum length of 255 letters.

To ensure compatibility with observability platforms, all instruments created
need to conform to this syntax. Not all implementations of the API will validate
these names, it is the callers responsibility to ensure compliance.

# Measurements

Measurements are made by recording values and information about the values with
an instrument. How these measurements are recorded depends on the instrument.

Measurements for synchronous instruments ([Int64Counter], [Int64UpDownCounter],
[Int64Histogram], [Float64Counter], [Float64UpDownCounter], and
[Float64Histogram]) are recorded using the instrument methods directly. All
counter instruments have an Add method that is used to measure an increment
value, and all histogram instruments have a Record method to measure a data
point.

Asynchronous instruments ([Int64ObservableCounter],
[Int64ObservableUpDownCounter], [Int64ObservableGauge],
[Float64ObservableCounter], [Float64ObservableUpDownCounter], and
[Float64ObservableGauge]) record measurements within a callback function. The
callback is registered with the Meter which ensures the callback is called once
per collection cycle. A callback can be registered two ways: during the
instrument's creation using an option, or later using the RegisterCallback
method of the [Meter] that created the instrument.

If the following criteria are met, an option ([WithInt64Callback] or
[WithFloat64Callback]) can be used during the asynchronous instrument's
creation to register a callback ([Int64Callback] or [Float64Callback],
respectively):

  - The measurement process is known when the instrument is created
  - Only that instrument will make a measurement within the callback
  - The callback never needs to be unregistered

If the criteria are not met, use the RegisterCallback method of the [Meter] that
created the instrument to register a [Callback].

# API Implementations

This package does not conform to the standard Go versioning policy, all of its
interfaces may have methods added to them without a package major version bump.
This non-standard API evolution could surprise an uninformed implementation
author. They could unknowingly build their implementation in a way that would
result in a runtime panic for their users that update to the new API.

The API is designed to help inform an instrumentation author about this
non-standard API evolution. It requires them to choose a default behavior for
unimplemented interface methods. There are three behavior choices they can
make:

  - Compilation failure
  - Panic
  - Default to another implementation

All interfaces in this API embed a corresponding interface from
[go.opentelemetry.io/otel/metric/embedded]. If an author wants the default
behavior of their implementations to be a compilation failure, signaling to
their users they need to update to the latest version of that implementation,
they need to embed the corresponding interface from
[go.opentelemetry.io/otel/metric/embedded] in their implementation. For
example,

	import "go.opentelemetry.io/otel/metric/embedded"

	type MeterProvider struct {
		embedded.MeterProvider
		// ...
	}

If an author wants the default behavior of their implementations to a panic,
they need to embed the API interface directly.

	import "go.opentelemetry.io/otel/metric"

	type MeterProvider struct {
		metric.MeterProvider
		// ...
	}

This is not a recommended behavior as it could lead to publishing packages that
contain runtime panics when users update other package that use newer versions
of [go.opentelemetry.io/otel/metric].

Finally, an author can embed another implementation in theirs. The embedded
implementation will be used for methods not defined by the author. For example,
an author who wants to default to silently dropping the call can use
[go.opentelemetry.io/otel/metric/noop]:

	import "go.opentelemetry.io/otel/metric/noop"

	type MeterProvider struct {
		noop.MeterProvider
		// ...
	}

It is strongly recommended that authors only embed
[go.opentelemetry.io/otel/metric/noop] if they choose this default behavior.
That implementation is the only one OpenTelemetry authors can guarantee will
fully implement all the API interfaces when a user updates their API.

[instrument name syntax]: https://opentelemetry.io/docs/specs/otel/metrics/api/#instrument-name-syntax
[OpenTelemetry documentation]: https://opentelemetry.io/docs/concepts/signals/metrics/
[GetMeterProvider]: https://pkg.go.dev/go.opentelemetry.io/otel#GetMeterProvider
*/
package metric // import "go.opentelemetry.io/otel/metric"
