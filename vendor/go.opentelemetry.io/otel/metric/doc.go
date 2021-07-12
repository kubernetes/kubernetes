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

/*
Package metric provides an implementation of the metrics part of the
OpenTelemetry API.

This package is currently in a pre-GA phase. Backwards incompatible changes
may be introduced in subsequent minor version releases as we work to track the
evolving OpenTelemetry specification and user feedback.

Measurements can be made about an operation being performed or the state of a
system in general. These measurements can be crucial to the reliable operation
of code and provide valuable insights about the inner workings of a system.

Measurements are made using instruments provided by this package. The type of
instrument used will depend on the type of measurement being made and of what
part of a system is being measured.

Instruments are categorized as Synchronous or Asynchronous and independently
as Adding or Grouping. Synchronous instruments are called by the user with a
Context. Asynchronous instruments are called by the SDK during collection.
Additive instruments are semantically intended for capturing a sum. Grouping
instruments are intended for capturing a distribution.

Additive instruments may be monotonic, in which case they are non-decreasing
and naturally define a rate.

The synchronous instrument names are:

  Counter:           additive, monotonic
  UpDownCounter:     additive
  ValueRecorder:     grouping

and the asynchronous instruments are:

  SumObserver:       additive, monotonic
  UpDownSumObserver: additive
  ValueObserver:     grouping

All instruments are provided with support for either float64 or int64 input
values.

An instrument is created using a Meter. Additionally, a Meter is used to
record batches of synchronous measurements or asynchronous observations. A
Meter is obtained using a MeterProvider. A Meter, like a Tracer, is unique to
the instrumentation it instruments and must be named and versioned when
created with a MeterProvider with the name and version of the instrumentation
library.

Instrumentation should be designed to accept a MeterProvider from which it can
create its own unique Meter. Alternatively, the registered global
MeterProvider from the go.opentelemetry.io/otel package can be used as a
default.
*/
package metric // import "go.opentelemetry.io/otel/metric"
