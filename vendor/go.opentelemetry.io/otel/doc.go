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
Package otel provides global access to the OpenTelemetry API. The subpackages of
the otel package provide an implementation of the OpenTelemetry API.

The provided API is used to instrument code and measure data about that code's
performance and operation. The measured data, by default, is not processed or
transmitted anywhere. An implementation of the OpenTelemetry SDK, like the
default SDK implementation (go.opentelemetry.io/otel/sdk), and associated
exporters are used to process and transport this data.

To read the getting started guide, see https://opentelemetry.io/docs/languages/go/getting-started/.

To read more about tracing, see go.opentelemetry.io/otel/trace.

To read more about metrics, see go.opentelemetry.io/otel/metric.

To read more about propagation, see go.opentelemetry.io/otel/propagation and
go.opentelemetry.io/otel/baggage.
*/
package otel // import "go.opentelemetry.io/otel"
