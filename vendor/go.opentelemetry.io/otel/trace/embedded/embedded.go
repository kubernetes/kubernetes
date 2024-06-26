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

// Package embedded provides interfaces embedded within the [OpenTelemetry
// trace API].
//
// Implementers of the [OpenTelemetry trace API] can embed the relevant type
// from this package into their implementation directly. Doing so will result
// in a compilation error for users when the [OpenTelemetry trace API] is
// extended (which is something that can happen without a major version bump of
// the API package).
//
// [OpenTelemetry trace API]: https://pkg.go.dev/go.opentelemetry.io/otel/trace
package embedded // import "go.opentelemetry.io/otel/trace/embedded"

// TracerProvider is embedded in
// [go.opentelemetry.io/otel/trace.TracerProvider].
//
// Embed this interface in your implementation of the
// [go.opentelemetry.io/otel/trace.TracerProvider] if you want users to
// experience a compilation error, signaling they need to update to your latest
// implementation, when the [go.opentelemetry.io/otel/trace.TracerProvider]
// interface is extended (which is something that can happen without a major
// version bump of the API package).
type TracerProvider interface{ tracerProvider() }

// Tracer is embedded in [go.opentelemetry.io/otel/trace.Tracer].
//
// Embed this interface in your implementation of the
// [go.opentelemetry.io/otel/trace.Tracer] if you want users to experience a
// compilation error, signaling they need to update to your latest
// implementation, when the [go.opentelemetry.io/otel/trace.Tracer] interface
// is extended (which is something that can happen without a major version bump
// of the API package).
type Tracer interface{ tracer() }

// Span is embedded in [go.opentelemetry.io/otel/trace.Span].
//
// Embed this interface in your implementation of the
// [go.opentelemetry.io/otel/trace.Span] if you want users to experience a
// compilation error, signaling they need to update to your latest
// implementation, when the [go.opentelemetry.io/otel/trace.Span] interface is
// extended (which is something that can happen without a major version bump of
// the API package).
type Span interface{ span() }
