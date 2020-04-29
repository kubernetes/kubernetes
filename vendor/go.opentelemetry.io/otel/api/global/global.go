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

package global

import (
	"go.opentelemetry.io/otel/api/global/internal"
	"go.opentelemetry.io/otel/api/metric"
	"go.opentelemetry.io/otel/api/propagation"
	"go.opentelemetry.io/otel/api/trace"
)

// Tracer creates a named tracer that implements Tracer interface.
// If the name is an empty string then provider uses default name.
//
// This is short for TraceProvider().Tracer(name)
func Tracer(name string) trace.Tracer {
	return TraceProvider().Tracer(name)
}

// TraceProvider returns the registered global trace provider.
// If none is registered then an instance of trace.NoopProvider is returned.
//
// Use the trace provider to create a named tracer. E.g.
//     tracer := global.TraceProvider().Tracer("example.com/foo")
// or
//     tracer := global.Tracer("example.com/foo")
func TraceProvider() trace.Provider {
	return internal.TraceProvider()
}

// SetTraceProvider registers `tp` as the global trace provider.
func SetTraceProvider(tp trace.Provider) {
	internal.SetTraceProvider(tp)
}

// Meter gets a named Meter interface.  If the name is an
// empty string, the provider uses a default name.
//
// This is short for MeterProvider().Meter(name)
func Meter(name string) metric.Meter {
	return MeterProvider().Meter(name)
}

// MeterProvider returns the registered global meter provider.  If
// none is registered then a default meter provider is returned that
// forwards the Meter interface to the first registered Meter.
//
// Use the meter provider to create a named meter. E.g.
//     meter := global.MeterProvider().Meter("example.com/foo")
// or
//     meter := global.Meter("example.com/foo")
func MeterProvider() metric.Provider {
	return internal.MeterProvider()
}

// SetMeterProvider registers `mp` as the global meter provider.
func SetMeterProvider(mp metric.Provider) {
	internal.SetMeterProvider(mp)
}

// Propagators returns the registered global propagators instance.  If
// none is registered then an instance of propagators.NoopPropagators
// is returned.
func Propagators() propagation.Propagators {
	return internal.Propagators()
}

// SetPropagators registers `p` as the global propagators instance.
func SetPropagators(p propagation.Propagators) {
	internal.SetPropagators(p)
}
