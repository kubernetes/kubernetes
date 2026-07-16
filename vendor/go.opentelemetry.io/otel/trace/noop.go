// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package trace // import "go.opentelemetry.io/otel/trace"

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace/embedded"
)

// NewNoopTracerProvider returns an implementation of TracerProvider that
// performs no operations. The Tracer and Spans created from the returned
// TracerProvider also perform no operations.
//
// Deprecated: Use [go.opentelemetry.io/otel/trace/noop.NewTracerProvider]
// instead.
func NewNoopTracerProvider() TracerProvider {
	return noopTracerProvider{}
}

type noopTracerProvider struct{ embedded.TracerProvider }

var _ TracerProvider = noopTracerProvider{}

// Tracer returns noop implementation of Tracer.
func (noopTracerProvider) Tracer(string, ...TracerOption) Tracer {
	return noopTracer{}
}

// noopTracer is an implementation of Tracer that performs no operations.
type noopTracer struct{ embedded.Tracer }

var _ Tracer = noopTracer{}

// Start carries forward a non-recording Span, if one is present in the context, otherwise it
// creates a no-op Span.
func (noopTracer) Start(ctx context.Context, _ string, _ ...SpanStartOption) (context.Context, Span) {
	span := SpanFromContext(ctx)
	if _, ok := span.(nonRecordingSpan); !ok {
		// span is likely already a noopSpan, but let's be sure
		span = noopSpanInstance
	}
	return ContextWithSpan(ctx, span), span
}

// noopSpan is an implementation of Span that performs no operations.
type noopSpan struct{ embedded.Span }

var noopSpanInstance Span = noopSpan{}

// SpanContext returns an empty span context.
func (noopSpan) SpanContext() SpanContext { return SpanContext{} }

// IsRecording always returns false.
func (noopSpan) IsRecording() bool { return false }

// SetStatus does nothing.
func (noopSpan) SetStatus(codes.Code, string) {}

// SetError does nothing.
func (noopSpan) SetError(bool) {}

// SetAttributes does nothing.
func (noopSpan) SetAttributes(...attribute.KeyValue) {}

// End does nothing.
func (noopSpan) End(...SpanEndOption) {}

// RecordError does nothing.
func (noopSpan) RecordError(error, ...EventOption) {}

// AddEvent does nothing.
func (noopSpan) AddEvent(string, ...EventOption) {}

// AddLink does nothing.
func (noopSpan) AddLink(Link) {}

// SetName does nothing.
func (noopSpan) SetName(string) {}

// TracerProvider returns a no-op TracerProvider.
func (s noopSpan) TracerProvider() TracerProvider {
	return s.tracerProvider(autoInstEnabled)
}

// autoInstEnabled defines if the auto-instrumentation SDK is enabled.
//
// The auto-instrumentation is expected to overwrite this value to true when it
// attaches to the process.
var autoInstEnabled = new(bool)

// tracerProvider return a noopTracerProvider if autoEnabled is false,
// otherwise it will return a TracerProvider from the sdk package used in
// auto-instrumentation.
//
//go:noinline
func (noopSpan) tracerProvider(autoEnabled *bool) TracerProvider {
	if *autoEnabled {
		return newAutoTracerProvider()
	}
	return noopTracerProvider{}
}
