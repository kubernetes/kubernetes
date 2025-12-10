// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package noop provides an implementation of the OpenTelemetry trace API that
// produces no telemetry and minimizes used computation resources.
//
// Using this package to implement the OpenTelemetry trace API will effectively
// disable OpenTelemetry.
//
// This implementation can be embedded in other implementations of the
// OpenTelemetry trace API. Doing so will mean the implementation defaults to
// no operation for methods it does not implement.
package noop // import "go.opentelemetry.io/otel/trace/noop"

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/embedded"
)

var (
	// Compile-time check this implements the OpenTelemetry API.

	_ trace.TracerProvider = TracerProvider{}
	_ trace.Tracer         = Tracer{}
	_ trace.Span           = Span{}
)

// TracerProvider is an OpenTelemetry No-Op TracerProvider.
type TracerProvider struct{ embedded.TracerProvider }

// NewTracerProvider returns a TracerProvider that does not record any telemetry.
func NewTracerProvider() TracerProvider {
	return TracerProvider{}
}

// Tracer returns an OpenTelemetry Tracer that does not record any telemetry.
func (TracerProvider) Tracer(string, ...trace.TracerOption) trace.Tracer {
	return Tracer{}
}

// Tracer is an OpenTelemetry No-Op Tracer.
type Tracer struct{ embedded.Tracer }

// Start creates a span. The created span will be set in a child context of ctx
// and returned with the span.
//
// If ctx contains a span context, the returned span will also contain that
// span context. If the span context in ctx is for a non-recording span, that
// span instance will be returned directly.
func (t Tracer) Start(ctx context.Context, _ string, _ ...trace.SpanStartOption) (context.Context, trace.Span) {
	span := trace.SpanFromContext(ctx)

	// If the parent context contains a non-zero span context, that span
	// context needs to be returned as a non-recording span
	// (https://github.com/open-telemetry/opentelemetry-specification/blob/3a1dde966a4ce87cce5adf464359fe369741bbea/specification/trace/api.md#behavior-of-the-api-in-the-absence-of-an-installed-sdk).
	var zeroSC trace.SpanContext
	if sc := span.SpanContext(); !sc.Equal(zeroSC) {
		if !span.IsRecording() {
			// If the span is not recording return it directly.
			return ctx, span
		}
		// Otherwise, return the span context needs in a non-recording span.
		span = Span{sc: sc}
	} else {
		// No parent, return a No-Op span with an empty span context.
		span = noopSpanInstance
	}
	return trace.ContextWithSpan(ctx, span), span
}

var noopSpanInstance trace.Span = Span{}

// Span is an OpenTelemetry No-Op Span.
type Span struct {
	embedded.Span

	sc trace.SpanContext
}

// SpanContext returns an empty span context.
func (s Span) SpanContext() trace.SpanContext { return s.sc }

// IsRecording always returns false.
func (Span) IsRecording() bool { return false }

// SetStatus does nothing.
func (Span) SetStatus(codes.Code, string) {}

// SetAttributes does nothing.
func (Span) SetAttributes(...attribute.KeyValue) {}

// End does nothing.
func (Span) End(...trace.SpanEndOption) {}

// RecordError does nothing.
func (Span) RecordError(error, ...trace.EventOption) {}

// AddEvent does nothing.
func (Span) AddEvent(string, ...trace.EventOption) {}

// AddLink does nothing.
func (Span) AddLink(trace.Link) {}

// SetName does nothing.
func (Span) SetName(string) {}

// TracerProvider returns a No-Op TracerProvider.
func (Span) TracerProvider() trace.TracerProvider { return TracerProvider{} }
