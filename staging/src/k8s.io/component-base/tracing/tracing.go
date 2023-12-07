/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tracing

import (
	"context"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	utiltrace "k8s.io/utils/trace"
)

const instrumentationScope = "k8s.io/component-base/tracing"

// Start creates spans using both OpenTelemetry, and the k8s.io/utils/trace package.
// It only creates an OpenTelemetry span if the incoming context already includes a span.
func Start(ctx context.Context, name string, attributes ...attribute.KeyValue) (context.Context, *Span) {
	// If the incoming context already includes an OpenTelemetry span, create a child span with the provided name and attributes.
	// If the caller is not using OpenTelemetry, or has tracing disabled (e.g. with a component-specific feature flag), this is a noop.
	ctx, otelSpan := trace.SpanFromContext(ctx).TracerProvider().Tracer(instrumentationScope).Start(ctx, name, trace.WithAttributes(attributes...))
	// If there is already a utiltrace span in the context, use that as our parent span.
	utilSpan := utiltrace.FromContext(ctx).Nest(name, attributesToFields(attributes)...)
	// Set the trace as active in the context so that subsequent Start calls create nested spans.
	return utiltrace.ContextWithTrace(ctx, utilSpan), &Span{
		otelSpan: otelSpan,
		utilSpan: utilSpan,
	}
}

// Span is a component part of a trace. It represents a single named
// and timed operation of a workflow being observed.
// This Span is a combination of an OpenTelemetry and k8s.io/utils/trace span
// to facilitate the migration to OpenTelemetry.
type Span struct {
	otelSpan trace.Span
	utilSpan *utiltrace.Trace
}

// AddEvent adds a point-in-time event with a name and attributes.
func (s *Span) AddEvent(name string, attributes ...attribute.KeyValue) {
	s.otelSpan.AddEvent(name, trace.WithAttributes(attributes...))
	if s.utilSpan != nil {
		s.utilSpan.Step(name, attributesToFields(attributes)...)
	}
}

// End ends the span, and logs if the span duration is greater than the logThreshold.
func (s *Span) End(logThreshold time.Duration) {
	s.otelSpan.End()
	if s.utilSpan != nil {
		s.utilSpan.LogIfLong(logThreshold)
	}
}

// RecordError will record err as an exception span event for this span.
// If this span is not being recorded or err is nil then this method does nothing.
func (s *Span) RecordError(err error, attributes ...attribute.KeyValue) {
	s.otelSpan.RecordError(err, trace.WithAttributes(attributes...))
}

func attributesToFields(attributes []attribute.KeyValue) []utiltrace.Field {
	fields := make([]utiltrace.Field, len(attributes))
	for i := range attributes {
		attr := attributes[i]
		fields[i] = utiltrace.Field{Key: string(attr.Key), Value: attr.Value.AsInterface()}
	}
	return fields
}

// SpanFromContext returns the *Span from the current context. It is composed of the active
// OpenTelemetry and k8s.io/utils/trace spans.
func SpanFromContext(ctx context.Context) *Span {
	return &Span{
		otelSpan: trace.SpanFromContext(ctx),
		utilSpan: utiltrace.FromContext(ctx),
	}
}

// ContextWithSpan returns a context with the Span included in the context.
func ContextWithSpan(ctx context.Context, s *Span) context.Context {
	return trace.ContextWithSpan(utiltrace.ContextWithTrace(ctx, s.utilSpan), s.otelSpan)
}
