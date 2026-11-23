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

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"k8s.io/klog/v2"
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
	return ContextWithSpanLogger(utiltrace.ContextWithTrace(ctx, utilSpan), otelSpan), &Span{
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
	if s == nil {
		return ctx
	}
	return ContextWithSpanLogger(trace.ContextWithSpan(utiltrace.ContextWithTrace(ctx, s.utilSpan), s.otelSpan), s.otelSpan)
}

// ContextWithSpanLogger updates the logger in ctx with the span's trace_id and
// span_id when span is recording and sampled. If ctx already carries a span
// logger from an outer span, the previous trace_id and span_id are replaced
// (or cleared if span is nil, not recording, or not sampled) while preserving
// any other values or names added to the logger.
func ContextWithSpanLogger(ctx context.Context, span trace.Span) context.Context {
	if ctx == nil {
		return nil
	}

	var traceID trace.TraceID
	var spanID trace.SpanID
	hasSpan := false
	if span != nil && span.IsRecording() {
		if spanCtx := span.SpanContext(); spanCtx.IsValid() && spanCtx.IsSampled() {
			traceID = spanCtx.TraceID()
			spanID = spanCtx.SpanID()
			hasSpan = true
		}
	}

	logger := klog.FromContext(ctx)
	sink := logger.GetSink()
	if sink == nil {
		return ctx
	}

	if existing, ok := sink.(*spanLogSink); ok {
		if existing.hasSpan == hasSpan && (!hasSpan || (existing.traceID == traceID && existing.spanID == spanID)) {
			return ctx
		}
		return klog.NewContext(ctx, logger.WithSink(&spanLogSink{
			sink:    existing.sink,
			traceID: traceID,
			spanID:  spanID,
			hasSpan: hasSpan,
		}))
	}

	if !hasSpan {
		return ctx
	}

	if cd, ok := sink.(logr.CallDepthLogSink); ok {
		sink = cd.WithCallDepth(1)
	}
	return klog.NewContext(ctx, logger.WithSink(&spanLogSink{
		sink:    sink,
		traceID: traceID,
		spanID:  spanID,
		hasSpan: true,
	}))
}

var (
	_ logr.LogSink                = (*spanLogSink)(nil)
	_ logr.CallDepthLogSink       = (*spanLogSink)(nil)
	_ logr.CallStackHelperLogSink = (*spanLogSink)(nil)
)

// spanLogSink wraps a logr.LogSink to inject the active span's trace_id and
// span_id at log time rather than via WithValues. This avoids emitting
// duplicate keys in loggers that do not deduplicate WithValues (such as zapr
// JSON logging) and allows nested spans to replace or clear the active IDs
// without losing other values or names added to the logger.
type spanLogSink struct {
	sink    logr.LogSink
	traceID trace.TraceID
	spanID  trace.SpanID
	hasSpan bool
}

func noopHelper() {}

// Init is a no-op because spanLogSink only wraps an already-initialized sink
// (with call depth adjusted once on wrap) that may be shared across derived
// contexts. Forwarding Init would both race with concurrent loggers and
// corrupt call depth when round-tripped through logr.New (e.g. via slog).
func (s *spanLogSink) Init(logr.RuntimeInfo) {}

func (s *spanLogSink) Enabled(level int) bool {
	return s.sink.Enabled(level)
}

func (s *spanLogSink) Info(level int, msg string, keysAndValues ...any) {
	if h, ok := s.sink.(logr.CallStackHelperLogSink); ok {
		h.GetCallStackHelper()()
	}
	if !s.hasSpan {
		s.sink.Info(level, msg, keysAndValues...)
		return
	}
	if len(keysAndValues) == 0 {
		s.sink.Info(level, msg, "trace_id", s.traceID, "span_id", s.spanID)
		return
	}
	s.sink.Info(level, msg, s.withSpanKVs(keysAndValues)...)
}

func (s *spanLogSink) Error(err error, msg string, keysAndValues ...any) {
	if h, ok := s.sink.(logr.CallStackHelperLogSink); ok {
		h.GetCallStackHelper()()
	}
	if !s.hasSpan {
		s.sink.Error(err, msg, keysAndValues...)
		return
	}
	if len(keysAndValues) == 0 {
		s.sink.Error(err, msg, "trace_id", s.traceID, "span_id", s.spanID)
		return
	}
	s.sink.Error(err, msg, s.withSpanKVs(keysAndValues)...)
}

func (s *spanLogSink) WithValues(keysAndValues ...any) logr.LogSink {
	clone := *s
	clone.sink = s.sink.WithValues(keysAndValues...)
	return &clone
}

func (s *spanLogSink) WithName(name string) logr.LogSink {
	clone := *s
	clone.sink = s.sink.WithName(name)
	return &clone
}

func (s *spanLogSink) WithCallDepth(depth int) logr.LogSink {
	if cd, ok := s.sink.(logr.CallDepthLogSink); ok {
		clone := *s
		clone.sink = cd.WithCallDepth(depth)
		return &clone
	}
	return s
}

func (s *spanLogSink) GetCallStackHelper() func() {
	if h, ok := s.sink.(logr.CallStackHelperLogSink); ok {
		return h.GetCallStackHelper()
	}
	return noopHelper
}

func (s *spanLogSink) withSpanKVs(keysAndValues []any) []any {
	kvs := make([]any, 0, 4+len(keysAndValues))
	kvs = append(kvs, "trace_id", s.traceID, "span_id", s.spanID)
	kvs = append(kvs, keysAndValues...)
	return kvs
}
