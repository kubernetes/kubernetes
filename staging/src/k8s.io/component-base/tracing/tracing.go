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
	"io"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"k8s.io/utils/clock"
	utiltrace "k8s.io/utils/trace"
)

const instrumentationScope = "k8s.io/component-base/tracing"

type contextKey struct{}

// Start creates spans using both OpenTelemetry, and the k8s.io/utils/trace package.
// It only creates an OpenTelemetry span if the incoming context already includes a span.
func Start(ctx context.Context, name string, attributes ...attribute.KeyValue) (context.Context, *Span) {
	// If the incoming context already includes an OpenTelemetry span, create a child span with the provided name and attributes.
	// If the caller is not using OpenTelemetry, or has tracing disabled (e.g. with a component-specific feature flag), this is a noop.
	ctx, otelSpan := trace.SpanFromContext(ctx).TracerProvider().Tracer(instrumentationScope).Start(ctx, name, trace.WithAttributes(attributes...))
	// If there is already a utiltrace span in the context, use that as our parent span.
	utilSpan := utiltrace.FromContext(ctx).Nest(name, attributesToFields(attributes)...)
	
	span := &Span{
		otelSpan: otelSpan,
		utilSpan: utilSpan,
		clock:    clock.RealClock{},
	}
	// Set the trace as active in the context so that subsequent Start calls create nested spans.
	return ContextWithSpan(ctx, span), span
}

// Span is a component part of a trace. It represents a single named
// and timed operation of a workflow being observed.
// This Span is a combination of an OpenTelemetry and k8s.io/utils/trace span
// to facilitate the migration to OpenTelemetry.
type Span struct {
	otelSpan trace.Span
	utilSpan *utiltrace.Trace

	layers      []*layer
	activeLayer *layer
	clock       clock.PassiveClock
}

type layer struct {
	name          string
	duration      time.Duration
	childDuration time.Duration
	bytes         int64
	count         int64
	child         *layer
}

// AddEvent adds a point-in-time event with a name and attributes.
func (s *Span) AddEvent(name string, attributes ...attribute.KeyValue) {
	s.otelSpan.AddEvent(name, trace.WithAttributes(attributes...))
	if s.utilSpan != nil {
		s.utilSpan.Step(name, attributesToFields(attributes)...)
	}
}

// WrapWriter wraps the given writer to track writes as a layer in the span.
func (s *Span) WrapWriter(w io.Writer, name string) io.Writer {
	l := &layer{name: name}
	tw := &TracedWriter{
		next:  w,
		layer: l,
		span:  s,
	}

	// Attempt to find a child layer if the writer is a TracedWriter or wraps one.
	// This allows us to calculate exclusive time without dynamic tracking in Write.
	if unwrapper, ok := w.(interface{ Unwrap() io.Writer }); ok {
		if childTw, ok := w.(*TracedWriter); ok {
			l.child = childTw.layer
		} else if childTw, ok := unwrapper.Unwrap().(*TracedWriter); ok {
			l.child = childTw.layer
		}
	} else if childTw, ok := w.(*TracedWriter); ok {
		l.child = childTw.layer
	}

	s.layers = append(s.layers, l)
	return tw
}

// WrapReader wraps the given reader to track reads as a layer in the span.
func (s *Span) WrapReader(r io.Reader, name string) io.Reader {
	l := &layer{name: name}
	tr := &TracedReader{
		next:  r,
		layer: l,
		span:  s,
	}

	// Attempt to find a child layer if the reader is a TracedReader or wraps one.
	// This allows us to calculate exclusive time without dynamic tracking in Read.
	if unwrapper, ok := r.(interface{ Unwrap() io.Reader }); ok {
		if childTr, ok := r.(*TracedReader); ok {
			l.child = childTr.layer
		} else if childTr, ok := unwrapper.Unwrap().(*TracedReader); ok {
			l.child = childTr.layer
		}
	} else if childTr, ok := r.(*TracedReader); ok {
		l.child = childTr.layer
	}

	s.layers = append(s.layers, l)
	return tr
}

// WithReader creates a new ReaderSpan for tracking a logical step in reading.
// It wraps the provided reader to track the IO time separately from the processing time.
func (s *Span) WithReader(name string, r io.Reader) (io.Reader, *ReaderSpan) {
	// Check if there is already a traced layer
	var childLayer *layer
	for current := r; current != nil; {
		if tr, ok := current.(*TracedReader); ok {
			childLayer = tr.layer
			break
		}
		if unwrapper, ok := current.(interface{ Unwrap() io.Reader }); ok {
			current = unwrapper.Unwrap()
		} else {
			break
		}
	}

	if childLayer != nil {
		l := &layer{name: name, child: childLayer}
		s.layers = append(s.layers, l)
		return r, &ReaderSpan{
			span:  s,
			layer: l,
			start: s.clock.Now(),
		}
	}

	// Layer for the IO part
	ioLayer := &layer{name: "Reader"}
	s.layers = append(s.layers, ioLayer)

	tr := &TracedReader{
		next:  r,
		layer: ioLayer,
		span:  s,
	}

	// Layer for the wrapper (e.g. Deserialize)
	l := &layer{name: name, child: ioLayer}
	s.layers = append(s.layers, l)

	return tr, &ReaderSpan{
		span:  s,
		layer: l,
		start: s.clock.Now(),
	}
}

// WithWriter creates a new WriterSpan for tracking a logical step in writing.
// It wraps the provided writer to track the IO time separately from the processing time.
func (s *Span) WithWriter(name string, w io.Writer) (io.Writer, *WriterSpan) {
	// Check if there is already a traced layer
	var childLayer *layer
	for current := w; current != nil; {
		if tw, ok := current.(*TracedWriter); ok {
			childLayer = tw.layer
			break
		}
		if unwrapper, ok := current.(interface{ Unwrap() io.Writer }); ok {
			current = unwrapper.Unwrap()
		} else {
			break
		}
	}

	if childLayer != nil {
		l := &layer{name: name, child: childLayer}
		s.layers = append(s.layers, l)
		return w, &WriterSpan{
			span:  s,
			layer: l,
			start: s.clock.Now(),
		}
	}

	// Layer for the IO part
	ioLayer := &layer{name: "Writer"}
	s.layers = append(s.layers, ioLayer)

	tw := &TracedWriter{
		next:  w,
		layer: ioLayer,
		span:  s,
	}

	// Layer for the wrapper (e.g. Serialize)
	l := &layer{name: name, child: ioLayer}
	s.layers = append(s.layers, l)

	return tw, &WriterSpan{
		span:  s,
		layer: l,
		start: s.clock.Now(),
	}
}

// End ends the span, and logs if the span duration is greater than the logThreshold.
func (s *Span) End(logThreshold time.Duration) {
	s.reportLayers()
	s.otelSpan.End()
	if s.utilSpan != nil {
		s.utilSpan.LogIfLong(logThreshold)
	}
}

func (s *Span) reportLayers() {

	var layers []string
	for _, l := range s.layers {
		layers = append(layers, l.name)
		
		var exclusive time.Duration
		if l.childDuration > 0 {
			exclusive = l.duration - l.childDuration
		} else if l.child != nil {
			exclusive = l.duration - l.child.duration
		} else {
			exclusive = l.duration
		}
		
		if exclusive < 0 {
			exclusive = 0
		}

		// If this layer has the same bytes and count as the previous layer,
		// it means it's just a wrapper that didn't add any writes.
		// We report 0 for size and count to avoid repetition, but keep duration.
		reportBytes := l.bytes
		reportCount := l.count


		var attrs []attribute.KeyValue
		if reportBytes > 0 || reportCount > 0 {
			attrs = append(attrs, attribute.Int64("size", reportBytes), attribute.Int64("count", reportCount))
		}

		otelAttrs := append(attrs, attribute.String("duration", exclusive.String()))
		s.otelSpan.AddEvent(l.name, trace.WithAttributes(otelAttrs...))
		if s.utilSpan != nil {
			s.utilSpan.ParallelStep(l.name, exclusive, attributesToFields(attrs)...)
		}

	}
	if len(layers) > 0 {
		s.otelSpan.SetAttributes(attribute.StringSlice("writer.layers", layers))
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
	if s, ok := ctx.Value(contextKey{}).(*Span); ok {
		return s
	}
	return &Span{
		otelSpan: trace.SpanFromContext(ctx),
		utilSpan: utiltrace.FromContext(ctx),
		clock:    clock.RealClock{},
	}
}

// ContextWithSpan returns a context with the Span included in the context.
func ContextWithSpan(ctx context.Context, s *Span) context.Context {
	ctx = context.WithValue(ctx, contextKey{}, s)
	return trace.ContextWithSpan(utiltrace.ContextWithTrace(ctx, s.utilSpan), s.otelSpan)
}
