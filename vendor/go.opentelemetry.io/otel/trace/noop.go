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

package trace // import "go.opentelemetry.io/otel/trace"

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
)

// NewNoopTracerProvider returns an implementation of TracerProvider that
// performs no operations. The Tracer and Spans created from the returned
// TracerProvider also perform no operations.
func NewNoopTracerProvider() TracerProvider {
	return noopTracerProvider{}
}

type noopTracerProvider struct{}

var _ TracerProvider = noopTracerProvider{}

// Tracer returns noop implementation of Tracer.
func (p noopTracerProvider) Tracer(string, ...TracerOption) Tracer {
	return noopTracer{}
}

// noopTracer is an implementation of Tracer that performs no operations.
type noopTracer struct{}

var _ Tracer = noopTracer{}

// Start carries forward a non-recording Span, if one is present in the context, otherwise it
// creates a no-op Span.
func (t noopTracer) Start(ctx context.Context, name string, _ ...SpanStartOption) (context.Context, Span) {
	span := SpanFromContext(ctx)
	if _, ok := span.(nonRecordingSpan); !ok {
		// span is likely already a noopSpan, but let's be sure
		span = noopSpan{}
	}
	return ContextWithSpan(ctx, span), span
}

// noopSpan is an implementation of Span that performs no operations.
type noopSpan struct{}

var _ Span = noopSpan{}

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

// SetName does nothing.
func (noopSpan) SetName(string) {}

// TracerProvider returns a no-op TracerProvider.
func (noopSpan) TracerProvider() TracerProvider { return noopTracerProvider{} }
