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

package trace

import (
	"context"
	"time"

	"google.golang.org/grpc/codes"

	"go.opentelemetry.io/otel/api/core"
)

type NoopSpan struct {
}

var _ Span = (*NoopSpan)(nil)

// SpanContext returns an invalid span context.
func (NoopSpan) SpanContext() core.SpanContext {
	return core.EmptySpanContext()
}

// IsRecording always returns false for NoopSpan.
func (NoopSpan) IsRecording() bool {
	return false
}

// SetStatus does nothing.
func (NoopSpan) SetStatus(status codes.Code, msg string) {
}

// SetError does nothing.
func (NoopSpan) SetError(v bool) {
}

// SetAttributes does nothing.
func (NoopSpan) SetAttributes(attributes ...core.KeyValue) {
}

// End does nothing.
func (NoopSpan) End(options ...EndOption) {
}

// RecordError does nothing.
func (NoopSpan) RecordError(ctx context.Context, err error, opts ...ErrorOption) {
}

// Tracer returns noop implementation of Tracer.
func (NoopSpan) Tracer() Tracer {
	return NoopTracer{}
}

// AddEvent does nothing.
func (NoopSpan) AddEvent(ctx context.Context, name string, attrs ...core.KeyValue) {
}

// AddEventWithTimestamp does nothing.
func (NoopSpan) AddEventWithTimestamp(ctx context.Context, timestamp time.Time, name string, attrs ...core.KeyValue) {
}

// SetName does nothing.
func (NoopSpan) SetName(name string) {
}
