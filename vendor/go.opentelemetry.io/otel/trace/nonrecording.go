// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package trace

// nonRecordingSpan is a minimal implementation of a Span that wraps a
// SpanContext. It performs no operations other than to return the wrapped
// SpanContext.
type nonRecordingSpan struct {
	noopSpan

	sc SpanContext
}

// SpanContext returns the wrapped SpanContext.
func (s nonRecordingSpan) SpanContext() SpanContext { return s.sc }
