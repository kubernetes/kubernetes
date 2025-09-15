// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package trace // import "go.opentelemetry.io/otel/sdk/trace"

import "context"

// SpanExporter handles the delivery of spans to external receivers. This is
// the final component in the trace export pipeline.
type SpanExporter interface {
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// ExportSpans exports a batch of spans.
	//
	// This function is called synchronously, so there is no concurrency
	// safety requirement. However, due to the synchronous calling pattern,
	// it is critical that all timeouts and cancellations contained in the
	// passed context must be honored.
	//
	// Any retry logic must be contained in this function. The SDK that
	// calls this function will not implement any retry logic. All errors
	// returned by this function are considered unrecoverable and will be
	// reported to a configured error Handler.
	ExportSpans(ctx context.Context, spans []ReadOnlySpan) error
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// Shutdown notifies the exporter of a pending halt to operations. The
	// exporter is expected to perform any cleanup or synchronization it
	// requires while honoring all timeouts and cancellations contained in
	// the passed context.
	Shutdown(ctx context.Context) error
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.
}
