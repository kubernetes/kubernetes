// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otlptrace // import "go.opentelemetry.io/otel/exporters/otlp/otlptrace"

import (
	"context"

	tracepb "go.opentelemetry.io/proto/otlp/trace/v1"
)

// Client manages connections to the collector, handles the
// transformation of data into wire format, and the transmission of that
// data to the collector.
type Client interface {
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// Start should establish connection(s) to endpoint(s). It is
	// called just once by the exporter, so the implementation
	// does not need to worry about idempotence and locking.
	Start(ctx context.Context) error
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// Stop should close the connections. The function is called
	// only once by the exporter, so the implementation does not
	// need to worry about idempotence, but it may be called
	// concurrently with UploadTraces, so proper
	// locking is required. The function serves as a
	// synchronization point - after the function returns, the
	// process of closing connections is assumed to be finished.
	Stop(ctx context.Context) error
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// UploadTraces should transform the passed traces to the wire
	// format and send it to the collector. May be called
	// concurrently.
	UploadTraces(ctx context.Context, protoSpans []*tracepb.ResourceSpans) error
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.
}
