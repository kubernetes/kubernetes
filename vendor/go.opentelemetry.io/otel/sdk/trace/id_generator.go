// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package trace // import "go.opentelemetry.io/otel/sdk/trace"

import (
	"context"
	"encoding/binary"
	"math/rand/v2"

	"go.opentelemetry.io/otel/trace"
)

// IDGenerator allows custom generators for TraceID and SpanID.
type IDGenerator interface {
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// NewIDs returns a new trace and span ID.
	NewIDs(ctx context.Context) (trace.TraceID, trace.SpanID)
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// NewSpanID returns a ID for a new span in the trace with traceID.
	NewSpanID(ctx context.Context, traceID trace.TraceID) trace.SpanID
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.
}

type randomIDGenerator struct{}

var _ IDGenerator = &randomIDGenerator{}

// NewSpanID returns a non-zero span ID from a randomly-chosen sequence.
func (gen *randomIDGenerator) NewSpanID(ctx context.Context, traceID trace.TraceID) trace.SpanID {
	sid := trace.SpanID{}
	for {
		binary.NativeEndian.PutUint64(sid[:], rand.Uint64())
		if sid.IsValid() {
			break
		}
	}
	return sid
}

// NewIDs returns a non-zero trace ID and a non-zero span ID from a
// randomly-chosen sequence.
func (gen *randomIDGenerator) NewIDs(ctx context.Context) (trace.TraceID, trace.SpanID) {
	tid := trace.TraceID{}
	sid := trace.SpanID{}
	for {
		binary.NativeEndian.PutUint64(tid[:8], rand.Uint64())
		binary.NativeEndian.PutUint64(tid[8:], rand.Uint64())
		if tid.IsValid() {
			break
		}
	}
	for {
		binary.NativeEndian.PutUint64(sid[:], rand.Uint64())
		if sid.IsValid() {
			break
		}
	}
	return tid, sid
}

func defaultIDGenerator() IDGenerator {
	return &randomIDGenerator{}
}
