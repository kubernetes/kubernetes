// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package propagation // import "go.opentelemetry.io/otel/propagation"

import (
	"context"
	"encoding/hex"
	"fmt"
	"strings"

	"go.opentelemetry.io/otel/trace"
)

const (
	supportedVersion  = 0
	maxVersion        = 254
	traceparentHeader = "traceparent"
	tracestateHeader  = "tracestate"
	delimiter         = "-"
)

// TraceContext is a propagator that supports the W3C Trace Context format
// (https://www.w3.org/TR/trace-context/)
//
// This propagator will propagate the traceparent and tracestate headers to
// guarantee traces are not broken. It is up to the users of this propagator
// to choose if they want to participate in a trace by modifying the
// traceparent header and relevant parts of the tracestate header containing
// their proprietary information.
type TraceContext struct{}

var (
	_           TextMapPropagator = TraceContext{}
	versionPart                   = fmt.Sprintf("%.2X", supportedVersion)
)

// Inject injects the trace context from ctx into carrier.
func (TraceContext) Inject(ctx context.Context, carrier TextMapCarrier) {
	sc := trace.SpanContextFromContext(ctx)
	if !sc.IsValid() {
		return
	}

	if ts := sc.TraceState().String(); ts != "" {
		carrier.Set(tracestateHeader, ts)
	}

	// Clear all flags other than the trace-context supported sampling bit.
	flags := sc.TraceFlags() & trace.FlagsSampled

	var sb strings.Builder
	sb.Grow(2 + 32 + 16 + 2 + 3)
	_, _ = sb.WriteString(versionPart)
	traceID := sc.TraceID()
	spanID := sc.SpanID()
	flagByte := [1]byte{byte(flags)}
	var buf [32]byte
	for _, src := range [][]byte{traceID[:], spanID[:], flagByte[:]} {
		_ = sb.WriteByte(delimiter[0])
		n := hex.Encode(buf[:], src)
		_, _ = sb.Write(buf[:n])
	}
	carrier.Set(traceparentHeader, sb.String())
}

// Extract reads tracecontext from the carrier into a returned Context.
//
// The returned Context will be a copy of ctx and contain the extracted
// tracecontext as the remote SpanContext. If the extracted tracecontext is
// invalid, the passed ctx will be returned directly instead.
func (tc TraceContext) Extract(ctx context.Context, carrier TextMapCarrier) context.Context {
	sc := tc.extract(carrier)
	if !sc.IsValid() {
		return ctx
	}
	return trace.ContextWithRemoteSpanContext(ctx, sc)
}

func (TraceContext) extract(carrier TextMapCarrier) trace.SpanContext {
	h := carrier.Get(traceparentHeader)
	if h == "" {
		return trace.SpanContext{}
	}

	var ver [1]byte
	if !extractPart(ver[:], &h, 2) {
		return trace.SpanContext{}
	}
	version := int(ver[0])
	if version > maxVersion {
		return trace.SpanContext{}
	}

	var scc trace.SpanContextConfig
	if !extractPart(scc.TraceID[:], &h, 32) {
		return trace.SpanContext{}
	}
	if !extractPart(scc.SpanID[:], &h, 16) {
		return trace.SpanContext{}
	}

	var opts [1]byte
	if !extractPart(opts[:], &h, 2) {
		return trace.SpanContext{}
	}
	if version == 0 && (h != "" || opts[0] > 2) {
		// version 0 not allow extra
		// version 0 not allow other flag
		return trace.SpanContext{}
	}

	// Clear all flags other than the trace-context supported sampling bit.
	scc.TraceFlags = trace.TraceFlags(opts[0]) & trace.FlagsSampled // nolint:gosec // slice size already checked.

	// Ignore the error returned here. Failure to parse tracestate MUST NOT
	// affect the parsing of traceparent according to the W3C tracecontext
	// specification.
	scc.TraceState, _ = trace.ParseTraceState(carrier.Get(tracestateHeader))
	scc.Remote = true

	sc := trace.NewSpanContext(scc)
	if !sc.IsValid() {
		return trace.SpanContext{}
	}

	return sc
}

// upperHex detect hex is upper case Unicode characters.
func upperHex(v string) bool {
	for _, c := range v {
		if c >= 'A' && c <= 'F' {
			return true
		}
	}
	return false
}

func extractPart(dst []byte, h *string, n int) bool {
	part, left, _ := strings.Cut(*h, delimiter)
	*h = left
	// hex.Decode decodes unsupported upper-case characters, so exclude explicitly.
	if len(part) != n || upperHex(part) {
		return false
	}
	if p, err := hex.Decode(dst, []byte(part)); err != nil || p != n/2 {
		return false
	}
	return true
}

// Fields returns the keys who's values are set with Inject.
func (TraceContext) Fields() []string {
	return []string{traceparentHeader, tracestateHeader}
}
