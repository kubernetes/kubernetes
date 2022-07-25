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

package propagation // import "go.opentelemetry.io/otel/propagation"

import (
	"context"
	"encoding/hex"
	"fmt"
	"regexp"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

const (
	supportedVersion  = 0
	maxVersion        = 254
	traceparentHeader = "traceparent"
	tracestateHeader  = "tracestate"
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

var _ TextMapPropagator = TraceContext{}
var traceCtxRegExp = regexp.MustCompile("^(?P<version>[0-9a-f]{2})-(?P<traceID>[a-f0-9]{32})-(?P<spanID>[a-f0-9]{16})-(?P<traceFlags>[a-f0-9]{2})(?:-.*)?$")

// Inject set tracecontext from the Context into the carrier.
func (tc TraceContext) Inject(ctx context.Context, carrier TextMapCarrier) {
	sc := trace.SpanContextFromContext(ctx)
	if !sc.IsValid() {
		return
	}

	carrier.Set(tracestateHeader, sc.TraceState().String())

	// Clear all flags other than the trace-context supported sampling bit.
	flags := sc.TraceFlags() & trace.FlagsSampled

	h := fmt.Sprintf("%.2x-%s-%s-%s",
		supportedVersion,
		sc.TraceID(),
		sc.SpanID(),
		flags)
	carrier.Set(traceparentHeader, h)
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

func (tc TraceContext) extract(carrier TextMapCarrier) trace.SpanContext {
	h := carrier.Get(traceparentHeader)
	if h == "" {
		return trace.SpanContext{}
	}

	matches := traceCtxRegExp.FindStringSubmatch(h)

	if len(matches) == 0 {
		return trace.SpanContext{}
	}

	if len(matches) < 5 { // four subgroups plus the overall match
		return trace.SpanContext{}
	}

	if len(matches[1]) != 2 {
		return trace.SpanContext{}
	}
	ver, err := hex.DecodeString(matches[1])
	if err != nil {
		return trace.SpanContext{}
	}
	version := int(ver[0])
	if version > maxVersion {
		return trace.SpanContext{}
	}

	if version == 0 && len(matches) != 5 { // four subgroups plus the overall match
		return trace.SpanContext{}
	}

	if len(matches[2]) != 32 {
		return trace.SpanContext{}
	}

	var scc trace.SpanContextConfig

	scc.TraceID, err = trace.TraceIDFromHex(matches[2][:32])
	if err != nil {
		return trace.SpanContext{}
	}

	if len(matches[3]) != 16 {
		return trace.SpanContext{}
	}
	scc.SpanID, err = trace.SpanIDFromHex(matches[3])
	if err != nil {
		return trace.SpanContext{}
	}

	if len(matches[4]) != 2 {
		return trace.SpanContext{}
	}
	opts, err := hex.DecodeString(matches[4])
	if err != nil || len(opts) < 1 || (version == 0 && opts[0] > 2) {
		return trace.SpanContext{}
	}
	// Clear all flags other than the trace-context supported sampling bit.
	scc.TraceFlags = trace.TraceFlags(opts[0]) & trace.FlagsSampled

	scc.TraceState = parseTraceState(carrier.Get(tracestateHeader))
	scc.Remote = true

	sc := trace.NewSpanContext(scc)
	if !sc.IsValid() {
		return trace.SpanContext{}
	}

	return sc
}

// Fields returns the keys who's values are set with Inject.
func (tc TraceContext) Fields() []string {
	return []string{traceparentHeader, tracestateHeader}
}

func parseTraceState(in string) trace.TraceState {
	if in == "" {
		return trace.TraceState{}
	}

	kvs := []attribute.KeyValue{}
	for _, entry := range strings.Split(in, ",") {
		parts := strings.SplitN(entry, "=", 2)
		if len(parts) != 2 {
			// Parse failure, abort!
			return trace.TraceState{}
		}
		kvs = append(kvs, attribute.String(parts[0], parts[1]))
	}

	// Ignoring error here as "failure to parse tracestate MUST NOT
	// affect the parsing of traceparent."
	// https://www.w3.org/TR/trace-context/#tracestate-header
	ts, _ := trace.TraceStateFromKeyValues(kvs...)
	return ts
}
