// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package sdk

import (
	"context"
	"math"
	"time"

	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	"go.opentelemetry.io/auto/sdk/internal/telemetry"
)

type tracer struct {
	noop.Tracer

	name, schemaURL, version string
}

var _ trace.Tracer = tracer{}

func (t tracer) Start(
	ctx context.Context,
	name string,
	opts ...trace.SpanStartOption,
) (context.Context, trace.Span) {
	var psc, sc trace.SpanContext
	sampled := true
	span := new(span)

	// Ask eBPF for sampling decision and span context info.
	t.start(ctx, span, &psc, &sampled, &sc)

	span.sampled.Store(sampled)
	span.spanContext = sc

	ctx = trace.ContextWithSpan(ctx, span)

	if sampled {
		// Only build traces if sampled.
		cfg := trace.NewSpanStartConfig(opts...)
		span.traces, span.span = t.traces(name, cfg, span.spanContext, psc)
	}

	return ctx, span
}

// Expected to be implemented in eBPF.
//
//go:noinline
func (t *tracer) start(
	ctx context.Context,
	spanPtr *span,
	psc *trace.SpanContext,
	sampled *bool,
	sc *trace.SpanContext,
) {
	start(ctx, spanPtr, psc, sampled, sc)
}

// start is used for testing.
var start = func(context.Context, *span, *trace.SpanContext, *bool, *trace.SpanContext) {}

var intToUint32Bound = min(math.MaxInt, math.MaxUint32)

func (t tracer) traces(
	name string,
	cfg trace.SpanConfig,
	sc, psc trace.SpanContext,
) (*telemetry.Traces, *telemetry.Span) {
	span := &telemetry.Span{
		TraceID:      telemetry.TraceID(sc.TraceID()),
		SpanID:       telemetry.SpanID(sc.SpanID()),
		Flags:        uint32(sc.TraceFlags()),
		TraceState:   sc.TraceState().String(),
		ParentSpanID: telemetry.SpanID(psc.SpanID()),
		Name:         name,
		Kind:         spanKind(cfg.SpanKind()),
	}

	span.Attrs, span.DroppedAttrs = convCappedAttrs(maxSpan.Attrs, cfg.Attributes())

	links := cfg.Links()
	if limit := maxSpan.Links; limit == 0 {
		n := len(links)
		if n > 0 {
			bounded := max(min(n, intToUint32Bound), 0)
			span.DroppedLinks = uint32(bounded) //nolint:gosec  // Bounds checked.
		}
	} else {
		if limit > 0 {
			n := max(len(links)-limit, 0)
			bounded := min(n, intToUint32Bound)
			span.DroppedLinks = uint32(bounded) //nolint:gosec  // Bounds checked.
			links = links[n:]
		}
		span.Links = convLinks(links)
	}

	if t := cfg.Timestamp(); !t.IsZero() {
		span.StartTime = cfg.Timestamp()
	} else {
		span.StartTime = time.Now()
	}

	return &telemetry.Traces{
		ResourceSpans: []*telemetry.ResourceSpans{
			{
				ScopeSpans: []*telemetry.ScopeSpans{
					{
						Scope: &telemetry.Scope{
							Name:    t.name,
							Version: t.version,
						},
						Spans:     []*telemetry.Span{span},
						SchemaURL: t.schemaURL,
					},
				},
			},
		},
	}, span
}

func spanKind(kind trace.SpanKind) telemetry.SpanKind {
	switch kind {
	case trace.SpanKindInternal:
		return telemetry.SpanKindInternal
	case trace.SpanKindServer:
		return telemetry.SpanKindServer
	case trace.SpanKindClient:
		return telemetry.SpanKindClient
	case trace.SpanKindProducer:
		return telemetry.SpanKindProducer
	case trace.SpanKindConsumer:
		return telemetry.SpanKindConsumer
	}
	return telemetry.SpanKind(0) // undefined.
}
