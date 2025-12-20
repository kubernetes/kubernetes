// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package trace // import "go.opentelemetry.io/otel/sdk/trace"

import (
	"context"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/instrumentation"
	"go.opentelemetry.io/otel/semconv/v1.37.0/otelconv"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/embedded"
)

type tracer struct {
	embedded.Tracer

	provider             *TracerProvider
	instrumentationScope instrumentation.Scope

	selfObservabilityEnabled bool
	spanLiveMetric           otelconv.SDKSpanLive
	spanStartedMetric        otelconv.SDKSpanStarted
}

var _ trace.Tracer = &tracer{}

// Start starts a Span and returns it along with a context containing it.
//
// The Span is created with the provided name and as a child of any existing
// span context found in the passed context. The created Span will be
// configured appropriately by any SpanOption passed.
func (tr *tracer) Start(
	ctx context.Context,
	name string,
	options ...trace.SpanStartOption,
) (context.Context, trace.Span) {
	config := trace.NewSpanStartConfig(options...)

	if ctx == nil {
		// Prevent trace.ContextWithSpan from panicking.
		ctx = context.Background()
	}

	// For local spans created by this SDK, track child span count.
	if p := trace.SpanFromContext(ctx); p != nil {
		if sdkSpan, ok := p.(*recordingSpan); ok {
			sdkSpan.addChild()
		}
	}

	s := tr.newSpan(ctx, name, &config)
	newCtx := trace.ContextWithSpan(ctx, s)
	if tr.selfObservabilityEnabled {
		psc := trace.SpanContextFromContext(ctx)
		set := spanStartedSet(psc, s)
		tr.spanStartedMetric.AddSet(newCtx, 1, set)
	}

	if rw, ok := s.(ReadWriteSpan); ok && s.IsRecording() {
		sps := tr.provider.getSpanProcessors()
		for _, sp := range sps {
			// Use original context.
			sp.sp.OnStart(ctx, rw)
		}
	}
	if rtt, ok := s.(runtimeTracer); ok {
		newCtx = rtt.runtimeTrace(newCtx)
	}

	return newCtx, s
}

type runtimeTracer interface {
	// runtimeTrace starts a "runtime/trace".Task for the span and
	// returns a context containing the task.
	runtimeTrace(ctx context.Context) context.Context
}

// newSpan returns a new configured span.
func (tr *tracer) newSpan(ctx context.Context, name string, config *trace.SpanConfig) trace.Span {
	// If told explicitly to make this a new root use a zero value SpanContext
	// as a parent which contains an invalid trace ID and is not remote.
	var psc trace.SpanContext
	if config.NewRoot() {
		ctx = trace.ContextWithSpanContext(ctx, psc)
	} else {
		psc = trace.SpanContextFromContext(ctx)
	}

	// If there is a valid parent trace ID, use it to ensure the continuity of
	// the trace. Always generate a new span ID so other components can rely
	// on a unique span ID, even if the Span is non-recording.
	var tid trace.TraceID
	var sid trace.SpanID
	if !psc.TraceID().IsValid() {
		tid, sid = tr.provider.idGenerator.NewIDs(ctx)
	} else {
		tid = psc.TraceID()
		sid = tr.provider.idGenerator.NewSpanID(ctx, tid)
	}

	samplingResult := tr.provider.sampler.ShouldSample(SamplingParameters{
		ParentContext: ctx,
		TraceID:       tid,
		Name:          name,
		Kind:          config.SpanKind(),
		Attributes:    config.Attributes(),
		Links:         config.Links(),
	})

	scc := trace.SpanContextConfig{
		TraceID:    tid,
		SpanID:     sid,
		TraceState: samplingResult.Tracestate,
	}
	if isSampled(samplingResult) {
		scc.TraceFlags = psc.TraceFlags() | trace.FlagsSampled
	} else {
		scc.TraceFlags = psc.TraceFlags() &^ trace.FlagsSampled
	}
	sc := trace.NewSpanContext(scc)

	if !isRecording(samplingResult) {
		return tr.newNonRecordingSpan(sc)
	}
	return tr.newRecordingSpan(ctx, psc, sc, name, samplingResult, config)
}

// newRecordingSpan returns a new configured recordingSpan.
func (tr *tracer) newRecordingSpan(
	ctx context.Context,
	psc, sc trace.SpanContext,
	name string,
	sr SamplingResult,
	config *trace.SpanConfig,
) *recordingSpan {
	startTime := config.Timestamp()
	if startTime.IsZero() {
		startTime = time.Now()
	}

	s := &recordingSpan{
		// Do not pre-allocate the attributes slice here! Doing so will
		// allocate memory that is likely never going to be used, or if used,
		// will be over-sized. The default Go compiler has been tested to
		// dynamically allocate needed space very well. Benchmarking has shown
		// it to be more performant than what we can predetermine here,
		// especially for the common use case of few to no added
		// attributes.

		parent:      psc,
		spanContext: sc,
		spanKind:    trace.ValidateSpanKind(config.SpanKind()),
		name:        name,
		startTime:   startTime,
		events:      newEvictedQueueEvent(tr.provider.spanLimits.EventCountLimit),
		links:       newEvictedQueueLink(tr.provider.spanLimits.LinkCountLimit),
		tracer:      tr,
	}

	for _, l := range config.Links() {
		s.AddLink(l)
	}

	s.SetAttributes(sr.Attributes...)
	s.SetAttributes(config.Attributes()...)

	if tr.selfObservabilityEnabled {
		// Propagate any existing values from the context with the new span to
		// the measurement context.
		ctx = trace.ContextWithSpan(ctx, s)
		set := spanLiveSet(s.spanContext.IsSampled())
		tr.spanLiveMetric.AddSet(ctx, 1, set)
	}

	return s
}

// newNonRecordingSpan returns a new configured nonRecordingSpan.
func (tr *tracer) newNonRecordingSpan(sc trace.SpanContext) nonRecordingSpan {
	return nonRecordingSpan{tracer: tr, sc: sc}
}

type parentState int

const (
	parentStateNoParent parentState = iota
	parentStateLocalParent
	parentStateRemoteParent
)

type samplingState int

const (
	samplingStateDrop samplingState = iota
	samplingStateRecordOnly
	samplingStateRecordAndSample
)

type spanStartedSetKey struct {
	parent   parentState
	sampling samplingState
}

var spanStartedSetCache = map[spanStartedSetKey]attribute.Set{
	{parentStateNoParent, samplingStateDrop}: attribute.NewSet(
		otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginNone),
		otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultDrop),
	),
	{parentStateLocalParent, samplingStateDrop}: attribute.NewSet(
		otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginLocal),
		otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultDrop),
	),
	{parentStateRemoteParent, samplingStateDrop}: attribute.NewSet(
		otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginRemote),
		otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultDrop),
	),

	{parentStateNoParent, samplingStateRecordOnly}: attribute.NewSet(
		otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginNone),
		otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultRecordOnly),
	),
	{parentStateLocalParent, samplingStateRecordOnly}: attribute.NewSet(
		otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginLocal),
		otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultRecordOnly),
	),
	{parentStateRemoteParent, samplingStateRecordOnly}: attribute.NewSet(
		otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginRemote),
		otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultRecordOnly),
	),

	{parentStateNoParent, samplingStateRecordAndSample}: attribute.NewSet(
		otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginNone),
		otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultRecordAndSample),
	),
	{parentStateLocalParent, samplingStateRecordAndSample}: attribute.NewSet(
		otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginLocal),
		otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultRecordAndSample),
	),
	{parentStateRemoteParent, samplingStateRecordAndSample}: attribute.NewSet(
		otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginRemote),
		otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultRecordAndSample),
	),
}

func spanStartedSet(psc trace.SpanContext, span trace.Span) attribute.Set {
	key := spanStartedSetKey{
		parent:   parentStateNoParent,
		sampling: samplingStateDrop,
	}

	if psc.IsValid() {
		if psc.IsRemote() {
			key.parent = parentStateRemoteParent
		} else {
			key.parent = parentStateLocalParent
		}
	}

	if span.IsRecording() {
		if span.SpanContext().IsSampled() {
			key.sampling = samplingStateRecordAndSample
		} else {
			key.sampling = samplingStateRecordOnly
		}
	}

	return spanStartedSetCache[key]
}

type spanLiveSetKey struct {
	sampled bool
}

var spanLiveSetCache = map[spanLiveSetKey]attribute.Set{
	{true}: attribute.NewSet(
		otelconv.SDKSpanLive{}.AttrSpanSamplingResult(
			otelconv.SpanSamplingResultRecordAndSample,
		),
	),
	{false}: attribute.NewSet(
		otelconv.SDKSpanLive{}.AttrSpanSamplingResult(
			otelconv.SpanSamplingResultRecordOnly,
		),
	),
}

func spanLiveSet(sampled bool) attribute.Set {
	key := spanLiveSetKey{sampled: sampled}
	return spanLiveSetCache[key]
}
