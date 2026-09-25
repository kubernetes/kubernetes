// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package observ // import "go.opentelemetry.io/otel/sdk/trace/internal/observ"

import (
	"context"
	"errors"
	"fmt"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/sdk"
	"go.opentelemetry.io/otel/sdk/internal/x"
	"go.opentelemetry.io/otel/semconv/v1.37.0/otelconv"
	"go.opentelemetry.io/otel/trace"
)

var meterOpts = []metric.MeterOption{
	metric.WithInstrumentationVersion(sdk.Version()),
	metric.WithSchemaURL(SchemaURL),
}

// Tracer is instrumentation for an OTel SDK Tracer.
type Tracer struct {
	enabled bool

	live    metric.Int64UpDownCounter
	started metric.Int64Counter
}

func NewTracer() (Tracer, error) {
	if !x.Observability.Enabled() {
		return Tracer{}, nil
	}
	meter := otel.GetMeterProvider().Meter(ScopeName, meterOpts...)

	var err error
	l, e := otelconv.NewSDKSpanLive(meter)
	if e != nil {
		e = fmt.Errorf("failed to create span live metric: %w", e)
		err = errors.Join(err, e)
	}

	s, e := otelconv.NewSDKSpanStarted(meter)
	if e != nil {
		e = fmt.Errorf("failed to create span started metric: %w", e)
		err = errors.Join(err, e)
	}

	return Tracer{enabled: true, live: l.Inst(), started: s.Inst()}, err
}

func (t Tracer) Enabled() bool { return t.enabled }

func (t Tracer) SpanStarted(ctx context.Context, psc trace.SpanContext, span trace.Span) {
	key := spanStartedKey{
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

	opts := spanStartedOpts[key]
	t.started.Add(ctx, 1, opts...)
}

func (t Tracer) SpanLive(ctx context.Context, span trace.Span) {
	t.spanLive(ctx, 1, span)
}

func (t Tracer) SpanEnded(ctx context.Context, span trace.Span) {
	t.spanLive(ctx, -1, span)
}

func (t Tracer) spanLive(ctx context.Context, value int64, span trace.Span) {
	key := spanLiveKey{sampled: span.SpanContext().IsSampled()}
	opts := spanLiveOpts[key]
	t.live.Add(ctx, value, opts...)
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

type spanStartedKey struct {
	parent   parentState
	sampling samplingState
}

var spanStartedOpts = map[spanStartedKey][]metric.AddOption{
	{
		parentStateNoParent,
		samplingStateDrop,
	}: {
		metric.WithAttributeSet(attribute.NewSet(
			otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginNone),
			otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultDrop),
		)),
	},
	{
		parentStateLocalParent,
		samplingStateDrop,
	}: {
		metric.WithAttributeSet(attribute.NewSet(
			otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginLocal),
			otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultDrop),
		)),
	},
	{
		parentStateRemoteParent,
		samplingStateDrop,
	}: {
		metric.WithAttributeSet(attribute.NewSet(
			otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginRemote),
			otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultDrop),
		)),
	},

	{
		parentStateNoParent,
		samplingStateRecordOnly,
	}: {
		metric.WithAttributeSet(attribute.NewSet(
			otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginNone),
			otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultRecordOnly),
		)),
	},
	{
		parentStateLocalParent,
		samplingStateRecordOnly,
	}: {
		metric.WithAttributeSet(attribute.NewSet(
			otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginLocal),
			otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultRecordOnly),
		)),
	},
	{
		parentStateRemoteParent,
		samplingStateRecordOnly,
	}: {
		metric.WithAttributeSet(attribute.NewSet(
			otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginRemote),
			otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultRecordOnly),
		)),
	},

	{
		parentStateNoParent,
		samplingStateRecordAndSample,
	}: {
		metric.WithAttributeSet(attribute.NewSet(
			otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginNone),
			otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultRecordAndSample),
		)),
	},
	{
		parentStateLocalParent,
		samplingStateRecordAndSample,
	}: {
		metric.WithAttributeSet(attribute.NewSet(
			otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginLocal),
			otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultRecordAndSample),
		)),
	},
	{
		parentStateRemoteParent,
		samplingStateRecordAndSample,
	}: {
		metric.WithAttributeSet(attribute.NewSet(
			otelconv.SDKSpanStarted{}.AttrSpanParentOrigin(otelconv.SpanParentOriginRemote),
			otelconv.SDKSpanStarted{}.AttrSpanSamplingResult(otelconv.SpanSamplingResultRecordAndSample),
		)),
	},
}

type spanLiveKey struct {
	sampled bool
}

var spanLiveOpts = map[spanLiveKey][]metric.AddOption{
	{true}: {
		metric.WithAttributeSet(attribute.NewSet(
			otelconv.SDKSpanLive{}.AttrSpanSamplingResult(
				otelconv.SpanSamplingResultRecordAndSample,
			),
		)),
	},
	{false}: {
		metric.WithAttributeSet(attribute.NewSet(
			otelconv.SDKSpanLive{}.AttrSpanSamplingResult(
				otelconv.SpanSamplingResultRecordOnly,
			),
		)),
	},
}
