// Code generated from semantic convention specification. DO NOT EDIT.

// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package httpconv provides types and functionality for OpenTelemetry semantic
// conventions in the "otel" namespace.
package otelconv

import (
	"context"
	"sync"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/noop"
)

var (
	addOptPool = &sync.Pool{New: func() any { return &[]metric.AddOption{} }}
	recOptPool = &sync.Pool{New: func() any { return &[]metric.RecordOption{} }}
)

// ErrorTypeAttr is an attribute conforming to the error.type semantic
// conventions. It represents the describes a class of error the operation ended
// with.
type ErrorTypeAttr string

var (
	// ErrorTypeOther is a fallback error value to be used when the instrumentation
	// doesn't define a custom value.
	ErrorTypeOther ErrorTypeAttr = "_OTHER"
)

// ComponentTypeAttr is an attribute conforming to the otel.component.type
// semantic conventions. It represents a name identifying the type of the
// OpenTelemetry component.
type ComponentTypeAttr string

var (
	// ComponentTypeBatchingSpanProcessor is the builtin SDK batching span
	// processor.
	ComponentTypeBatchingSpanProcessor ComponentTypeAttr = "batching_span_processor"
	// ComponentTypeSimpleSpanProcessor is the builtin SDK simple span processor.
	ComponentTypeSimpleSpanProcessor ComponentTypeAttr = "simple_span_processor"
	// ComponentTypeBatchingLogProcessor is the builtin SDK batching log record
	// processor.
	ComponentTypeBatchingLogProcessor ComponentTypeAttr = "batching_log_processor"
	// ComponentTypeSimpleLogProcessor is the builtin SDK simple log record
	// processor.
	ComponentTypeSimpleLogProcessor ComponentTypeAttr = "simple_log_processor"
	// ComponentTypeOtlpGRPCSpanExporter is the OTLP span exporter over gRPC with
	// protobuf serialization.
	ComponentTypeOtlpGRPCSpanExporter ComponentTypeAttr = "otlp_grpc_span_exporter"
	// ComponentTypeOtlpHTTPSpanExporter is the OTLP span exporter over HTTP with
	// protobuf serialization.
	ComponentTypeOtlpHTTPSpanExporter ComponentTypeAttr = "otlp_http_span_exporter"
	// ComponentTypeOtlpHTTPJSONSpanExporter is the OTLP span exporter over HTTP
	// with JSON serialization.
	ComponentTypeOtlpHTTPJSONSpanExporter ComponentTypeAttr = "otlp_http_json_span_exporter"
	// ComponentTypeZipkinHTTPSpanExporter is the zipkin span exporter over HTTP.
	ComponentTypeZipkinHTTPSpanExporter ComponentTypeAttr = "zipkin_http_span_exporter"
	// ComponentTypeOtlpGRPCLogExporter is the OTLP log record exporter over gRPC
	// with protobuf serialization.
	ComponentTypeOtlpGRPCLogExporter ComponentTypeAttr = "otlp_grpc_log_exporter"
	// ComponentTypeOtlpHTTPLogExporter is the OTLP log record exporter over HTTP
	// with protobuf serialization.
	ComponentTypeOtlpHTTPLogExporter ComponentTypeAttr = "otlp_http_log_exporter"
	// ComponentTypeOtlpHTTPJSONLogExporter is the OTLP log record exporter over
	// HTTP with JSON serialization.
	ComponentTypeOtlpHTTPJSONLogExporter ComponentTypeAttr = "otlp_http_json_log_exporter"
	// ComponentTypePeriodicMetricReader is the builtin SDK periodically exporting
	// metric reader.
	ComponentTypePeriodicMetricReader ComponentTypeAttr = "periodic_metric_reader"
	// ComponentTypeOtlpGRPCMetricExporter is the OTLP metric exporter over gRPC
	// with protobuf serialization.
	ComponentTypeOtlpGRPCMetricExporter ComponentTypeAttr = "otlp_grpc_metric_exporter"
	// ComponentTypeOtlpHTTPMetricExporter is the OTLP metric exporter over HTTP
	// with protobuf serialization.
	ComponentTypeOtlpHTTPMetricExporter ComponentTypeAttr = "otlp_http_metric_exporter"
	// ComponentTypeOtlpHTTPJSONMetricExporter is the OTLP metric exporter over HTTP
	// with JSON serialization.
	ComponentTypeOtlpHTTPJSONMetricExporter ComponentTypeAttr = "otlp_http_json_metric_exporter"
	// ComponentTypePrometheusHTTPTextMetricExporter is the prometheus metric
	// exporter over HTTP with the default text-based format.
	ComponentTypePrometheusHTTPTextMetricExporter ComponentTypeAttr = "prometheus_http_text_metric_exporter"
)

// SpanParentOriginAttr is an attribute conforming to the otel.span.parent.origin
// semantic conventions. It represents the determines whether the span has a
// parent span, and if so, [whether it is a remote parent].
//
// [whether it is a remote parent]: https://opentelemetry.io/docs/specs/otel/trace/api/#isremote
type SpanParentOriginAttr string

var (
	// SpanParentOriginNone is the span does not have a parent, it is a root span.
	SpanParentOriginNone SpanParentOriginAttr = "none"
	// SpanParentOriginLocal is the span has a parent and the parent's span context
	// [isRemote()] is false.
	//
	// [isRemote()]: https://opentelemetry.io/docs/specs/otel/trace/api/#isremote
	SpanParentOriginLocal SpanParentOriginAttr = "local"
	// SpanParentOriginRemote is the span has a parent and the parent's span context
	// [isRemote()] is true.
	//
	// [isRemote()]: https://opentelemetry.io/docs/specs/otel/trace/api/#isremote
	SpanParentOriginRemote SpanParentOriginAttr = "remote"
)

// SpanSamplingResultAttr is an attribute conforming to the
// otel.span.sampling_result semantic conventions. It represents the result value
// of the sampler for this span.
type SpanSamplingResultAttr string

var (
	// SpanSamplingResultDrop is the span is not sampled and not recording.
	SpanSamplingResultDrop SpanSamplingResultAttr = "DROP"
	// SpanSamplingResultRecordOnly is the span is not sampled, but recording.
	SpanSamplingResultRecordOnly SpanSamplingResultAttr = "RECORD_ONLY"
	// SpanSamplingResultRecordAndSample is the span is sampled and recording.
	SpanSamplingResultRecordAndSample SpanSamplingResultAttr = "RECORD_AND_SAMPLE"
)

// RPCGRPCStatusCodeAttr is an attribute conforming to the rpc.grpc.status_code
// semantic conventions. It represents the gRPC status code of the last gRPC
// requests performed in scope of this export call.
type RPCGRPCStatusCodeAttr int64

var (
	// RPCGRPCStatusCodeOk is the OK.
	RPCGRPCStatusCodeOk RPCGRPCStatusCodeAttr = 0
	// RPCGRPCStatusCodeCancelled is the CANCELLED.
	RPCGRPCStatusCodeCancelled RPCGRPCStatusCodeAttr = 1
	// RPCGRPCStatusCodeUnknown is the UNKNOWN.
	RPCGRPCStatusCodeUnknown RPCGRPCStatusCodeAttr = 2
	// RPCGRPCStatusCodeInvalidArgument is the INVALID_ARGUMENT.
	RPCGRPCStatusCodeInvalidArgument RPCGRPCStatusCodeAttr = 3
	// RPCGRPCStatusCodeDeadlineExceeded is the DEADLINE_EXCEEDED.
	RPCGRPCStatusCodeDeadlineExceeded RPCGRPCStatusCodeAttr = 4
	// RPCGRPCStatusCodeNotFound is the NOT_FOUND.
	RPCGRPCStatusCodeNotFound RPCGRPCStatusCodeAttr = 5
	// RPCGRPCStatusCodeAlreadyExists is the ALREADY_EXISTS.
	RPCGRPCStatusCodeAlreadyExists RPCGRPCStatusCodeAttr = 6
	// RPCGRPCStatusCodePermissionDenied is the PERMISSION_DENIED.
	RPCGRPCStatusCodePermissionDenied RPCGRPCStatusCodeAttr = 7
	// RPCGRPCStatusCodeResourceExhausted is the RESOURCE_EXHAUSTED.
	RPCGRPCStatusCodeResourceExhausted RPCGRPCStatusCodeAttr = 8
	// RPCGRPCStatusCodeFailedPrecondition is the FAILED_PRECONDITION.
	RPCGRPCStatusCodeFailedPrecondition RPCGRPCStatusCodeAttr = 9
	// RPCGRPCStatusCodeAborted is the ABORTED.
	RPCGRPCStatusCodeAborted RPCGRPCStatusCodeAttr = 10
	// RPCGRPCStatusCodeOutOfRange is the OUT_OF_RANGE.
	RPCGRPCStatusCodeOutOfRange RPCGRPCStatusCodeAttr = 11
	// RPCGRPCStatusCodeUnimplemented is the UNIMPLEMENTED.
	RPCGRPCStatusCodeUnimplemented RPCGRPCStatusCodeAttr = 12
	// RPCGRPCStatusCodeInternal is the INTERNAL.
	RPCGRPCStatusCodeInternal RPCGRPCStatusCodeAttr = 13
	// RPCGRPCStatusCodeUnavailable is the UNAVAILABLE.
	RPCGRPCStatusCodeUnavailable RPCGRPCStatusCodeAttr = 14
	// RPCGRPCStatusCodeDataLoss is the DATA_LOSS.
	RPCGRPCStatusCodeDataLoss RPCGRPCStatusCodeAttr = 15
	// RPCGRPCStatusCodeUnauthenticated is the UNAUTHENTICATED.
	RPCGRPCStatusCodeUnauthenticated RPCGRPCStatusCodeAttr = 16
)

// SDKExporterLogExported is an instrument used to record metric values
// conforming to the "otel.sdk.exporter.log.exported" semantic conventions. It
// represents the number of log records for which the export has finished, either
// successful or failed.
type SDKExporterLogExported struct {
	metric.Int64Counter
}

// NewSDKExporterLogExported returns a new SDKExporterLogExported instrument.
func NewSDKExporterLogExported(
	m metric.Meter,
	opt ...metric.Int64CounterOption,
) (SDKExporterLogExported, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKExporterLogExported{noop.Int64Counter{}}, nil
	}

	i, err := m.Int64Counter(
		"otel.sdk.exporter.log.exported",
		append([]metric.Int64CounterOption{
			metric.WithDescription("The number of log records for which the export has finished, either successful or failed."),
			metric.WithUnit("{log_record}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKExporterLogExported{noop.Int64Counter{}}, err
	}
	return SDKExporterLogExported{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKExporterLogExported) Inst() metric.Int64Counter {
	return m.Int64Counter
}

// Name returns the semantic convention name of the instrument.
func (SDKExporterLogExported) Name() string {
	return "otel.sdk.exporter.log.exported"
}

// Unit returns the semantic convention unit of the instrument
func (SDKExporterLogExported) Unit() string {
	return "{log_record}"
}

// Description returns the semantic convention description of the instrument
func (SDKExporterLogExported) Description() string {
	return "The number of log records for which the export has finished, either successful or failed."
}

// Add adds incr to the existing count for attrs.
//
// All additional attrs passed are included in the recorded value.
//
// For successful exports, `error.type` MUST NOT be set. For failed exports,
// `error.type` MUST contain the failure cause.
// For exporters with partial success semantics (e.g. OTLP with
// `rejected_log_records`), rejected log records MUST count as failed and only
// non-rejected log records count as success.
// If no rejection reason is available, `rejected` SHOULD be used as value for
// `error.type`.
func (m SDKExporterLogExported) Add(
	ctx context.Context,
	incr int64,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64Counter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			attrs...,
		),
	)

	m.Int64Counter.Add(ctx, incr, *o...)
}

// AddSet adds incr to the existing count for set.
//
// For successful exports, `error.type` MUST NOT be set. For failed exports,
// `error.type` MUST contain the failure cause.
// For exporters with partial success semantics (e.g. OTLP with
// `rejected_log_records`), rejected log records MUST count as failed and only
// non-rejected log records count as success.
// If no rejection reason is available, `rejected` SHOULD be used as value for
// `error.type`.
func (m SDKExporterLogExported) AddSet(ctx context.Context, incr int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64Counter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64Counter.Add(ctx, incr, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents the describes a class of error the operation ended
// with.
func (SDKExporterLogExported) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrComponentName returns an optional attribute for the "otel.component.name"
// semantic convention. It represents a name uniquely identifying the instance of
// the OpenTelemetry component within its containing SDK instance.
func (SDKExporterLogExported) AttrComponentName(val string) attribute.KeyValue {
	return attribute.String("otel.component.name", val)
}

// AttrComponentType returns an optional attribute for the "otel.component.type"
// semantic convention. It represents a name identifying the type of the
// OpenTelemetry component.
func (SDKExporterLogExported) AttrComponentType(val ComponentTypeAttr) attribute.KeyValue {
	return attribute.String("otel.component.type", string(val))
}

// AttrServerAddress returns an optional attribute for the "server.address"
// semantic convention. It represents the server domain name if available without
// reverse DNS lookup; otherwise, IP address or Unix domain socket name.
func (SDKExporterLogExported) AttrServerAddress(val string) attribute.KeyValue {
	return attribute.String("server.address", val)
}

// AttrServerPort returns an optional attribute for the "server.port" semantic
// convention. It represents the server port number.
func (SDKExporterLogExported) AttrServerPort(val int) attribute.KeyValue {
	return attribute.Int("server.port", val)
}

// SDKExporterLogInflight is an instrument used to record metric values
// conforming to the "otel.sdk.exporter.log.inflight" semantic conventions. It
// represents the number of log records which were passed to the exporter, but
// that have not been exported yet (neither successful, nor failed).
type SDKExporterLogInflight struct {
	metric.Int64UpDownCounter
}

// NewSDKExporterLogInflight returns a new SDKExporterLogInflight instrument.
func NewSDKExporterLogInflight(
	m metric.Meter,
	opt ...metric.Int64UpDownCounterOption,
) (SDKExporterLogInflight, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKExporterLogInflight{noop.Int64UpDownCounter{}}, nil
	}

	i, err := m.Int64UpDownCounter(
		"otel.sdk.exporter.log.inflight",
		append([]metric.Int64UpDownCounterOption{
			metric.WithDescription("The number of log records which were passed to the exporter, but that have not been exported yet (neither successful, nor failed)."),
			metric.WithUnit("{log_record}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKExporterLogInflight{noop.Int64UpDownCounter{}}, err
	}
	return SDKExporterLogInflight{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKExporterLogInflight) Inst() metric.Int64UpDownCounter {
	return m.Int64UpDownCounter
}

// Name returns the semantic convention name of the instrument.
func (SDKExporterLogInflight) Name() string {
	return "otel.sdk.exporter.log.inflight"
}

// Unit returns the semantic convention unit of the instrument
func (SDKExporterLogInflight) Unit() string {
	return "{log_record}"
}

// Description returns the semantic convention description of the instrument
func (SDKExporterLogInflight) Description() string {
	return "The number of log records which were passed to the exporter, but that have not been exported yet (neither successful, nor failed)."
}

// Add adds incr to the existing count for attrs.
//
// All additional attrs passed are included in the recorded value.
//
// For successful exports, `error.type` MUST NOT be set. For failed exports,
// `error.type` MUST contain the failure cause.
func (m SDKExporterLogInflight) Add(
	ctx context.Context,
	incr int64,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64UpDownCounter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			attrs...,
		),
	)

	m.Int64UpDownCounter.Add(ctx, incr, *o...)
}

// AddSet adds incr to the existing count for set.
//
// For successful exports, `error.type` MUST NOT be set. For failed exports,
// `error.type` MUST contain the failure cause.
func (m SDKExporterLogInflight) AddSet(ctx context.Context, incr int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64UpDownCounter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64UpDownCounter.Add(ctx, incr, *o...)
}

// AttrComponentName returns an optional attribute for the "otel.component.name"
// semantic convention. It represents a name uniquely identifying the instance of
// the OpenTelemetry component within its containing SDK instance.
func (SDKExporterLogInflight) AttrComponentName(val string) attribute.KeyValue {
	return attribute.String("otel.component.name", val)
}

// AttrComponentType returns an optional attribute for the "otel.component.type"
// semantic convention. It represents a name identifying the type of the
// OpenTelemetry component.
func (SDKExporterLogInflight) AttrComponentType(val ComponentTypeAttr) attribute.KeyValue {
	return attribute.String("otel.component.type", string(val))
}

// AttrServerAddress returns an optional attribute for the "server.address"
// semantic convention. It represents the server domain name if available without
// reverse DNS lookup; otherwise, IP address or Unix domain socket name.
func (SDKExporterLogInflight) AttrServerAddress(val string) attribute.KeyValue {
	return attribute.String("server.address", val)
}

// AttrServerPort returns an optional attribute for the "server.port" semantic
// convention. It represents the server port number.
func (SDKExporterLogInflight) AttrServerPort(val int) attribute.KeyValue {
	return attribute.Int("server.port", val)
}

// SDKExporterMetricDataPointExported is an instrument used to record metric
// values conforming to the "otel.sdk.exporter.metric_data_point.exported"
// semantic conventions. It represents the number of metric data points for which
// the export has finished, either successful or failed.
type SDKExporterMetricDataPointExported struct {
	metric.Int64Counter
}

// NewSDKExporterMetricDataPointExported returns a new
// SDKExporterMetricDataPointExported instrument.
func NewSDKExporterMetricDataPointExported(
	m metric.Meter,
	opt ...metric.Int64CounterOption,
) (SDKExporterMetricDataPointExported, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKExporterMetricDataPointExported{noop.Int64Counter{}}, nil
	}

	i, err := m.Int64Counter(
		"otel.sdk.exporter.metric_data_point.exported",
		append([]metric.Int64CounterOption{
			metric.WithDescription("The number of metric data points for which the export has finished, either successful or failed."),
			metric.WithUnit("{data_point}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKExporterMetricDataPointExported{noop.Int64Counter{}}, err
	}
	return SDKExporterMetricDataPointExported{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKExporterMetricDataPointExported) Inst() metric.Int64Counter {
	return m.Int64Counter
}

// Name returns the semantic convention name of the instrument.
func (SDKExporterMetricDataPointExported) Name() string {
	return "otel.sdk.exporter.metric_data_point.exported"
}

// Unit returns the semantic convention unit of the instrument
func (SDKExporterMetricDataPointExported) Unit() string {
	return "{data_point}"
}

// Description returns the semantic convention description of the instrument
func (SDKExporterMetricDataPointExported) Description() string {
	return "The number of metric data points for which the export has finished, either successful or failed."
}

// Add adds incr to the existing count for attrs.
//
// All additional attrs passed are included in the recorded value.
//
// For successful exports, `error.type` MUST NOT be set. For failed exports,
// `error.type` MUST contain the failure cause.
// For exporters with partial success semantics (e.g. OTLP with
// `rejected_data_points`), rejected data points MUST count as failed and only
// non-rejected data points count as success.
// If no rejection reason is available, `rejected` SHOULD be used as value for
// `error.type`.
func (m SDKExporterMetricDataPointExported) Add(
	ctx context.Context,
	incr int64,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64Counter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			attrs...,
		),
	)

	m.Int64Counter.Add(ctx, incr, *o...)
}

// AddSet adds incr to the existing count for set.
//
// For successful exports, `error.type` MUST NOT be set. For failed exports,
// `error.type` MUST contain the failure cause.
// For exporters with partial success semantics (e.g. OTLP with
// `rejected_data_points`), rejected data points MUST count as failed and only
// non-rejected data points count as success.
// If no rejection reason is available, `rejected` SHOULD be used as value for
// `error.type`.
func (m SDKExporterMetricDataPointExported) AddSet(ctx context.Context, incr int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64Counter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64Counter.Add(ctx, incr, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents the describes a class of error the operation ended
// with.
func (SDKExporterMetricDataPointExported) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrComponentName returns an optional attribute for the "otel.component.name"
// semantic convention. It represents a name uniquely identifying the instance of
// the OpenTelemetry component within its containing SDK instance.
func (SDKExporterMetricDataPointExported) AttrComponentName(val string) attribute.KeyValue {
	return attribute.String("otel.component.name", val)
}

// AttrComponentType returns an optional attribute for the "otel.component.type"
// semantic convention. It represents a name identifying the type of the
// OpenTelemetry component.
func (SDKExporterMetricDataPointExported) AttrComponentType(val ComponentTypeAttr) attribute.KeyValue {
	return attribute.String("otel.component.type", string(val))
}

// AttrServerAddress returns an optional attribute for the "server.address"
// semantic convention. It represents the server domain name if available without
// reverse DNS lookup; otherwise, IP address or Unix domain socket name.
func (SDKExporterMetricDataPointExported) AttrServerAddress(val string) attribute.KeyValue {
	return attribute.String("server.address", val)
}

// AttrServerPort returns an optional attribute for the "server.port" semantic
// convention. It represents the server port number.
func (SDKExporterMetricDataPointExported) AttrServerPort(val int) attribute.KeyValue {
	return attribute.Int("server.port", val)
}

// SDKExporterMetricDataPointInflight is an instrument used to record metric
// values conforming to the "otel.sdk.exporter.metric_data_point.inflight"
// semantic conventions. It represents the number of metric data points which
// were passed to the exporter, but that have not been exported yet (neither
// successful, nor failed).
type SDKExporterMetricDataPointInflight struct {
	metric.Int64UpDownCounter
}

// NewSDKExporterMetricDataPointInflight returns a new
// SDKExporterMetricDataPointInflight instrument.
func NewSDKExporterMetricDataPointInflight(
	m metric.Meter,
	opt ...metric.Int64UpDownCounterOption,
) (SDKExporterMetricDataPointInflight, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKExporterMetricDataPointInflight{noop.Int64UpDownCounter{}}, nil
	}

	i, err := m.Int64UpDownCounter(
		"otel.sdk.exporter.metric_data_point.inflight",
		append([]metric.Int64UpDownCounterOption{
			metric.WithDescription("The number of metric data points which were passed to the exporter, but that have not been exported yet (neither successful, nor failed)."),
			metric.WithUnit("{data_point}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKExporterMetricDataPointInflight{noop.Int64UpDownCounter{}}, err
	}
	return SDKExporterMetricDataPointInflight{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKExporterMetricDataPointInflight) Inst() metric.Int64UpDownCounter {
	return m.Int64UpDownCounter
}

// Name returns the semantic convention name of the instrument.
func (SDKExporterMetricDataPointInflight) Name() string {
	return "otel.sdk.exporter.metric_data_point.inflight"
}

// Unit returns the semantic convention unit of the instrument
func (SDKExporterMetricDataPointInflight) Unit() string {
	return "{data_point}"
}

// Description returns the semantic convention description of the instrument
func (SDKExporterMetricDataPointInflight) Description() string {
	return "The number of metric data points which were passed to the exporter, but that have not been exported yet (neither successful, nor failed)."
}

// Add adds incr to the existing count for attrs.
//
// All additional attrs passed are included in the recorded value.
//
// For successful exports, `error.type` MUST NOT be set. For failed exports,
// `error.type` MUST contain the failure cause.
func (m SDKExporterMetricDataPointInflight) Add(
	ctx context.Context,
	incr int64,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64UpDownCounter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			attrs...,
		),
	)

	m.Int64UpDownCounter.Add(ctx, incr, *o...)
}

// AddSet adds incr to the existing count for set.
//
// For successful exports, `error.type` MUST NOT be set. For failed exports,
// `error.type` MUST contain the failure cause.
func (m SDKExporterMetricDataPointInflight) AddSet(ctx context.Context, incr int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64UpDownCounter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64UpDownCounter.Add(ctx, incr, *o...)
}

// AttrComponentName returns an optional attribute for the "otel.component.name"
// semantic convention. It represents a name uniquely identifying the instance of
// the OpenTelemetry component within its containing SDK instance.
func (SDKExporterMetricDataPointInflight) AttrComponentName(val string) attribute.KeyValue {
	return attribute.String("otel.component.name", val)
}

// AttrComponentType returns an optional attribute for the "otel.component.type"
// semantic convention. It represents a name identifying the type of the
// OpenTelemetry component.
func (SDKExporterMetricDataPointInflight) AttrComponentType(val ComponentTypeAttr) attribute.KeyValue {
	return attribute.String("otel.component.type", string(val))
}

// AttrServerAddress returns an optional attribute for the "server.address"
// semantic convention. It represents the server domain name if available without
// reverse DNS lookup; otherwise, IP address or Unix domain socket name.
func (SDKExporterMetricDataPointInflight) AttrServerAddress(val string) attribute.KeyValue {
	return attribute.String("server.address", val)
}

// AttrServerPort returns an optional attribute for the "server.port" semantic
// convention. It represents the server port number.
func (SDKExporterMetricDataPointInflight) AttrServerPort(val int) attribute.KeyValue {
	return attribute.Int("server.port", val)
}

// SDKExporterOperationDuration is an instrument used to record metric values
// conforming to the "otel.sdk.exporter.operation.duration" semantic conventions.
// It represents the duration of exporting a batch of telemetry records.
type SDKExporterOperationDuration struct {
	metric.Float64Histogram
}

// NewSDKExporterOperationDuration returns a new SDKExporterOperationDuration
// instrument.
func NewSDKExporterOperationDuration(
	m metric.Meter,
	opt ...metric.Float64HistogramOption,
) (SDKExporterOperationDuration, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKExporterOperationDuration{noop.Float64Histogram{}}, nil
	}

	i, err := m.Float64Histogram(
		"otel.sdk.exporter.operation.duration",
		append([]metric.Float64HistogramOption{
			metric.WithDescription("The duration of exporting a batch of telemetry records."),
			metric.WithUnit("s"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKExporterOperationDuration{noop.Float64Histogram{}}, err
	}
	return SDKExporterOperationDuration{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKExporterOperationDuration) Inst() metric.Float64Histogram {
	return m.Float64Histogram
}

// Name returns the semantic convention name of the instrument.
func (SDKExporterOperationDuration) Name() string {
	return "otel.sdk.exporter.operation.duration"
}

// Unit returns the semantic convention unit of the instrument
func (SDKExporterOperationDuration) Unit() string {
	return "s"
}

// Description returns the semantic convention description of the instrument
func (SDKExporterOperationDuration) Description() string {
	return "The duration of exporting a batch of telemetry records."
}

// Record records val to the current distribution for attrs.
//
// All additional attrs passed are included in the recorded value.
//
// This metric defines successful operations using the full success definitions
// for [http]
// and [grpc]. Anything else is defined as an unsuccessful operation. For
// successful
// operations, `error.type` MUST NOT be set. For unsuccessful export operations,
// `error.type` MUST contain a relevant failure cause.
//
// [http]: https://github.com/open-telemetry/opentelemetry-proto/blob/v1.5.0/docs/specification.md#full-success-1
// [grpc]: https://github.com/open-telemetry/opentelemetry-proto/blob/v1.5.0/docs/specification.md#full-success
func (m SDKExporterOperationDuration) Record(
	ctx context.Context,
	val float64,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Float64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			attrs...,
		),
	)

	m.Float64Histogram.Record(ctx, val, *o...)
}

// RecordSet records val to the current distribution for set.
//
// This metric defines successful operations using the full success definitions
// for [http]
// and [grpc]. Anything else is defined as an unsuccessful operation. For
// successful
// operations, `error.type` MUST NOT be set. For unsuccessful export operations,
// `error.type` MUST contain a relevant failure cause.
//
// [http]: https://github.com/open-telemetry/opentelemetry-proto/blob/v1.5.0/docs/specification.md#full-success-1
// [grpc]: https://github.com/open-telemetry/opentelemetry-proto/blob/v1.5.0/docs/specification.md#full-success
func (m SDKExporterOperationDuration) RecordSet(ctx context.Context, val float64, set attribute.Set) {
	if set.Len() == 0 {
		m.Float64Histogram.Record(ctx, val)
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Float64Histogram.Record(ctx, val, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents the describes a class of error the operation ended
// with.
func (SDKExporterOperationDuration) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrHTTPResponseStatusCode returns an optional attribute for the
// "http.response.status_code" semantic convention. It represents the HTTP status
// code of the last HTTP request performed in scope of this export call.
func (SDKExporterOperationDuration) AttrHTTPResponseStatusCode(val int) attribute.KeyValue {
	return attribute.Int("http.response.status_code", val)
}

// AttrComponentName returns an optional attribute for the "otel.component.name"
// semantic convention. It represents a name uniquely identifying the instance of
// the OpenTelemetry component within its containing SDK instance.
func (SDKExporterOperationDuration) AttrComponentName(val string) attribute.KeyValue {
	return attribute.String("otel.component.name", val)
}

// AttrComponentType returns an optional attribute for the "otel.component.type"
// semantic convention. It represents a name identifying the type of the
// OpenTelemetry component.
func (SDKExporterOperationDuration) AttrComponentType(val ComponentTypeAttr) attribute.KeyValue {
	return attribute.String("otel.component.type", string(val))
}

// AttrRPCGRPCStatusCode returns an optional attribute for the
// "rpc.grpc.status_code" semantic convention. It represents the gRPC status code
// of the last gRPC requests performed in scope of this export call.
func (SDKExporterOperationDuration) AttrRPCGRPCStatusCode(val RPCGRPCStatusCodeAttr) attribute.KeyValue {
	return attribute.Int64("rpc.grpc.status_code", int64(val))
}

// AttrServerAddress returns an optional attribute for the "server.address"
// semantic convention. It represents the server domain name if available without
// reverse DNS lookup; otherwise, IP address or Unix domain socket name.
func (SDKExporterOperationDuration) AttrServerAddress(val string) attribute.KeyValue {
	return attribute.String("server.address", val)
}

// AttrServerPort returns an optional attribute for the "server.port" semantic
// convention. It represents the server port number.
func (SDKExporterOperationDuration) AttrServerPort(val int) attribute.KeyValue {
	return attribute.Int("server.port", val)
}

// SDKExporterSpanExported is an instrument used to record metric values
// conforming to the "otel.sdk.exporter.span.exported" semantic conventions. It
// represents the number of spans for which the export has finished, either
// successful or failed.
type SDKExporterSpanExported struct {
	metric.Int64Counter
}

// NewSDKExporterSpanExported returns a new SDKExporterSpanExported instrument.
func NewSDKExporterSpanExported(
	m metric.Meter,
	opt ...metric.Int64CounterOption,
) (SDKExporterSpanExported, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKExporterSpanExported{noop.Int64Counter{}}, nil
	}

	i, err := m.Int64Counter(
		"otel.sdk.exporter.span.exported",
		append([]metric.Int64CounterOption{
			metric.WithDescription("The number of spans for which the export has finished, either successful or failed."),
			metric.WithUnit("{span}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKExporterSpanExported{noop.Int64Counter{}}, err
	}
	return SDKExporterSpanExported{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKExporterSpanExported) Inst() metric.Int64Counter {
	return m.Int64Counter
}

// Name returns the semantic convention name of the instrument.
func (SDKExporterSpanExported) Name() string {
	return "otel.sdk.exporter.span.exported"
}

// Unit returns the semantic convention unit of the instrument
func (SDKExporterSpanExported) Unit() string {
	return "{span}"
}

// Description returns the semantic convention description of the instrument
func (SDKExporterSpanExported) Description() string {
	return "The number of spans for which the export has finished, either successful or failed."
}

// Add adds incr to the existing count for attrs.
//
// All additional attrs passed are included in the recorded value.
//
// For successful exports, `error.type` MUST NOT be set. For failed exports,
// `error.type` MUST contain the failure cause.
// For exporters with partial success semantics (e.g. OTLP with `rejected_spans`
// ), rejected spans MUST count as failed and only non-rejected spans count as
// success.
// If no rejection reason is available, `rejected` SHOULD be used as value for
// `error.type`.
func (m SDKExporterSpanExported) Add(
	ctx context.Context,
	incr int64,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64Counter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			attrs...,
		),
	)

	m.Int64Counter.Add(ctx, incr, *o...)
}

// AddSet adds incr to the existing count for set.
//
// For successful exports, `error.type` MUST NOT be set. For failed exports,
// `error.type` MUST contain the failure cause.
// For exporters with partial success semantics (e.g. OTLP with `rejected_spans`
// ), rejected spans MUST count as failed and only non-rejected spans count as
// success.
// If no rejection reason is available, `rejected` SHOULD be used as value for
// `error.type`.
func (m SDKExporterSpanExported) AddSet(ctx context.Context, incr int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64Counter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64Counter.Add(ctx, incr, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents the describes a class of error the operation ended
// with.
func (SDKExporterSpanExported) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrComponentName returns an optional attribute for the "otel.component.name"
// semantic convention. It represents a name uniquely identifying the instance of
// the OpenTelemetry component within its containing SDK instance.
func (SDKExporterSpanExported) AttrComponentName(val string) attribute.KeyValue {
	return attribute.String("otel.component.name", val)
}

// AttrComponentType returns an optional attribute for the "otel.component.type"
// semantic convention. It represents a name identifying the type of the
// OpenTelemetry component.
func (SDKExporterSpanExported) AttrComponentType(val ComponentTypeAttr) attribute.KeyValue {
	return attribute.String("otel.component.type", string(val))
}

// AttrServerAddress returns an optional attribute for the "server.address"
// semantic convention. It represents the server domain name if available without
// reverse DNS lookup; otherwise, IP address or Unix domain socket name.
func (SDKExporterSpanExported) AttrServerAddress(val string) attribute.KeyValue {
	return attribute.String("server.address", val)
}

// AttrServerPort returns an optional attribute for the "server.port" semantic
// convention. It represents the server port number.
func (SDKExporterSpanExported) AttrServerPort(val int) attribute.KeyValue {
	return attribute.Int("server.port", val)
}

// SDKExporterSpanInflight is an instrument used to record metric values
// conforming to the "otel.sdk.exporter.span.inflight" semantic conventions. It
// represents the number of spans which were passed to the exporter, but that
// have not been exported yet (neither successful, nor failed).
type SDKExporterSpanInflight struct {
	metric.Int64UpDownCounter
}

// NewSDKExporterSpanInflight returns a new SDKExporterSpanInflight instrument.
func NewSDKExporterSpanInflight(
	m metric.Meter,
	opt ...metric.Int64UpDownCounterOption,
) (SDKExporterSpanInflight, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKExporterSpanInflight{noop.Int64UpDownCounter{}}, nil
	}

	i, err := m.Int64UpDownCounter(
		"otel.sdk.exporter.span.inflight",
		append([]metric.Int64UpDownCounterOption{
			metric.WithDescription("The number of spans which were passed to the exporter, but that have not been exported yet (neither successful, nor failed)."),
			metric.WithUnit("{span}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKExporterSpanInflight{noop.Int64UpDownCounter{}}, err
	}
	return SDKExporterSpanInflight{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKExporterSpanInflight) Inst() metric.Int64UpDownCounter {
	return m.Int64UpDownCounter
}

// Name returns the semantic convention name of the instrument.
func (SDKExporterSpanInflight) Name() string {
	return "otel.sdk.exporter.span.inflight"
}

// Unit returns the semantic convention unit of the instrument
func (SDKExporterSpanInflight) Unit() string {
	return "{span}"
}

// Description returns the semantic convention description of the instrument
func (SDKExporterSpanInflight) Description() string {
	return "The number of spans which were passed to the exporter, but that have not been exported yet (neither successful, nor failed)."
}

// Add adds incr to the existing count for attrs.
//
// All additional attrs passed are included in the recorded value.
//
// For successful exports, `error.type` MUST NOT be set. For failed exports,
// `error.type` MUST contain the failure cause.
func (m SDKExporterSpanInflight) Add(
	ctx context.Context,
	incr int64,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64UpDownCounter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			attrs...,
		),
	)

	m.Int64UpDownCounter.Add(ctx, incr, *o...)
}

// AddSet adds incr to the existing count for set.
//
// For successful exports, `error.type` MUST NOT be set. For failed exports,
// `error.type` MUST contain the failure cause.
func (m SDKExporterSpanInflight) AddSet(ctx context.Context, incr int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64UpDownCounter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64UpDownCounter.Add(ctx, incr, *o...)
}

// AttrComponentName returns an optional attribute for the "otel.component.name"
// semantic convention. It represents a name uniquely identifying the instance of
// the OpenTelemetry component within its containing SDK instance.
func (SDKExporterSpanInflight) AttrComponentName(val string) attribute.KeyValue {
	return attribute.String("otel.component.name", val)
}

// AttrComponentType returns an optional attribute for the "otel.component.type"
// semantic convention. It represents a name identifying the type of the
// OpenTelemetry component.
func (SDKExporterSpanInflight) AttrComponentType(val ComponentTypeAttr) attribute.KeyValue {
	return attribute.String("otel.component.type", string(val))
}

// AttrServerAddress returns an optional attribute for the "server.address"
// semantic convention. It represents the server domain name if available without
// reverse DNS lookup; otherwise, IP address or Unix domain socket name.
func (SDKExporterSpanInflight) AttrServerAddress(val string) attribute.KeyValue {
	return attribute.String("server.address", val)
}

// AttrServerPort returns an optional attribute for the "server.port" semantic
// convention. It represents the server port number.
func (SDKExporterSpanInflight) AttrServerPort(val int) attribute.KeyValue {
	return attribute.Int("server.port", val)
}

// SDKLogCreated is an instrument used to record metric values conforming to the
// "otel.sdk.log.created" semantic conventions. It represents the number of logs
// submitted to enabled SDK Loggers.
type SDKLogCreated struct {
	metric.Int64Counter
}

// NewSDKLogCreated returns a new SDKLogCreated instrument.
func NewSDKLogCreated(
	m metric.Meter,
	opt ...metric.Int64CounterOption,
) (SDKLogCreated, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKLogCreated{noop.Int64Counter{}}, nil
	}

	i, err := m.Int64Counter(
		"otel.sdk.log.created",
		append([]metric.Int64CounterOption{
			metric.WithDescription("The number of logs submitted to enabled SDK Loggers."),
			metric.WithUnit("{log_record}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKLogCreated{noop.Int64Counter{}}, err
	}
	return SDKLogCreated{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKLogCreated) Inst() metric.Int64Counter {
	return m.Int64Counter
}

// Name returns the semantic convention name of the instrument.
func (SDKLogCreated) Name() string {
	return "otel.sdk.log.created"
}

// Unit returns the semantic convention unit of the instrument
func (SDKLogCreated) Unit() string {
	return "{log_record}"
}

// Description returns the semantic convention description of the instrument
func (SDKLogCreated) Description() string {
	return "The number of logs submitted to enabled SDK Loggers."
}

// Add adds incr to the existing count for attrs.
func (m SDKLogCreated) Add(ctx context.Context, incr int64, attrs ...attribute.KeyValue) {
	if len(attrs) == 0 {
		m.Int64Counter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributes(attrs...))
	m.Int64Counter.Add(ctx, incr, *o...)
}

// AddSet adds incr to the existing count for set.
func (m SDKLogCreated) AddSet(ctx context.Context, incr int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64Counter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64Counter.Add(ctx, incr, *o...)
}

// SDKMetricReaderCollectionDuration is an instrument used to record metric
// values conforming to the "otel.sdk.metric_reader.collection.duration" semantic
// conventions. It represents the duration of the collect operation of the metric
// reader.
type SDKMetricReaderCollectionDuration struct {
	metric.Float64Histogram
}

// NewSDKMetricReaderCollectionDuration returns a new
// SDKMetricReaderCollectionDuration instrument.
func NewSDKMetricReaderCollectionDuration(
	m metric.Meter,
	opt ...metric.Float64HistogramOption,
) (SDKMetricReaderCollectionDuration, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKMetricReaderCollectionDuration{noop.Float64Histogram{}}, nil
	}

	i, err := m.Float64Histogram(
		"otel.sdk.metric_reader.collection.duration",
		append([]metric.Float64HistogramOption{
			metric.WithDescription("The duration of the collect operation of the metric reader."),
			metric.WithUnit("s"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKMetricReaderCollectionDuration{noop.Float64Histogram{}}, err
	}
	return SDKMetricReaderCollectionDuration{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKMetricReaderCollectionDuration) Inst() metric.Float64Histogram {
	return m.Float64Histogram
}

// Name returns the semantic convention name of the instrument.
func (SDKMetricReaderCollectionDuration) Name() string {
	return "otel.sdk.metric_reader.collection.duration"
}

// Unit returns the semantic convention unit of the instrument
func (SDKMetricReaderCollectionDuration) Unit() string {
	return "s"
}

// Description returns the semantic convention description of the instrument
func (SDKMetricReaderCollectionDuration) Description() string {
	return "The duration of the collect operation of the metric reader."
}

// Record records val to the current distribution for attrs.
//
// All additional attrs passed are included in the recorded value.
//
// For successful collections, `error.type` MUST NOT be set. For failed
// collections, `error.type` SHOULD contain the failure cause.
// It can happen that metrics collection is successful for some MetricProducers,
// while others fail. In that case `error.type` SHOULD be set to any of the
// failure causes.
func (m SDKMetricReaderCollectionDuration) Record(
	ctx context.Context,
	val float64,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Float64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			attrs...,
		),
	)

	m.Float64Histogram.Record(ctx, val, *o...)
}

// RecordSet records val to the current distribution for set.
//
// For successful collections, `error.type` MUST NOT be set. For failed
// collections, `error.type` SHOULD contain the failure cause.
// It can happen that metrics collection is successful for some MetricProducers,
// while others fail. In that case `error.type` SHOULD be set to any of the
// failure causes.
func (m SDKMetricReaderCollectionDuration) RecordSet(ctx context.Context, val float64, set attribute.Set) {
	if set.Len() == 0 {
		m.Float64Histogram.Record(ctx, val)
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Float64Histogram.Record(ctx, val, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents the describes a class of error the operation ended
// with.
func (SDKMetricReaderCollectionDuration) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrComponentName returns an optional attribute for the "otel.component.name"
// semantic convention. It represents a name uniquely identifying the instance of
// the OpenTelemetry component within its containing SDK instance.
func (SDKMetricReaderCollectionDuration) AttrComponentName(val string) attribute.KeyValue {
	return attribute.String("otel.component.name", val)
}

// AttrComponentType returns an optional attribute for the "otel.component.type"
// semantic convention. It represents a name identifying the type of the
// OpenTelemetry component.
func (SDKMetricReaderCollectionDuration) AttrComponentType(val ComponentTypeAttr) attribute.KeyValue {
	return attribute.String("otel.component.type", string(val))
}

// SDKProcessorLogProcessed is an instrument used to record metric values
// conforming to the "otel.sdk.processor.log.processed" semantic conventions. It
// represents the number of log records for which the processing has finished,
// either successful or failed.
type SDKProcessorLogProcessed struct {
	metric.Int64Counter
}

// NewSDKProcessorLogProcessed returns a new SDKProcessorLogProcessed instrument.
func NewSDKProcessorLogProcessed(
	m metric.Meter,
	opt ...metric.Int64CounterOption,
) (SDKProcessorLogProcessed, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKProcessorLogProcessed{noop.Int64Counter{}}, nil
	}

	i, err := m.Int64Counter(
		"otel.sdk.processor.log.processed",
		append([]metric.Int64CounterOption{
			metric.WithDescription("The number of log records for which the processing has finished, either successful or failed."),
			metric.WithUnit("{log_record}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKProcessorLogProcessed{noop.Int64Counter{}}, err
	}
	return SDKProcessorLogProcessed{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKProcessorLogProcessed) Inst() metric.Int64Counter {
	return m.Int64Counter
}

// Name returns the semantic convention name of the instrument.
func (SDKProcessorLogProcessed) Name() string {
	return "otel.sdk.processor.log.processed"
}

// Unit returns the semantic convention unit of the instrument
func (SDKProcessorLogProcessed) Unit() string {
	return "{log_record}"
}

// Description returns the semantic convention description of the instrument
func (SDKProcessorLogProcessed) Description() string {
	return "The number of log records for which the processing has finished, either successful or failed."
}

// Add adds incr to the existing count for attrs.
//
// All additional attrs passed are included in the recorded value.
//
// For successful processing, `error.type` MUST NOT be set. For failed
// processing, `error.type` MUST contain the failure cause.
// For the SDK Simple and Batching Log Record Processor a log record is
// considered to be processed already when it has been submitted to the exporter,
// not when the corresponding export call has finished.
func (m SDKProcessorLogProcessed) Add(
	ctx context.Context,
	incr int64,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64Counter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			attrs...,
		),
	)

	m.Int64Counter.Add(ctx, incr, *o...)
}

// AddSet adds incr to the existing count for set.
//
// For successful processing, `error.type` MUST NOT be set. For failed
// processing, `error.type` MUST contain the failure cause.
// For the SDK Simple and Batching Log Record Processor a log record is
// considered to be processed already when it has been submitted to the exporter,
// not when the corresponding export call has finished.
func (m SDKProcessorLogProcessed) AddSet(ctx context.Context, incr int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64Counter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64Counter.Add(ctx, incr, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents a low-cardinality description of the failure reason.
// SDK Batching Log Record Processors MUST use `queue_full` for log records
// dropped due to a full queue.
func (SDKProcessorLogProcessed) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrComponentName returns an optional attribute for the "otel.component.name"
// semantic convention. It represents a name uniquely identifying the instance of
// the OpenTelemetry component within its containing SDK instance.
func (SDKProcessorLogProcessed) AttrComponentName(val string) attribute.KeyValue {
	return attribute.String("otel.component.name", val)
}

// AttrComponentType returns an optional attribute for the "otel.component.type"
// semantic convention. It represents a name identifying the type of the
// OpenTelemetry component.
func (SDKProcessorLogProcessed) AttrComponentType(val ComponentTypeAttr) attribute.KeyValue {
	return attribute.String("otel.component.type", string(val))
}

// SDKProcessorLogQueueCapacity is an instrument used to record metric values
// conforming to the "otel.sdk.processor.log.queue.capacity" semantic
// conventions. It represents the maximum number of log records the queue of a
// given instance of an SDK Log Record processor can hold.
type SDKProcessorLogQueueCapacity struct {
	metric.Int64ObservableUpDownCounter
}

// NewSDKProcessorLogQueueCapacity returns a new SDKProcessorLogQueueCapacity
// instrument.
func NewSDKProcessorLogQueueCapacity(
	m metric.Meter,
	opt ...metric.Int64ObservableUpDownCounterOption,
) (SDKProcessorLogQueueCapacity, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKProcessorLogQueueCapacity{noop.Int64ObservableUpDownCounter{}}, nil
	}

	i, err := m.Int64ObservableUpDownCounter(
		"otel.sdk.processor.log.queue.capacity",
		append([]metric.Int64ObservableUpDownCounterOption{
			metric.WithDescription("The maximum number of log records the queue of a given instance of an SDK Log Record processor can hold."),
			metric.WithUnit("{log_record}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKProcessorLogQueueCapacity{noop.Int64ObservableUpDownCounter{}}, err
	}
	return SDKProcessorLogQueueCapacity{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKProcessorLogQueueCapacity) Inst() metric.Int64ObservableUpDownCounter {
	return m.Int64ObservableUpDownCounter
}

// Name returns the semantic convention name of the instrument.
func (SDKProcessorLogQueueCapacity) Name() string {
	return "otel.sdk.processor.log.queue.capacity"
}

// Unit returns the semantic convention unit of the instrument
func (SDKProcessorLogQueueCapacity) Unit() string {
	return "{log_record}"
}

// Description returns the semantic convention description of the instrument
func (SDKProcessorLogQueueCapacity) Description() string {
	return "The maximum number of log records the queue of a given instance of an SDK Log Record processor can hold."
}

// AttrComponentName returns an optional attribute for the "otel.component.name"
// semantic convention. It represents a name uniquely identifying the instance of
// the OpenTelemetry component within its containing SDK instance.
func (SDKProcessorLogQueueCapacity) AttrComponentName(val string) attribute.KeyValue {
	return attribute.String("otel.component.name", val)
}

// AttrComponentType returns an optional attribute for the "otel.component.type"
// semantic convention. It represents a name identifying the type of the
// OpenTelemetry component.
func (SDKProcessorLogQueueCapacity) AttrComponentType(val ComponentTypeAttr) attribute.KeyValue {
	return attribute.String("otel.component.type", string(val))
}

// SDKProcessorLogQueueSize is an instrument used to record metric values
// conforming to the "otel.sdk.processor.log.queue.size" semantic conventions. It
// represents the number of log records in the queue of a given instance of an
// SDK log processor.
type SDKProcessorLogQueueSize struct {
	metric.Int64ObservableUpDownCounter
}

// NewSDKProcessorLogQueueSize returns a new SDKProcessorLogQueueSize instrument.
func NewSDKProcessorLogQueueSize(
	m metric.Meter,
	opt ...metric.Int64ObservableUpDownCounterOption,
) (SDKProcessorLogQueueSize, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKProcessorLogQueueSize{noop.Int64ObservableUpDownCounter{}}, nil
	}

	i, err := m.Int64ObservableUpDownCounter(
		"otel.sdk.processor.log.queue.size",
		append([]metric.Int64ObservableUpDownCounterOption{
			metric.WithDescription("The number of log records in the queue of a given instance of an SDK log processor."),
			metric.WithUnit("{log_record}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKProcessorLogQueueSize{noop.Int64ObservableUpDownCounter{}}, err
	}
	return SDKProcessorLogQueueSize{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKProcessorLogQueueSize) Inst() metric.Int64ObservableUpDownCounter {
	return m.Int64ObservableUpDownCounter
}

// Name returns the semantic convention name of the instrument.
func (SDKProcessorLogQueueSize) Name() string {
	return "otel.sdk.processor.log.queue.size"
}

// Unit returns the semantic convention unit of the instrument
func (SDKProcessorLogQueueSize) Unit() string {
	return "{log_record}"
}

// Description returns the semantic convention description of the instrument
func (SDKProcessorLogQueueSize) Description() string {
	return "The number of log records in the queue of a given instance of an SDK log processor."
}

// AttrComponentName returns an optional attribute for the "otel.component.name"
// semantic convention. It represents a name uniquely identifying the instance of
// the OpenTelemetry component within its containing SDK instance.
func (SDKProcessorLogQueueSize) AttrComponentName(val string) attribute.KeyValue {
	return attribute.String("otel.component.name", val)
}

// AttrComponentType returns an optional attribute for the "otel.component.type"
// semantic convention. It represents a name identifying the type of the
// OpenTelemetry component.
func (SDKProcessorLogQueueSize) AttrComponentType(val ComponentTypeAttr) attribute.KeyValue {
	return attribute.String("otel.component.type", string(val))
}

// SDKProcessorSpanProcessed is an instrument used to record metric values
// conforming to the "otel.sdk.processor.span.processed" semantic conventions. It
// represents the number of spans for which the processing has finished, either
// successful or failed.
type SDKProcessorSpanProcessed struct {
	metric.Int64Counter
}

// NewSDKProcessorSpanProcessed returns a new SDKProcessorSpanProcessed
// instrument.
func NewSDKProcessorSpanProcessed(
	m metric.Meter,
	opt ...metric.Int64CounterOption,
) (SDKProcessorSpanProcessed, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKProcessorSpanProcessed{noop.Int64Counter{}}, nil
	}

	i, err := m.Int64Counter(
		"otel.sdk.processor.span.processed",
		append([]metric.Int64CounterOption{
			metric.WithDescription("The number of spans for which the processing has finished, either successful or failed."),
			metric.WithUnit("{span}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKProcessorSpanProcessed{noop.Int64Counter{}}, err
	}
	return SDKProcessorSpanProcessed{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKProcessorSpanProcessed) Inst() metric.Int64Counter {
	return m.Int64Counter
}

// Name returns the semantic convention name of the instrument.
func (SDKProcessorSpanProcessed) Name() string {
	return "otel.sdk.processor.span.processed"
}

// Unit returns the semantic convention unit of the instrument
func (SDKProcessorSpanProcessed) Unit() string {
	return "{span}"
}

// Description returns the semantic convention description of the instrument
func (SDKProcessorSpanProcessed) Description() string {
	return "The number of spans for which the processing has finished, either successful or failed."
}

// Add adds incr to the existing count for attrs.
//
// All additional attrs passed are included in the recorded value.
//
// For successful processing, `error.type` MUST NOT be set. For failed
// processing, `error.type` MUST contain the failure cause.
// For the SDK Simple and Batching Span Processor a span is considered to be
// processed already when it has been submitted to the exporter, not when the
// corresponding export call has finished.
func (m SDKProcessorSpanProcessed) Add(
	ctx context.Context,
	incr int64,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64Counter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			attrs...,
		),
	)

	m.Int64Counter.Add(ctx, incr, *o...)
}

// AddSet adds incr to the existing count for set.
//
// For successful processing, `error.type` MUST NOT be set. For failed
// processing, `error.type` MUST contain the failure cause.
// For the SDK Simple and Batching Span Processor a span is considered to be
// processed already when it has been submitted to the exporter, not when the
// corresponding export call has finished.
func (m SDKProcessorSpanProcessed) AddSet(ctx context.Context, incr int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64Counter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64Counter.Add(ctx, incr, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents a low-cardinality description of the failure reason.
// SDK Batching Span Processors MUST use `queue_full` for spans dropped due to a
// full queue.
func (SDKProcessorSpanProcessed) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrComponentName returns an optional attribute for the "otel.component.name"
// semantic convention. It represents a name uniquely identifying the instance of
// the OpenTelemetry component within its containing SDK instance.
func (SDKProcessorSpanProcessed) AttrComponentName(val string) attribute.KeyValue {
	return attribute.String("otel.component.name", val)
}

// AttrComponentType returns an optional attribute for the "otel.component.type"
// semantic convention. It represents a name identifying the type of the
// OpenTelemetry component.
func (SDKProcessorSpanProcessed) AttrComponentType(val ComponentTypeAttr) attribute.KeyValue {
	return attribute.String("otel.component.type", string(val))
}

// SDKProcessorSpanQueueCapacity is an instrument used to record metric values
// conforming to the "otel.sdk.processor.span.queue.capacity" semantic
// conventions. It represents the maximum number of spans the queue of a given
// instance of an SDK span processor can hold.
type SDKProcessorSpanQueueCapacity struct {
	metric.Int64ObservableUpDownCounter
}

// NewSDKProcessorSpanQueueCapacity returns a new SDKProcessorSpanQueueCapacity
// instrument.
func NewSDKProcessorSpanQueueCapacity(
	m metric.Meter,
	opt ...metric.Int64ObservableUpDownCounterOption,
) (SDKProcessorSpanQueueCapacity, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKProcessorSpanQueueCapacity{noop.Int64ObservableUpDownCounter{}}, nil
	}

	i, err := m.Int64ObservableUpDownCounter(
		"otel.sdk.processor.span.queue.capacity",
		append([]metric.Int64ObservableUpDownCounterOption{
			metric.WithDescription("The maximum number of spans the queue of a given instance of an SDK span processor can hold."),
			metric.WithUnit("{span}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKProcessorSpanQueueCapacity{noop.Int64ObservableUpDownCounter{}}, err
	}
	return SDKProcessorSpanQueueCapacity{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKProcessorSpanQueueCapacity) Inst() metric.Int64ObservableUpDownCounter {
	return m.Int64ObservableUpDownCounter
}

// Name returns the semantic convention name of the instrument.
func (SDKProcessorSpanQueueCapacity) Name() string {
	return "otel.sdk.processor.span.queue.capacity"
}

// Unit returns the semantic convention unit of the instrument
func (SDKProcessorSpanQueueCapacity) Unit() string {
	return "{span}"
}

// Description returns the semantic convention description of the instrument
func (SDKProcessorSpanQueueCapacity) Description() string {
	return "The maximum number of spans the queue of a given instance of an SDK span processor can hold."
}

// AttrComponentName returns an optional attribute for the "otel.component.name"
// semantic convention. It represents a name uniquely identifying the instance of
// the OpenTelemetry component within its containing SDK instance.
func (SDKProcessorSpanQueueCapacity) AttrComponentName(val string) attribute.KeyValue {
	return attribute.String("otel.component.name", val)
}

// AttrComponentType returns an optional attribute for the "otel.component.type"
// semantic convention. It represents a name identifying the type of the
// OpenTelemetry component.
func (SDKProcessorSpanQueueCapacity) AttrComponentType(val ComponentTypeAttr) attribute.KeyValue {
	return attribute.String("otel.component.type", string(val))
}

// SDKProcessorSpanQueueSize is an instrument used to record metric values
// conforming to the "otel.sdk.processor.span.queue.size" semantic conventions.
// It represents the number of spans in the queue of a given instance of an SDK
// span processor.
type SDKProcessorSpanQueueSize struct {
	metric.Int64ObservableUpDownCounter
}

// NewSDKProcessorSpanQueueSize returns a new SDKProcessorSpanQueueSize
// instrument.
func NewSDKProcessorSpanQueueSize(
	m metric.Meter,
	opt ...metric.Int64ObservableUpDownCounterOption,
) (SDKProcessorSpanQueueSize, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKProcessorSpanQueueSize{noop.Int64ObservableUpDownCounter{}}, nil
	}

	i, err := m.Int64ObservableUpDownCounter(
		"otel.sdk.processor.span.queue.size",
		append([]metric.Int64ObservableUpDownCounterOption{
			metric.WithDescription("The number of spans in the queue of a given instance of an SDK span processor."),
			metric.WithUnit("{span}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKProcessorSpanQueueSize{noop.Int64ObservableUpDownCounter{}}, err
	}
	return SDKProcessorSpanQueueSize{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKProcessorSpanQueueSize) Inst() metric.Int64ObservableUpDownCounter {
	return m.Int64ObservableUpDownCounter
}

// Name returns the semantic convention name of the instrument.
func (SDKProcessorSpanQueueSize) Name() string {
	return "otel.sdk.processor.span.queue.size"
}

// Unit returns the semantic convention unit of the instrument
func (SDKProcessorSpanQueueSize) Unit() string {
	return "{span}"
}

// Description returns the semantic convention description of the instrument
func (SDKProcessorSpanQueueSize) Description() string {
	return "The number of spans in the queue of a given instance of an SDK span processor."
}

// AttrComponentName returns an optional attribute for the "otel.component.name"
// semantic convention. It represents a name uniquely identifying the instance of
// the OpenTelemetry component within its containing SDK instance.
func (SDKProcessorSpanQueueSize) AttrComponentName(val string) attribute.KeyValue {
	return attribute.String("otel.component.name", val)
}

// AttrComponentType returns an optional attribute for the "otel.component.type"
// semantic convention. It represents a name identifying the type of the
// OpenTelemetry component.
func (SDKProcessorSpanQueueSize) AttrComponentType(val ComponentTypeAttr) attribute.KeyValue {
	return attribute.String("otel.component.type", string(val))
}

// SDKSpanLive is an instrument used to record metric values conforming to the
// "otel.sdk.span.live" semantic conventions. It represents the number of created
// spans with `recording=true` for which the end operation has not been called
// yet.
type SDKSpanLive struct {
	metric.Int64UpDownCounter
}

// NewSDKSpanLive returns a new SDKSpanLive instrument.
func NewSDKSpanLive(
	m metric.Meter,
	opt ...metric.Int64UpDownCounterOption,
) (SDKSpanLive, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKSpanLive{noop.Int64UpDownCounter{}}, nil
	}

	i, err := m.Int64UpDownCounter(
		"otel.sdk.span.live",
		append([]metric.Int64UpDownCounterOption{
			metric.WithDescription("The number of created spans with `recording=true` for which the end operation has not been called yet."),
			metric.WithUnit("{span}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKSpanLive{noop.Int64UpDownCounter{}}, err
	}
	return SDKSpanLive{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKSpanLive) Inst() metric.Int64UpDownCounter {
	return m.Int64UpDownCounter
}

// Name returns the semantic convention name of the instrument.
func (SDKSpanLive) Name() string {
	return "otel.sdk.span.live"
}

// Unit returns the semantic convention unit of the instrument
func (SDKSpanLive) Unit() string {
	return "{span}"
}

// Description returns the semantic convention description of the instrument
func (SDKSpanLive) Description() string {
	return "The number of created spans with `recording=true` for which the end operation has not been called yet."
}

// Add adds incr to the existing count for attrs.
//
// All additional attrs passed are included in the recorded value.
func (m SDKSpanLive) Add(
	ctx context.Context,
	incr int64,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64UpDownCounter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			attrs...,
		),
	)

	m.Int64UpDownCounter.Add(ctx, incr, *o...)
}

// AddSet adds incr to the existing count for set.
func (m SDKSpanLive) AddSet(ctx context.Context, incr int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64UpDownCounter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64UpDownCounter.Add(ctx, incr, *o...)
}

// AttrSpanSamplingResult returns an optional attribute for the
// "otel.span.sampling_result" semantic convention. It represents the result
// value of the sampler for this span.
func (SDKSpanLive) AttrSpanSamplingResult(val SpanSamplingResultAttr) attribute.KeyValue {
	return attribute.String("otel.span.sampling_result", string(val))
}

// SDKSpanStarted is an instrument used to record metric values conforming to the
// "otel.sdk.span.started" semantic conventions. It represents the number of
// created spans.
type SDKSpanStarted struct {
	metric.Int64Counter
}

// NewSDKSpanStarted returns a new SDKSpanStarted instrument.
func NewSDKSpanStarted(
	m metric.Meter,
	opt ...metric.Int64CounterOption,
) (SDKSpanStarted, error) {
	// Check if the meter is nil.
	if m == nil {
		return SDKSpanStarted{noop.Int64Counter{}}, nil
	}

	i, err := m.Int64Counter(
		"otel.sdk.span.started",
		append([]metric.Int64CounterOption{
			metric.WithDescription("The number of created spans."),
			metric.WithUnit("{span}"),
		}, opt...)...,
	)
	if err != nil {
	    return SDKSpanStarted{noop.Int64Counter{}}, err
	}
	return SDKSpanStarted{i}, nil
}

// Inst returns the underlying metric instrument.
func (m SDKSpanStarted) Inst() metric.Int64Counter {
	return m.Int64Counter
}

// Name returns the semantic convention name of the instrument.
func (SDKSpanStarted) Name() string {
	return "otel.sdk.span.started"
}

// Unit returns the semantic convention unit of the instrument
func (SDKSpanStarted) Unit() string {
	return "{span}"
}

// Description returns the semantic convention description of the instrument
func (SDKSpanStarted) Description() string {
	return "The number of created spans."
}

// Add adds incr to the existing count for attrs.
//
// All additional attrs passed are included in the recorded value.
//
// Implementations MUST record this metric for all spans, even for non-recording
// ones.
func (m SDKSpanStarted) Add(
	ctx context.Context,
	incr int64,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64Counter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			attrs...,
		),
	)

	m.Int64Counter.Add(ctx, incr, *o...)
}

// AddSet adds incr to the existing count for set.
//
// Implementations MUST record this metric for all spans, even for non-recording
// ones.
func (m SDKSpanStarted) AddSet(ctx context.Context, incr int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64Counter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64Counter.Add(ctx, incr, *o...)
}

// AttrSpanParentOrigin returns an optional attribute for the
// "otel.span.parent.origin" semantic convention. It represents the determines
// whether the span has a parent span, and if so, [whether it is a remote parent]
// .
//
// [whether it is a remote parent]: https://opentelemetry.io/docs/specs/otel/trace/api/#isremote
func (SDKSpanStarted) AttrSpanParentOrigin(val SpanParentOriginAttr) attribute.KeyValue {
	return attribute.String("otel.span.parent.origin", string(val))
}

// AttrSpanSamplingResult returns an optional attribute for the
// "otel.span.sampling_result" semantic convention. It represents the result
// value of the sampler for this span.
func (SDKSpanStarted) AttrSpanSamplingResult(val SpanSamplingResultAttr) attribute.KeyValue {
	return attribute.String("otel.span.sampling_result", string(val))
}