// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package observ // import "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc/internal/observ"

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc/codes"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc/internal"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc/internal/x"
	"go.opentelemetry.io/otel/internal/global"
	"go.opentelemetry.io/otel/metric"
	semconv "go.opentelemetry.io/otel/semconv/v1.37.0"
	"go.opentelemetry.io/otel/semconv/v1.37.0/otelconv"
)

const (
	// ScopeName is the unique name of the meter used for instrumentation.
	ScopeName = "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc/internal/observ"

	// SchemaURL is the schema URL of the metrics produced by this
	// instrumentation.
	SchemaURL = semconv.SchemaURL

	// Version is the current version of this instrumentation.
	//
	// This matches the version of the exporter.
	Version = internal.Version
)

var (
	measureAttrsPool = &sync.Pool{
		New: func() any {
			const n = 1 + // component.name
				1 + // component.type
				1 + // server.addr
				1 + // server.port
				1 + // error.type
				1 // rpc.grpc.status_code
			s := make([]attribute.KeyValue, 0, n)
			// Return a pointer to a slice instead of a slice itself
			// to avoid allocations on every call.
			return &s
		},
	}

	addOptPool = &sync.Pool{
		New: func() any {
			const n = 1 // WithAttributeSet
			o := make([]metric.AddOption, 0, n)
			return &o
		},
	}

	recordOptPool = &sync.Pool{
		New: func() any {
			const n = 1 // WithAttributeSet
			o := make([]metric.RecordOption, 0, n)
			return &o
		},
	}
)

func get[T any](p *sync.Pool) *[]T { return p.Get().(*[]T) }

func put[T any](p *sync.Pool, s *[]T) {
	*s = (*s)[:0] // Reset.
	p.Put(s)
}

// ComponentName returns the component name for the exporter with the
// provided ID.
func ComponentName(id int64) string {
	t := semconv.OTelComponentTypeOtlpGRPCSpanExporter.Value.AsString()
	return fmt.Sprintf("%s/%d", t, id)
}

// Instrumentation is experimental instrumentation for the exporter.
type Instrumentation struct {
	inflightSpans metric.Int64UpDownCounter
	exportedSpans metric.Int64Counter
	opDuration    metric.Float64Histogram

	attrs  []attribute.KeyValue
	addOpt metric.AddOption
	recOpt metric.RecordOption
}

// NewInstrumentation returns instrumentation for an OTLP over gPRC trace
// exporter with the provided ID using the global MeterProvider.
//
// The id should be the unique exporter instance ID. It is used
// to set the "component.name" attribute.
//
// The target is the endpoint the exporter is exporting to.
//
// If the experimental observability is disabled, nil is returned.
func NewInstrumentation(id int64, target string) (*Instrumentation, error) {
	if !x.Observability.Enabled() {
		return nil, nil
	}

	attrs := BaseAttrs(id, target)
	i := &Instrumentation{
		attrs:  attrs,
		addOpt: metric.WithAttributeSet(attribute.NewSet(attrs...)),

		// Do not modify attrs (NewSet sorts in-place), make a new slice.
		recOpt: metric.WithAttributeSet(attribute.NewSet(append(
			// Default to OK status code.
			[]attribute.KeyValue{semconv.RPCGRPCStatusCodeOk},
			attrs...,
		)...)),
	}

	mp := otel.GetMeterProvider()
	m := mp.Meter(
		ScopeName,
		metric.WithInstrumentationVersion(Version),
		metric.WithSchemaURL(SchemaURL),
	)

	var err error

	inflightSpans, e := otelconv.NewSDKExporterSpanInflight(m)
	if e != nil {
		e = fmt.Errorf("failed to create span inflight metric: %w", e)
		err = errors.Join(err, e)
	}
	i.inflightSpans = inflightSpans.Inst()

	exportedSpans, e := otelconv.NewSDKExporterSpanExported(m)
	if e != nil {
		e = fmt.Errorf("failed to create span exported metric: %w", e)
		err = errors.Join(err, e)
	}
	i.exportedSpans = exportedSpans.Inst()

	opDuration, e := otelconv.NewSDKExporterOperationDuration(m)
	if e != nil {
		e = fmt.Errorf("failed to create operation duration metric: %w", e)
		err = errors.Join(err, e)
	}
	i.opDuration = opDuration.Inst()

	return i, err
}

// BaseAttrs returns the base attributes for the exporter with the provided ID
// and target.
//
// The id should be the unique exporter instance ID. It is used
// to set the "component.name" attribute.
//
// The target is the gRPC target the exporter is exporting to. It is expected
// to be the output of the Client's CanonicalTarget method.
func BaseAttrs(id int64, target string) []attribute.KeyValue {
	host, port, err := ParseCanonicalTarget(target)
	if err != nil || (host == "" && port < 0) {
		if err != nil {
			global.Debug("failed to parse target", "target", target, "error", err)
		}
		return []attribute.KeyValue{
			semconv.OTelComponentName(ComponentName(id)),
			semconv.OTelComponentTypeOtlpGRPCSpanExporter,
		}
	}

	// Do not use append so the slice is exactly allocated.

	if port < 0 {
		return []attribute.KeyValue{
			semconv.OTelComponentName(ComponentName(id)),
			semconv.OTelComponentTypeOtlpGRPCSpanExporter,
			semconv.ServerAddress(host),
		}
	}

	if host == "" {
		return []attribute.KeyValue{
			semconv.OTelComponentName(ComponentName(id)),
			semconv.OTelComponentTypeOtlpGRPCSpanExporter,
			semconv.ServerPort(port),
		}
	}

	return []attribute.KeyValue{
		semconv.OTelComponentName(ComponentName(id)),
		semconv.OTelComponentTypeOtlpGRPCSpanExporter,
		semconv.ServerAddress(host),
		semconv.ServerPort(port),
	}
}

// ExportSpans instruments the ExportSpans method of the exporter. It returns
// an [ExportOp] that must have its [ExportOp.End] method called when the
// ExportSpans method returns.
func (i *Instrumentation) ExportSpans(ctx context.Context, nSpans int) ExportOp {
	start := time.Now()

	addOpt := get[metric.AddOption](addOptPool)
	defer put(addOptPool, addOpt)
	*addOpt = append(*addOpt, i.addOpt)
	i.inflightSpans.Add(ctx, int64(nSpans), *addOpt...)

	return ExportOp{
		ctx:    ctx,
		start:  start,
		nSpans: int64(nSpans),
		inst:   i,
	}
}

// ExportOp tracks the operation being observed by [Instrumentation.ExportSpans].
type ExportOp struct {
	ctx    context.Context
	start  time.Time
	nSpans int64

	inst *Instrumentation
}

// End completes the observation of the operation being observed by a call to
// [Instrumentation.ExportSpans].
//
// Any error that is encountered is provided as err.
//
// If err is not nil, all spans will be recorded as failures unless error is of
// type [internal.PartialSuccess]. In the case of a PartialSuccess, the number
// of successfully exported spans will be determined by inspecting the
// RejectedItems field of the PartialSuccess.
func (e ExportOp) End(err error, code codes.Code) {
	addOpt := get[metric.AddOption](addOptPool)
	defer put(addOptPool, addOpt)
	*addOpt = append(*addOpt, e.inst.addOpt)

	e.inst.inflightSpans.Add(e.ctx, -e.nSpans, *addOpt...)

	success := successful(e.nSpans, err)
	// Record successfully exported spans, even if the value is 0 which are
	// meaningful to distribution aggregations.
	e.inst.exportedSpans.Add(e.ctx, success, *addOpt...)

	if err != nil {
		attrs := get[attribute.KeyValue](measureAttrsPool)
		defer put(measureAttrsPool, attrs)
		*attrs = append(*attrs, e.inst.attrs...)
		*attrs = append(*attrs, semconv.ErrorType(err))

		// Do not inefficiently make a copy of attrs by using
		// WithAttributes instead of WithAttributeSet.
		o := metric.WithAttributeSet(attribute.NewSet(*attrs...))
		// Reset addOpt with new attribute set.
		*addOpt = append((*addOpt)[:0], o)

		e.inst.exportedSpans.Add(e.ctx, e.nSpans-success, *addOpt...)
	}

	recOpt := get[metric.RecordOption](recordOptPool)
	defer put(recordOptPool, recOpt)
	*recOpt = append(*recOpt, e.inst.recordOption(err, code))

	d := time.Since(e.start).Seconds()
	e.inst.opDuration.Record(e.ctx, d, *recOpt...)
}

// recordOption returns a RecordOption with attributes representing the
// outcome of the operation being recorded.
//
// If err is nil and code is codes.OK, the default recOpt of the
// Instrumentation is returned.
//
// If err is not nil or code is not codes.OK, a new RecordOption is returned
// with the base attributes of the Instrumentation plus the rpc.grpc.status_code
// attribute set to the provided code, and if err is not nil, the error.type
// attribute set to the type of the error.
func (i *Instrumentation) recordOption(err error, code codes.Code) metric.RecordOption {
	if err == nil && code == codes.OK {
		return i.recOpt
	}

	attrs := get[attribute.KeyValue](measureAttrsPool)
	defer put(measureAttrsPool, attrs)
	*attrs = append(*attrs, i.attrs...)

	c := int64(code) // uint32 -> int64.
	*attrs = append(*attrs, semconv.RPCGRPCStatusCodeKey.Int64(c))
	if err != nil {
		*attrs = append(*attrs, semconv.ErrorType(err))
	}

	// Do not inefficiently make a copy of attrs by using WithAttributes
	// instead of WithAttributeSet.
	return metric.WithAttributeSet(attribute.NewSet(*attrs...))
}

// successful returns the number of successfully exported spans out of the n
// that were exported based on the provided error.
//
// If err is nil, n is returned. All spans were successfully exported.
//
// If err is not nil and not an [internal.PartialSuccess] error, 0 is returned.
// It is assumed all spans failed to be exported.
//
// If err is an [internal.PartialSuccess] error, the number of successfully
// exported spans is computed by subtracting the RejectedItems field from n. If
// RejectedItems is negative, n is returned. If RejectedItems is greater than
// n, 0 is returned.
func successful(n int64, err error) int64 {
	if err == nil {
		return n // All spans successfully exported.
	}
	// Split rejection calculation so successful is inlinable.
	return n - rejected(n, err)
}

var errPartialPool = &sync.Pool{
	New: func() any { return new(internal.PartialSuccess) },
}

// rejected returns how many out of the n spans exporter were rejected based on
// the provided non-nil err.
func rejected(n int64, err error) int64 {
	ps := errPartialPool.Get().(*internal.PartialSuccess)
	defer errPartialPool.Put(ps)
	// Check for partial success.
	if errors.As(err, ps) {
		// Bound RejectedItems to [0, n]. This should not be needed,
		// but be defensive as this is from an external source.
		return min(max(ps.RejectedItems, 0), n)
	}
	return n // All spans rejected.
}
