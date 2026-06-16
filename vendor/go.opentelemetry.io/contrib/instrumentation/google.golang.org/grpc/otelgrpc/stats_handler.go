// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otelgrpc // import "go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"

import (
	"context"
	"strconv"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	oldrpcconv "go.opentelemetry.io/otel/semconv/v1.37.0/rpcconv" //nolint:depguard // Use of v1.37.0 is required for backward compatibility stability opt-in.
	semconv "go.opentelemetry.io/otel/semconv/v1.40.0"
	"go.opentelemetry.io/otel/semconv/v1.40.0/rpcconv"
	"go.opentelemetry.io/otel/trace"

	grpc_codes "google.golang.org/grpc/codes"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"

	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc/internal"
)

type gRPCContextKey struct{}

type gRPCContext struct {
	metricAttrs []attribute.KeyValue
	record      bool
}

type serverHandler struct {
	*config

	tracer trace.Tracer

	duration    rpcconv.ServerCallDuration
	oldDuration oldrpcconv.ServerDuration
}

// NewServerHandler creates a stats.Handler for a gRPC server.
func NewServerHandler(opts ...Option) stats.Handler {
	c := newConfig(opts)
	if c.SpanKind == trace.SpanKindUnspecified {
		c.SpanKind = trace.SpanKindServer
	}

	h := &serverHandler{config: c}

	h.tracer = c.TracerProvider.Tracer(
		ScopeName,
		trace.WithInstrumentationVersion(Version),
	)

	meter := c.MeterProvider.Meter(
		ScopeName,
		metric.WithInstrumentationVersion(Version),
		metric.WithSchemaURL(semconv.SchemaURL),
	)

	var err error
	if c.semconvMode == semconvModeOld || c.semconvMode == semconvModeDup {
		oldDur, err := oldrpcconv.NewServerDuration(meter)
		if err != nil {
			otel.Handle(err)
		} else {
			h.oldDuration = oldDur
		}
	}

	if c.semconvMode == semconvModeNew || c.semconvMode == semconvModeDup {
		h.duration, err = rpcconv.NewServerCallDuration(
			meter,
			metric.WithExplicitBucketBoundaries(
				0.005, 0.01, 0.025, 0.05, 0.075, 0.1,
				0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10,
			),
		)
		if err != nil {
			otel.Handle(err)
		}
	}

	return h
}

// TagConn can attach some information to the given context.
func (*serverHandler) TagConn(ctx context.Context, _ *stats.ConnTagInfo) context.Context {
	return ctx
}

// HandleConn processes the Conn stats.
func (*serverHandler) HandleConn(context.Context, stats.ConnStats) {
}

// TagRPC can attach some information to the given context.
func (h *serverHandler) TagRPC(ctx context.Context, info *stats.RPCTagInfo) context.Context {
	ctx = extract(ctx, h.Propagators)

	var name string
	var attrs []attribute.KeyValue

	switch h.semconvMode {
	case semconvModeOld:
		name, attrs = internal.ParseFullMethodOld(info.FullMethodName)
	case semconvModeDup:
		var attrsNew, attrsOld []attribute.KeyValue
		name, attrsNew = internal.ParseFullMethod(info.FullMethodName)
		_, attrsOld = internal.ParseFullMethodOld(info.FullMethodName)
		// Combine both. We append New last so its rpc.method (fully qualified) wins when deduplicated.
		attrs = append(append([]attribute.KeyValue{}, attrsOld...), attrsNew...)
		attrs = append(attrs, semconv.RPCSystemNameGRPC) // New convention
	default: // semconvModeNew
		name, attrs = internal.ParseFullMethod(info.FullMethodName)
		attrs = append(attrs, semconv.RPCSystemNameGRPC)
	}

	record := true
	if h.Filter != nil {
		record = h.Filter(info)
	}

	if record {
		// Make a new slice to avoid aliasing into the same attrs slice used by metrics.
		spanAttributes := make([]attribute.KeyValue, 0, len(attrs)+len(h.SpanAttributes))
		spanAttributes = append(append(spanAttributes, attrs...), h.SpanAttributes...)
		opts := []trace.SpanStartOption{
			trace.WithSpanKind(h.SpanKind),
			trace.WithAttributes(spanAttributes...),
		}
		if h.PublicEndpoint || (h.PublicEndpointFn != nil && h.PublicEndpointFn(ctx, info)) {
			opts = append(opts, trace.WithNewRoot())
			// Linking incoming span context if any for public endpoint.
			if s := trace.SpanContextFromContext(ctx); s.IsValid() && s.IsRemote() {
				opts = append(opts, trace.WithLinks(trace.Link{SpanContext: s}))
			}
		}
		ctx, _ = h.tracer.Start(
			trace.ContextWithRemoteSpanContext(ctx, trace.SpanContextFromContext(ctx)),
			name,
			opts...,
		)
	}

	gctx := gRPCContext{
		metricAttrs: append(attrs, h.MetricAttributes...),
		record:      record,
	}

	if h.MetricAttributesFn != nil {
		extraAttrs := h.MetricAttributesFn(ctx)
		gctx.metricAttrs = append(gctx.metricAttrs, extraAttrs...)
	}

	return context.WithValue(ctx, gRPCContextKey{}, &gctx)
}

// HandleRPC processes the RPC stats.
func (h *serverHandler) HandleRPC(ctx context.Context, rs stats.RPCStats) {
	var dur metric.Float64Histogram
	if h.semconvMode == semconvModeNew || h.semconvMode == semconvModeDup {
		dur = h.duration.Inst()
	}
	var oldDur metric.Float64Histogram
	if h.semconvMode == semconvModeOld || h.semconvMode == semconvModeDup {
		oldDur = h.oldDuration.Inst()
	}
	h.handleRPC(
		ctx,
		rs,
		dur,
		oldDur,
		serverStatus,
	)
}

type clientHandler struct {
	*config

	tracer trace.Tracer

	duration    rpcconv.ClientCallDuration
	oldDuration oldrpcconv.ClientDuration
}

// NewClientHandler creates a stats.Handler for a gRPC client.
func NewClientHandler(opts ...Option) stats.Handler {
	c := newConfig(opts)
	if c.SpanKind == trace.SpanKindUnspecified {
		c.SpanKind = trace.SpanKindClient
	}

	h := &clientHandler{config: c}

	h.tracer = c.TracerProvider.Tracer(
		ScopeName,
		trace.WithInstrumentationVersion(Version),
	)

	meter := c.MeterProvider.Meter(
		ScopeName,
		metric.WithInstrumentationVersion(Version),
		metric.WithSchemaURL(semconv.SchemaURL),
	)

	var err error
	if c.semconvMode == semconvModeOld || c.semconvMode == semconvModeDup {
		oldDur, err := oldrpcconv.NewClientDuration(meter)
		if err != nil {
			otel.Handle(err)
		} else {
			h.oldDuration = oldDur
		}
	}

	if c.semconvMode == semconvModeNew || c.semconvMode == semconvModeDup {
		h.duration, err = rpcconv.NewClientCallDuration(
			meter,
			metric.WithExplicitBucketBoundaries(
				0.005, 0.01, 0.025, 0.05, 0.075, 0.1,
				0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10,
			),
		)
		if err != nil {
			otel.Handle(err)
		}
	}

	return h
}

// TagRPC can attach some information to the given context.
func (h *clientHandler) TagRPC(ctx context.Context, info *stats.RPCTagInfo) context.Context {
	var name string
	var attrs []attribute.KeyValue

	switch h.semconvMode {
	case semconvModeOld:
		name, attrs = internal.ParseFullMethodOld(info.FullMethodName)
	case semconvModeDup:
		var attrsNew, attrsOld []attribute.KeyValue
		name, attrsNew = internal.ParseFullMethod(info.FullMethodName)
		_, attrsOld = internal.ParseFullMethodOld(info.FullMethodName)
		// Combine both. We append New last so its rpc.method (fully qualified) wins when deduplicated.
		attrs = append(append([]attribute.KeyValue{}, attrsOld...), attrsNew...)
		attrs = append(attrs, semconv.RPCSystemNameGRPC) // New convention
	default: // semconvModeNew
		name, attrs = internal.ParseFullMethod(info.FullMethodName)
		attrs = append(attrs, semconv.RPCSystemNameGRPC)
	}

	record := true
	if h.Filter != nil {
		record = h.Filter(info)
	}

	if record {
		// Make a new slice to avoid aliasing into the same attrs slice used by metrics.
		spanAttributes := make([]attribute.KeyValue, 0, len(attrs)+len(h.SpanAttributes))
		spanAttributes = append(append(spanAttributes, attrs...), h.SpanAttributes...)
		ctx, _ = h.tracer.Start(
			ctx,
			name,
			trace.WithSpanKind(h.SpanKind),
			trace.WithAttributes(spanAttributes...),
		)
	}

	gctx := gRPCContext{
		metricAttrs: append(attrs, h.MetricAttributes...),
		record:      record,
	}

	if h.MetricAttributesFn != nil {
		extraAttrs := h.MetricAttributesFn(ctx)
		gctx.metricAttrs = append(gctx.metricAttrs, extraAttrs...)
	}

	return inject(context.WithValue(ctx, gRPCContextKey{}, &gctx), h.Propagators)
}

// HandleRPC processes the RPC stats.
func (h *clientHandler) HandleRPC(ctx context.Context, rs stats.RPCStats) {
	var dur metric.Float64Histogram
	if h.semconvMode == semconvModeNew || h.semconvMode == semconvModeDup {
		dur = h.duration.Inst()
	}
	var oldDur metric.Float64Histogram
	if h.semconvMode == semconvModeOld || h.semconvMode == semconvModeDup {
		oldDur = h.oldDuration.Inst()
	}
	h.handleRPC(
		ctx,
		rs,
		dur,
		oldDur,
		func(s *status.Status) (codes.Code, string) {
			return codes.Error, s.Message()
		},
	)
}

// TagConn can attach some information to the given context.
func (*clientHandler) TagConn(ctx context.Context, _ *stats.ConnTagInfo) context.Context {
	return ctx
}

// HandleConn processes the Conn stats.
func (*clientHandler) HandleConn(context.Context, stats.ConnStats) {
	// no-op
}

func (*config) handleRPC(
	ctx context.Context,
	rs stats.RPCStats,
	duration metric.Float64Histogram,
	oldDuration metric.Float64Histogram,
	recordStatus func(*status.Status) (codes.Code, string),
) {
	gctx, _ := ctx.Value(gRPCContextKey{}).(*gRPCContext)
	if gctx != nil && !gctx.record {
		return
	}

	span := trace.SpanFromContext(ctx)

	switch rs := rs.(type) {
	case *stats.Begin:
	case *stats.InPayload:
	case *stats.InHeader:
		if !rs.Client && rs.LocalAddr != nil {
			if span.IsRecording() {
				span.SetAttributes(serverAddrAttrs(rs.LocalAddr.String())...)
			}
			// TODO: add server.address and server.port to metrics once the API supports opt-in attributes.
		}
	case *stats.OutPayload:
	case *stats.OutTrailer:
	case *stats.OutHeader:
		if rs.Client && rs.RemoteAddr != nil && (span.IsRecording() || gctx != nil) {
			attrs := serverAddrAttrs(rs.RemoteAddr.String())
			if span.IsRecording() {
				span.SetAttributes(attrs...)
			}
			if gctx != nil {
				gctx.metricAttrs = append(gctx.metricAttrs, attrs...)
			}
		}
	case *stats.End:
		var rpcStatusAttr attribute.KeyValue

		var s *status.Status
		if rs.Error != nil {
			s, _ = status.FromError(rs.Error)
			rpcStatusAttr = semconv.RPCResponseStatusCode(canonicalString(s.Code()))
		} else {
			rpcStatusAttr = semconv.RPCResponseStatusCode(canonicalString(grpc_codes.OK))
		}
		if span.IsRecording() {
			if s != nil {
				c, m := recordStatus(s)
				span.SetStatus(c, m)
			}
			span.SetAttributes(rpcStatusAttr)
			span.End()
		}

		var metricAttrs []attribute.KeyValue
		if gctx != nil {
			// Don't use gctx.metricAttrSet here, because it requires passing
			// multiple RecordOptions, which would call metric.mergeSets and
			// allocate a new set for each Record call.
			metricAttrs = make([]attribute.KeyValue, 0, len(gctx.metricAttrs)+1)
			metricAttrs = append(metricAttrs, gctx.metricAttrs...)
		}
		metricAttrs = append(metricAttrs, rpcStatusAttr)
		// Allocate vararg slice once.
		recordOpts := []metric.RecordOption{metric.WithAttributeSet(attribute.NewSet(metricAttrs...))}

		// Use floating point division here for higher precision (instead of Millisecond method).
		// Measure right before calling Record() to capture as much elapsed time as possible.
		elapsedTime := float64(rs.EndTime.Sub(rs.BeginTime)) / float64(time.Second)

		if duration != nil {
			duration.Record(ctx, elapsedTime, recordOpts...)
		}
		if oldDuration != nil {
			oldDuration.Record(ctx, elapsedTime*1000.0, recordOpts...)
		}

	default:
		return
	}
}

func canonicalString(code grpc_codes.Code) string {
	switch code {
	case grpc_codes.OK:
		return "OK"
	case grpc_codes.Canceled:
		return "CANCELLED"
	case grpc_codes.Unknown:
		return "UNKNOWN"
	case grpc_codes.InvalidArgument:
		return "INVALID_ARGUMENT"
	case grpc_codes.DeadlineExceeded:
		return "DEADLINE_EXCEEDED"
	case grpc_codes.NotFound:
		return "NOT_FOUND"
	case grpc_codes.AlreadyExists:
		return "ALREADY_EXISTS"
	case grpc_codes.PermissionDenied:
		return "PERMISSION_DENIED"
	case grpc_codes.ResourceExhausted:
		return "RESOURCE_EXHAUSTED"
	case grpc_codes.FailedPrecondition:
		return "FAILED_PRECONDITION"
	case grpc_codes.Aborted:
		return "ABORTED"
	case grpc_codes.OutOfRange:
		return "OUT_OF_RANGE"
	case grpc_codes.Unimplemented:
		return "UNIMPLEMENTED"
	case grpc_codes.Internal:
		return "INTERNAL"
	case grpc_codes.Unavailable:
		return "UNAVAILABLE"
	case grpc_codes.DataLoss:
		return "DATA_LOSS"
	case grpc_codes.Unauthenticated:
		return "UNAUTHENTICATED"
	default:
		return "CODE(" + strconv.FormatInt(int64(code), 10) + ")"
	}
}
