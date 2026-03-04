// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otelgrpc // import "go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"

import (
	"context"
	"sync/atomic"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	semconv "go.opentelemetry.io/otel/semconv/v1.39.0"
	"go.opentelemetry.io/otel/semconv/v1.39.0/rpcconv"
	"go.opentelemetry.io/otel/trace"
	grpc_codes "google.golang.org/grpc/codes"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"

	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc/internal"
)

type gRPCContextKey struct{}

type gRPCContext struct {
	inMessages    int64
	outMessages   int64
	metricAttrs   []attribute.KeyValue
	metricAttrSet attribute.Set
	record        bool
}

type serverHandler struct {
	*config

	tracer trace.Tracer

	duration rpcconv.ServerCallDuration
	inSize   int64Hist
	outSize  int64Hist
}

// NewServerHandler creates a stats.Handler for a gRPC server.
func NewServerHandler(opts ...Option) stats.Handler {
	c := newConfig(opts)
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
	h.duration, err = rpcconv.NewServerCallDuration(meter)
	if err != nil {
		otel.Handle(err)
	}

	h.inSize, err = rpcconv.NewServerRequestSize(meter)
	if err != nil {
		otel.Handle(err)
	}

	h.outSize, err = rpcconv.NewServerResponseSize(meter)
	if err != nil {
		otel.Handle(err)
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

	name, attrs := internal.ParseFullMethod(info.FullMethodName)
	attrs = append(attrs, semconv.RPCSystemNameGRPC)

	record := true
	if h.Filter != nil {
		record = h.Filter(info)
	}

	if record {
		// Make a new slice to avoid aliasing into the same attrs slice used by metrics.
		spanAttributes := make([]attribute.KeyValue, 0, len(attrs)+len(h.SpanAttributes))
		spanAttributes = append(append(spanAttributes, attrs...), h.SpanAttributes...)
		opts := []trace.SpanStartOption{
			trace.WithSpanKind(trace.SpanKindServer),
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

	gctx.metricAttrSet = attribute.NewSet(gctx.metricAttrs...)

	return context.WithValue(ctx, gRPCContextKey{}, &gctx)
}

// HandleRPC processes the RPC stats.
func (h *serverHandler) HandleRPC(ctx context.Context, rs stats.RPCStats) {
	h.handleRPC(
		ctx,
		rs,
		h.duration.Inst(),
		h.inSize,
		h.outSize,
		serverStatus,
	)
}

type clientHandler struct {
	*config

	tracer trace.Tracer

	duration rpcconv.ClientCallDuration
	inSize   int64Hist
	outSize  int64Hist
}

// NewClientHandler creates a stats.Handler for a gRPC client.
func NewClientHandler(opts ...Option) stats.Handler {
	c := newConfig(opts)
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
	h.duration, err = rpcconv.NewClientCallDuration(meter)
	if err != nil {
		otel.Handle(err)
	}

	h.inSize, err = rpcconv.NewClientResponseSize(meter)
	if err != nil {
		otel.Handle(err)
	}

	h.outSize, err = rpcconv.NewClientRequestSize(meter)
	if err != nil {
		otel.Handle(err)
	}

	return h
}

// TagRPC can attach some information to the given context.
func (h *clientHandler) TagRPC(ctx context.Context, info *stats.RPCTagInfo) context.Context {
	name, attrs := internal.ParseFullMethod(info.FullMethodName)
	attrs = append(attrs, semconv.RPCSystemNameGRPC)

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
			trace.WithSpanKind(trace.SpanKindClient),
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

	gctx.metricAttrSet = attribute.NewSet(gctx.metricAttrs...)

	return inject(context.WithValue(ctx, gRPCContextKey{}, &gctx), h.Propagators)
}

// HandleRPC processes the RPC stats.
func (h *clientHandler) HandleRPC(ctx context.Context, rs stats.RPCStats) {
	h.handleRPC(
		ctx,
		rs,
		h.duration.Inst(),
		h.inSize,
		h.outSize,
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

type int64Hist interface {
	RecordSet(context.Context, int64, attribute.Set)
}

func (c *config) handleRPC(
	ctx context.Context,
	rs stats.RPCStats,
	duration metric.Float64Histogram,
	inSize, outSize int64Hist,
	recordStatus func(*status.Status) (codes.Code, string),
) {
	gctx, _ := ctx.Value(gRPCContextKey{}).(*gRPCContext)
	if gctx != nil && !gctx.record {
		return
	}

	span := trace.SpanFromContext(ctx)
	var messageId int64

	switch rs := rs.(type) {
	case *stats.Begin:
	case *stats.InPayload:
		if gctx != nil {
			messageId = atomic.AddInt64(&gctx.inMessages, 1)
			inSize.RecordSet(ctx, int64(rs.Length), gctx.metricAttrSet)
		}

		if c.ReceivedEvent && span.IsRecording() {
			span.AddEvent("message",
				trace.WithAttributes(
					semconv.RPCMessageTypeReceived,
					semconv.RPCMessageIDKey.Int64(messageId),
					semconv.RPCMessageCompressedSizeKey.Int(rs.CompressedLength),
					semconv.RPCMessageUncompressedSizeKey.Int(rs.Length),
				),
			)
		}
	case *stats.OutPayload:
		if gctx != nil {
			messageId = atomic.AddInt64(&gctx.outMessages, 1)
			outSize.RecordSet(ctx, int64(rs.Length), gctx.metricAttrSet)
		}

		if c.SentEvent && span.IsRecording() {
			span.AddEvent("message",
				trace.WithAttributes(
					semconv.RPCMessageTypeSent,
					semconv.RPCMessageIDKey.Int64(messageId),
					semconv.RPCMessageCompressedSizeKey.Int(rs.CompressedLength),
					semconv.RPCMessageUncompressedSizeKey.Int(rs.Length),
				),
			)
		}
	case *stats.OutTrailer:
	case *stats.OutHeader:
		if span.IsRecording() {
			if p, ok := peer.FromContext(ctx); ok {
				span.SetAttributes(serverAddrAttrs(p.Addr.String())...)
			}
		}
	case *stats.End:
		var rpcStatusAttr attribute.KeyValue

		var s *status.Status
		if rs.Error != nil {
			s, _ = status.FromError(rs.Error)
			rpcStatusAttr = semconv.RPCResponseStatusCode(s.Code().String())
		} else {
			rpcStatusAttr = semconv.RPCResponseStatusCode(grpc_codes.OK.String())
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
		elapsedTime := float64(rs.EndTime.Sub(rs.BeginTime)) / float64(time.Millisecond)

		duration.Record(ctx, elapsedTime, recordOpts...)
	default:
		return
	}
}
