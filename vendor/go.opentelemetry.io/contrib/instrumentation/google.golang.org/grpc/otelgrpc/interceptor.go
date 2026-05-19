// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otelgrpc // import "go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"

// gRPC tracing middleware
// https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/semantic_conventions/rpc.md
import (
	"context"
	"errors"
	"io"
	"net"
	"strconv"
	"time"

	"google.golang.org/grpc"
	grpc_codes "google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc/internal"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
	"go.opentelemetry.io/otel/trace"
)

type messageType attribute.KeyValue

// Event adds an event of the messageType to the span associated with the
// passed context with a message id.
func (m messageType) Event(ctx context.Context, id int, _ interface{}) {
	span := trace.SpanFromContext(ctx)
	if !span.IsRecording() {
		return
	}
	span.AddEvent("message", trace.WithAttributes(
		attribute.KeyValue(m),
		RPCMessageIDKey.Int(id),
	))
}

var (
	messageSent     = messageType(RPCMessageTypeSent)
	messageReceived = messageType(RPCMessageTypeReceived)
)

// UnaryClientInterceptor returns a grpc.UnaryClientInterceptor suitable
// for use in a grpc.NewClient call.
//
// Deprecated: Use [NewClientHandler] instead.
func UnaryClientInterceptor(opts ...Option) grpc.UnaryClientInterceptor {
	cfg := newConfig(opts, "client")
	tracer := cfg.TracerProvider.Tracer(
		ScopeName,
		trace.WithInstrumentationVersion(Version()),
	)

	return func(
		ctx context.Context,
		method string,
		req, reply interface{},
		cc *grpc.ClientConn,
		invoker grpc.UnaryInvoker,
		callOpts ...grpc.CallOption,
	) error {
		i := &InterceptorInfo{
			Method: method,
			Type:   UnaryClient,
		}
		if cfg.InterceptorFilter != nil && !cfg.InterceptorFilter(i) {
			return invoker(ctx, method, req, reply, cc, callOpts...)
		}

		name, attr, _ := telemetryAttributes(method, cc.Target())

		startOpts := append([]trace.SpanStartOption{
			trace.WithSpanKind(trace.SpanKindClient),
			trace.WithAttributes(attr...),
		},
			cfg.SpanStartOptions...,
		)

		ctx, span := tracer.Start(
			ctx,
			name,
			startOpts...,
		)
		defer span.End()

		ctx = inject(ctx, cfg.Propagators)

		if cfg.SentEvent {
			messageSent.Event(ctx, 1, req)
		}

		err := invoker(ctx, method, req, reply, cc, callOpts...)

		if cfg.ReceivedEvent {
			messageReceived.Event(ctx, 1, reply)
		}

		if err != nil {
			s, _ := status.FromError(err)
			span.SetStatus(codes.Error, s.Message())
			span.SetAttributes(statusCodeAttr(s.Code()))
		} else {
			span.SetAttributes(statusCodeAttr(grpc_codes.OK))
		}

		return err
	}
}

// clientStream  wraps around the embedded grpc.ClientStream, and intercepts the RecvMsg and
// SendMsg method call.
type clientStream struct {
	grpc.ClientStream
	desc *grpc.StreamDesc

	span trace.Span

	receivedEvent bool
	sentEvent     bool

	receivedMessageID int
	sentMessageID     int
}

var _ = proto.Marshal

func (w *clientStream) RecvMsg(m interface{}) error {
	err := w.ClientStream.RecvMsg(m)

	if err == nil && !w.desc.ServerStreams {
		w.endSpan(nil)
	} else if errors.Is(err, io.EOF) {
		w.endSpan(nil)
	} else if err != nil {
		w.endSpan(err)
	} else {
		w.receivedMessageID++

		if w.receivedEvent {
			messageReceived.Event(w.Context(), w.receivedMessageID, m)
		}
	}

	return err
}

func (w *clientStream) SendMsg(m interface{}) error {
	err := w.ClientStream.SendMsg(m)

	w.sentMessageID++

	if w.sentEvent {
		messageSent.Event(w.Context(), w.sentMessageID, m)
	}

	if err != nil {
		w.endSpan(err)
	}

	return err
}

func (w *clientStream) Header() (metadata.MD, error) {
	md, err := w.ClientStream.Header()
	if err != nil {
		w.endSpan(err)
	}

	return md, err
}

func (w *clientStream) CloseSend() error {
	err := w.ClientStream.CloseSend()
	if err != nil {
		w.endSpan(err)
	}

	return err
}

func wrapClientStream(s grpc.ClientStream, desc *grpc.StreamDesc, span trace.Span, cfg *config) *clientStream {
	return &clientStream{
		ClientStream:  s,
		span:          span,
		desc:          desc,
		receivedEvent: cfg.ReceivedEvent,
		sentEvent:     cfg.SentEvent,
	}
}

func (w *clientStream) endSpan(err error) {
	if err != nil {
		s, _ := status.FromError(err)
		w.span.SetStatus(codes.Error, s.Message())
		w.span.SetAttributes(statusCodeAttr(s.Code()))
	} else {
		w.span.SetAttributes(statusCodeAttr(grpc_codes.OK))
	}

	w.span.End()
}

// StreamClientInterceptor returns a grpc.StreamClientInterceptor suitable
// for use in a grpc.NewClient call.
//
// Deprecated: Use [NewClientHandler] instead.
func StreamClientInterceptor(opts ...Option) grpc.StreamClientInterceptor {
	cfg := newConfig(opts, "client")
	tracer := cfg.TracerProvider.Tracer(
		ScopeName,
		trace.WithInstrumentationVersion(Version()),
	)

	return func(
		ctx context.Context,
		desc *grpc.StreamDesc,
		cc *grpc.ClientConn,
		method string,
		streamer grpc.Streamer,
		callOpts ...grpc.CallOption,
	) (grpc.ClientStream, error) {
		i := &InterceptorInfo{
			Method: method,
			Type:   StreamClient,
		}
		if cfg.InterceptorFilter != nil && !cfg.InterceptorFilter(i) {
			return streamer(ctx, desc, cc, method, callOpts...)
		}

		name, attr, _ := telemetryAttributes(method, cc.Target())

		startOpts := append([]trace.SpanStartOption{
			trace.WithSpanKind(trace.SpanKindClient),
			trace.WithAttributes(attr...),
		},
			cfg.SpanStartOptions...,
		)

		ctx, span := tracer.Start(
			ctx,
			name,
			startOpts...,
		)

		ctx = inject(ctx, cfg.Propagators)

		s, err := streamer(ctx, desc, cc, method, callOpts...)
		if err != nil {
			grpcStatus, _ := status.FromError(err)
			span.SetStatus(codes.Error, grpcStatus.Message())
			span.SetAttributes(statusCodeAttr(grpcStatus.Code()))
			span.End()
			return s, err
		}
		stream := wrapClientStream(s, desc, span, cfg)
		return stream, nil
	}
}

// UnaryServerInterceptor returns a grpc.UnaryServerInterceptor suitable
// for use in a grpc.NewServer call.
//
// Deprecated: Use [NewServerHandler] instead.
func UnaryServerInterceptor(opts ...Option) grpc.UnaryServerInterceptor {
	cfg := newConfig(opts, "server")
	tracer := cfg.TracerProvider.Tracer(
		ScopeName,
		trace.WithInstrumentationVersion(Version()),
	)

	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		i := &InterceptorInfo{
			UnaryServerInfo: info,
			Type:            UnaryServer,
		}
		if cfg.InterceptorFilter != nil && !cfg.InterceptorFilter(i) {
			return handler(ctx, req)
		}

		ctx = extract(ctx, cfg.Propagators)
		name, attr, metricAttrs := telemetryAttributes(info.FullMethod, peerFromCtx(ctx))

		startOpts := append([]trace.SpanStartOption{
			trace.WithSpanKind(trace.SpanKindServer),
			trace.WithAttributes(attr...),
		},
			cfg.SpanStartOptions...,
		)

		ctx, span := tracer.Start(
			trace.ContextWithRemoteSpanContext(ctx, trace.SpanContextFromContext(ctx)),
			name,
			startOpts...,
		)
		defer span.End()

		if cfg.ReceivedEvent {
			messageReceived.Event(ctx, 1, req)
		}

		before := time.Now()

		resp, err := handler(ctx, req)

		s, _ := status.FromError(err)
		if err != nil {
			statusCode, msg := serverStatus(s)
			span.SetStatus(statusCode, msg)
			if cfg.SentEvent {
				messageSent.Event(ctx, 1, s.Proto())
			}
		} else {
			if cfg.SentEvent {
				messageSent.Event(ctx, 1, resp)
			}
		}
		grpcStatusCodeAttr := statusCodeAttr(s.Code())
		span.SetAttributes(grpcStatusCodeAttr)

		// Use floating point division here for higher precision (instead of Millisecond method).
		elapsedTime := float64(time.Since(before)) / float64(time.Millisecond)

		metricAttrs = append(metricAttrs, grpcStatusCodeAttr)
		cfg.rpcDuration.Record(ctx, elapsedTime, metric.WithAttributeSet(attribute.NewSet(metricAttrs...)))

		return resp, err
	}
}

// serverStream wraps around the embedded grpc.ServerStream, and intercepts the RecvMsg and
// SendMsg method call.
type serverStream struct {
	grpc.ServerStream
	ctx context.Context

	receivedMessageID int
	sentMessageID     int

	receivedEvent bool
	sentEvent     bool
}

func (w *serverStream) Context() context.Context {
	return w.ctx
}

func (w *serverStream) RecvMsg(m interface{}) error {
	err := w.ServerStream.RecvMsg(m)

	if err == nil {
		w.receivedMessageID++
		if w.receivedEvent {
			messageReceived.Event(w.Context(), w.receivedMessageID, m)
		}
	}

	return err
}

func (w *serverStream) SendMsg(m interface{}) error {
	err := w.ServerStream.SendMsg(m)

	w.sentMessageID++
	if w.sentEvent {
		messageSent.Event(w.Context(), w.sentMessageID, m)
	}

	return err
}

func wrapServerStream(ctx context.Context, ss grpc.ServerStream, cfg *config) *serverStream {
	return &serverStream{
		ServerStream:  ss,
		ctx:           ctx,
		receivedEvent: cfg.ReceivedEvent,
		sentEvent:     cfg.SentEvent,
	}
}

// StreamServerInterceptor returns a grpc.StreamServerInterceptor suitable
// for use in a grpc.NewServer call.
//
// Deprecated: Use [NewServerHandler] instead.
func StreamServerInterceptor(opts ...Option) grpc.StreamServerInterceptor {
	cfg := newConfig(opts, "server")
	tracer := cfg.TracerProvider.Tracer(
		ScopeName,
		trace.WithInstrumentationVersion(Version()),
	)

	return func(
		srv interface{},
		ss grpc.ServerStream,
		info *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		ctx := ss.Context()
		i := &InterceptorInfo{
			StreamServerInfo: info,
			Type:             StreamServer,
		}
		if cfg.InterceptorFilter != nil && !cfg.InterceptorFilter(i) {
			return handler(srv, wrapServerStream(ctx, ss, cfg))
		}

		ctx = extract(ctx, cfg.Propagators)
		name, attr, _ := telemetryAttributes(info.FullMethod, peerFromCtx(ctx))

		startOpts := append([]trace.SpanStartOption{
			trace.WithSpanKind(trace.SpanKindServer),
			trace.WithAttributes(attr...),
		},
			cfg.SpanStartOptions...,
		)

		ctx, span := tracer.Start(
			trace.ContextWithRemoteSpanContext(ctx, trace.SpanContextFromContext(ctx)),
			name,
			startOpts...,
		)
		defer span.End()

		err := handler(srv, wrapServerStream(ctx, ss, cfg))
		if err != nil {
			s, _ := status.FromError(err)
			statusCode, msg := serverStatus(s)
			span.SetStatus(statusCode, msg)
			span.SetAttributes(statusCodeAttr(s.Code()))
		} else {
			span.SetAttributes(statusCodeAttr(grpc_codes.OK))
		}

		return err
	}
}

// telemetryAttributes returns a span name and span and metric attributes from
// the gRPC method and peer address.
func telemetryAttributes(fullMethod, peerAddress string) (string, []attribute.KeyValue, []attribute.KeyValue) {
	name, methodAttrs := internal.ParseFullMethod(fullMethod)
	peerAttrs := peerAttr(peerAddress)

	attrs := make([]attribute.KeyValue, 0, 1+len(methodAttrs)+len(peerAttrs))
	attrs = append(attrs, RPCSystemGRPC)
	attrs = append(attrs, methodAttrs...)
	metricAttrs := attrs[:1+len(methodAttrs)]
	attrs = append(attrs, peerAttrs...)
	return name, attrs, metricAttrs
}

// peerAttr returns attributes about the peer address.
func peerAttr(addr string) []attribute.KeyValue {
	host, p, err := net.SplitHostPort(addr)
	if err != nil {
		return nil
	}

	if host == "" {
		host = "127.0.0.1"
	}
	port, err := strconv.Atoi(p)
	if err != nil {
		return nil
	}

	var attr []attribute.KeyValue
	if ip := net.ParseIP(host); ip != nil {
		attr = []attribute.KeyValue{
			semconv.NetSockPeerAddr(host),
			semconv.NetSockPeerPort(port),
		}
	} else {
		attr = []attribute.KeyValue{
			semconv.NetPeerName(host),
			semconv.NetPeerPort(port),
		}
	}

	return attr
}

// peerFromCtx returns a peer address from a context, if one exists.
func peerFromCtx(ctx context.Context) string {
	p, ok := peer.FromContext(ctx)
	if !ok {
		return ""
	}
	return p.Addr.String()
}

// statusCodeAttr returns status code attribute based on given gRPC code.
func statusCodeAttr(c grpc_codes.Code) attribute.KeyValue {
	return GRPCStatusCodeKey.Int64(int64(c))
}

// serverStatus returns a span status code and message for a given gRPC
// status code. It maps specific gRPC status codes to a corresponding span
// status code and message. This function is intended for use on the server
// side of a gRPC connection.
//
// If the gRPC status code is Unknown, DeadlineExceeded, Unimplemented,
// Internal, Unavailable, or DataLoss, it returns a span status code of Error
// and the message from the gRPC status. Otherwise, it returns a span status
// code of Unset and an empty message.
func serverStatus(grpcStatus *status.Status) (codes.Code, string) {
	switch grpcStatus.Code() {
	case grpc_codes.Unknown,
		grpc_codes.DeadlineExceeded,
		grpc_codes.Unimplemented,
		grpc_codes.Internal,
		grpc_codes.Unavailable,
		grpc_codes.DataLoss:
		return codes.Error, grpcStatus.Message()
	default:
		return codes.Unset, ""
	}
}
