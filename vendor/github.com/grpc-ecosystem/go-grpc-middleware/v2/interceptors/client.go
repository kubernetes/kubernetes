// Copyright (c) The go-grpc-middleware Authors.
// Licensed under the Apache License 2.0.

// Go gRPC Middleware monitoring interceptors for client-side gRPC.

package interceptors

import (
	"context"
	"io"
	"time"

	"google.golang.org/grpc"
)

// UnaryClientInterceptor is a gRPC client-side interceptor that provides reporting for Unary RPCs.
func UnaryClientInterceptor(reportable ClientReportable) grpc.UnaryClientInterceptor {
	return func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		r := newReport(NewClientCallMeta(method, nil, req))
		reporter, newCtx := reportable.ClientReporter(ctx, r.callMeta)

		reporter.PostMsgSend(req, nil, time.Since(r.startTime))
		err := invoker(newCtx, method, req, reply, cc, opts...)
		reporter.PostMsgReceive(reply, err, time.Since(r.startTime))
		reporter.PostCall(err, time.Since(r.startTime))
		return err
	}
}

// StreamClientInterceptor is a gRPC client-side interceptor that provides reporting for Stream RPCs.
func StreamClientInterceptor(reportable ClientReportable) grpc.StreamClientInterceptor {
	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		r := newReport(NewClientCallMeta(method, desc, nil))
		reporter, newCtx := reportable.ClientReporter(ctx, r.callMeta)

		clientStream, err := streamer(newCtx, desc, cc, method, opts...)
		if err != nil {
			reporter.PostCall(err, time.Since(r.startTime))
			return nil, err
		}
		return &monitoredClientStream{ClientStream: clientStream, startTime: r.startTime, hasServerStream: desc.ServerStreams, reporter: reporter}, nil
	}
}

// monitoredClientStream wraps grpc.ClientStream allowing each Sent/Recv of message to report.
type monitoredClientStream struct {
	grpc.ClientStream

	startTime       time.Time
	hasServerStream bool
	reporter        Reporter
}

func (s *monitoredClientStream) SendMsg(m any) error {
	start := time.Now()
	err := s.ClientStream.SendMsg(m)
	s.reporter.PostMsgSend(m, err, time.Since(start))
	return err
}

func (s *monitoredClientStream) RecvMsg(m any) error {
	start := time.Now()
	err := s.ClientStream.RecvMsg(m)
	s.reporter.PostMsgReceive(m, err, time.Since(start))

	if s.hasServerStream {
		if err == nil {
			return nil
		}
		var postErr error
		if err != io.EOF {
			postErr = err
		}
		s.reporter.PostCall(postErr, time.Since(s.startTime))
	} else {
		s.reporter.PostCall(err, time.Since(s.startTime))
	}
	return err
}
