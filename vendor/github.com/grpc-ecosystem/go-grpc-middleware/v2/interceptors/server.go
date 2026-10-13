// Copyright (c) The go-grpc-middleware Authors.
// Licensed under the Apache License 2.0.

// Go gRPC Middleware monitoring interceptors for server-side gRPC.

package interceptors

import (
	"context"
	"time"

	"google.golang.org/grpc"
)

// UnaryServerInterceptor is a gRPC server-side interceptor that provides reporting for Unary RPCs.
func UnaryServerInterceptor(reportable ServerReportable) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		r := newReport(NewServerCallMeta(info.FullMethod, nil, req))
		reporter, newCtx := reportable.ServerReporter(ctx, r.callMeta)

		reporter.PostMsgReceive(req, nil, time.Since(r.startTime))
		resp, err := handler(newCtx, req)
		reporter.PostMsgSend(resp, err, time.Since(r.startTime))

		reporter.PostCall(err, time.Since(r.startTime))
		return resp, err
	}
}

// StreamServerInterceptor is a gRPC server-side interceptor that provides reporting for Streaming RPCs.
func StreamServerInterceptor(reportable ServerReportable) grpc.StreamServerInterceptor {
	return func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		r := newReport(NewServerCallMeta(info.FullMethod, info, nil))
		reporter, newCtx := reportable.ServerReporter(ss.Context(), r.callMeta)
		err := handler(srv, &monitoredServerStream{ServerStream: ss, newCtx: newCtx, reporter: reporter})
		reporter.PostCall(err, time.Since(r.startTime))
		return err
	}
}

// monitoredStream wraps grpc.ServerStream allowing each Sent/Recv of message to report.
type monitoredServerStream struct {
	grpc.ServerStream

	newCtx   context.Context
	reporter Reporter
}

func (s *monitoredServerStream) Context() context.Context {
	return s.newCtx
}

func (s *monitoredServerStream) SendMsg(m any) error {
	start := time.Now()
	err := s.ServerStream.SendMsg(m)
	s.reporter.PostMsgSend(m, err, time.Since(start))
	return err
}

func (s *monitoredServerStream) RecvMsg(m any) error {
	start := time.Now()
	err := s.ServerStream.RecvMsg(m)
	s.reporter.PostMsgReceive(m, err, time.Since(start))
	return err
}
