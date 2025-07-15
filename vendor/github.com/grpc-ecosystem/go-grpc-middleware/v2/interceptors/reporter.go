// Copyright (c) The go-grpc-middleware Authors.
// Licensed under the Apache License 2.0.

package interceptors

import (
	"context"
	"time"

	"google.golang.org/grpc/codes"
)

type GRPCType string

const (
	Unary        GRPCType = "unary"
	ClientStream GRPCType = "client_stream"
	ServerStream GRPCType = "server_stream"
	BidiStream   GRPCType = "bidi_stream"
)

var (
	AllCodes = []codes.Code{
		codes.OK, codes.Canceled, codes.Unknown, codes.InvalidArgument, codes.DeadlineExceeded, codes.NotFound,
		codes.AlreadyExists, codes.PermissionDenied, codes.Unauthenticated, codes.ResourceExhausted,
		codes.FailedPrecondition, codes.Aborted, codes.OutOfRange, codes.Unimplemented, codes.Internal,
		codes.Unavailable, codes.DataLoss,
	}
)

type ClientReportable interface {
	ClientReporter(context.Context, CallMeta) (Reporter, context.Context)
}

type ServerReportable interface {
	ServerReporter(context.Context, CallMeta) (Reporter, context.Context)
}

// CommonReportableFunc helper allows an easy way to implement reporter with common client and server logic.
type CommonReportableFunc func(ctx context.Context, c CallMeta) (Reporter, context.Context)

func (f CommonReportableFunc) ClientReporter(ctx context.Context, c CallMeta) (Reporter, context.Context) {
	return f(ctx, c)
}

func (f CommonReportableFunc) ServerReporter(ctx context.Context, c CallMeta) (Reporter, context.Context) {
	return f(ctx, c)
}

type Reporter interface {
	PostCall(err error, rpcDuration time.Duration)
	PostMsgSend(reqProto any, err error, sendDuration time.Duration)
	PostMsgReceive(replyProto any, err error, recvDuration time.Duration)
}

var _ Reporter = NoopReporter{}

type NoopReporter struct{}

func (NoopReporter) PostCall(error, time.Duration)            {}
func (NoopReporter) PostMsgSend(any, error, time.Duration)    {}
func (NoopReporter) PostMsgReceive(any, error, time.Duration) {}

type report struct {
	callMeta  CallMeta
	startTime time.Time
}

func newReport(callMeta CallMeta) report {
	r := report{
		startTime: time.Now(),
		callMeta:  callMeta,
	}
	return r
}
