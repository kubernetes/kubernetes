// Copyright 2016 Michal Witkowski. All Rights Reserved.
// See LICENSE for licensing terms.

package grpc_prometheus

import (
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

type grpcType string

const (
	Unary        grpcType = "unary"
	ClientStream grpcType = "client_stream"
	ServerStream grpcType = "server_stream"
	BidiStream   grpcType = "bidi_stream"
)

var (
	allCodes = []codes.Code{
		codes.OK, codes.Canceled, codes.Unknown, codes.InvalidArgument, codes.DeadlineExceeded, codes.NotFound,
		codes.AlreadyExists, codes.PermissionDenied, codes.Unauthenticated, codes.ResourceExhausted,
		codes.FailedPrecondition, codes.Aborted, codes.OutOfRange, codes.Unimplemented, codes.Internal,
		codes.Unavailable, codes.DataLoss,
	}
)

func splitMethodName(fullMethodName string) (string, string) {
	fullMethodName = strings.TrimPrefix(fullMethodName, "/") // remove leading slash
	if i := strings.Index(fullMethodName, "/"); i >= 0 {
		return fullMethodName[:i], fullMethodName[i+1:]
	}
	return "unknown", "unknown"
}

func typeFromMethodInfo(mInfo *grpc.MethodInfo) grpcType {
	if !mInfo.IsClientStream && !mInfo.IsServerStream {
		return Unary
	}
	if mInfo.IsClientStream && !mInfo.IsServerStream {
		return ClientStream
	}
	if !mInfo.IsClientStream && mInfo.IsServerStream {
		return ServerStream
	}
	return BidiStream
}
