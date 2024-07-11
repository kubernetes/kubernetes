// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otelgrpc // import "go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"

import (
	"google.golang.org/grpc"
)

// InterceptorType is the flag to define which gRPC interceptor
// the InterceptorInfo object is.
type InterceptorType uint8

const (
	// UndefinedInterceptor is the type for the interceptor information that is not
	// well initialized or categorized to other types.
	UndefinedInterceptor InterceptorType = iota
	// UnaryClient is the type for grpc.UnaryClient interceptor.
	UnaryClient
	// StreamClient is the type for grpc.StreamClient interceptor.
	StreamClient
	// UnaryServer is the type for grpc.UnaryServer interceptor.
	UnaryServer
	// StreamServer is the type for grpc.StreamServer interceptor.
	StreamServer
)

// InterceptorInfo is the union of some arguments to four types of
// gRPC interceptors.
type InterceptorInfo struct {
	// Method is method name registered to UnaryClient and StreamClient
	Method string
	// UnaryServerInfo is the metadata for UnaryServer
	UnaryServerInfo *grpc.UnaryServerInfo
	// StreamServerInfo if the metadata for StreamServer
	StreamServerInfo *grpc.StreamServerInfo
	// Type is the type for interceptor
	Type InterceptorType
}
