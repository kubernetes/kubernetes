// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
