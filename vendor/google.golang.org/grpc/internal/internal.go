/*
 * Copyright 2016 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package internal contains gRPC-internal code, to avoid polluting
// the godoc of the top-level grpc package.  It must not import any grpc
// symbols to avoid circular dependencies.
package internal

var (

	// TestingUseHandlerImpl enables the http.Handler-based server implementation.
	// It must be called before Serve and requires TLS credentials.
	//
	// The provided grpcServer must be of type *grpc.Server. It is untyped
	// for circular dependency reasons.
	TestingUseHandlerImpl func(grpcServer interface{})

	// WithContextDialer is exported by clientconn.go
	WithContextDialer interface{} // func(context.Context, string) (net.Conn, error) grpc.DialOption
	// WithResolverBuilder is exported by clientconn.go
	WithResolverBuilder interface{} // func (resolver.Builder) grpc.DialOption
)
