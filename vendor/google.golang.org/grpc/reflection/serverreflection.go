/*
 *
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

/*
Package reflection implements server reflection service.

The service implemented is defined in:
https://github.com/grpc/grpc/blob/master/src/proto/grpc/reflection/v1alpha/reflection.proto.

To register server reflection on a gRPC server:

	import "google.golang.org/grpc/reflection"

	s := grpc.NewServer()
	pb.RegisterYourOwnServer(s, &server{})

	// Register reflection service on gRPC server.
	reflection.Register(s)

	s.Serve(lis)
*/
package reflection // import "google.golang.org/grpc/reflection"

import (
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection/internal"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"

	v1reflectiongrpc "google.golang.org/grpc/reflection/grpc_reflection_v1"
	v1alphareflectiongrpc "google.golang.org/grpc/reflection/grpc_reflection_v1alpha"
)

// GRPCServer is the interface provided by a gRPC server. It is implemented by
// *grpc.Server, but could also be implemented by other concrete types. It acts
// as a registry, for accumulating the services exposed by the server.
type GRPCServer interface {
	grpc.ServiceRegistrar
	ServiceInfoProvider
}

var _ GRPCServer = (*grpc.Server)(nil)

// Register registers the server reflection service on the given gRPC server.
// Both the v1 and v1alpha versions are registered.
func Register(s GRPCServer) {
	svr := NewServerV1(ServerOptions{Services: s})
	v1alphareflectiongrpc.RegisterServerReflectionServer(s, asV1Alpha(svr))
	v1reflectiongrpc.RegisterServerReflectionServer(s, svr)
}

// RegisterV1 registers only the v1 version of the server reflection service
// on the given gRPC server. Many clients may only support v1alpha so most
// users should use Register instead, at least until clients have upgraded.
func RegisterV1(s GRPCServer) {
	svr := NewServerV1(ServerOptions{Services: s})
	v1reflectiongrpc.RegisterServerReflectionServer(s, svr)
}

// ServiceInfoProvider is an interface used to retrieve metadata about the
// services to expose.
//
// The reflection service is only interested in the service names, but the
// signature is this way so that *grpc.Server implements it. So it is okay
// for a custom implementation to return zero values for the
// grpc.ServiceInfo values in the map.
//
// # Experimental
//
// Notice: This type is EXPERIMENTAL and may be changed or removed in a
// later release.
type ServiceInfoProvider interface {
	GetServiceInfo() map[string]grpc.ServiceInfo
}

// ExtensionResolver is the interface used to query details about extensions.
// This interface is satisfied by protoregistry.GlobalTypes.
//
// # Experimental
//
// Notice: This type is EXPERIMENTAL and may be changed or removed in a
// later release.
type ExtensionResolver interface {
	protoregistry.ExtensionTypeResolver
	RangeExtensionsByMessage(message protoreflect.FullName, f func(protoreflect.ExtensionType) bool)
}

// ServerOptions represents the options used to construct a reflection server.
//
// # Experimental
//
// Notice: This type is EXPERIMENTAL and may be changed or removed in a
// later release.
type ServerOptions struct {
	// The source of advertised RPC services. If not specified, the reflection
	// server will report an empty list when asked to list services.
	//
	// This value will typically be a *grpc.Server. But the set of advertised
	// services can be customized by wrapping a *grpc.Server or using an
	// alternate implementation that returns a custom set of service names.
	Services ServiceInfoProvider
	// Optional resolver used to load descriptors. If not specified,
	// protoregistry.GlobalFiles will be used.
	DescriptorResolver protodesc.Resolver
	// Optional resolver used to query for known extensions. If not specified,
	// protoregistry.GlobalTypes will be used.
	ExtensionResolver ExtensionResolver
}

// NewServer returns a reflection server implementation using the given options.
// This can be used to customize behavior of the reflection service. Most usages
// should prefer to use Register instead. For backwards compatibility reasons,
// this returns the v1alpha version of the reflection server. For a v1 version
// of the reflection server, see NewServerV1.
//
// # Experimental
//
// Notice: This function is EXPERIMENTAL and may be changed or removed in a
// later release.
func NewServer(opts ServerOptions) v1alphareflectiongrpc.ServerReflectionServer {
	return asV1Alpha(NewServerV1(opts))
}

// NewServerV1 returns a reflection server implementation using the given options.
// This can be used to customize behavior of the reflection service. Most usages
// should prefer to use Register instead.
//
// # Experimental
//
// Notice: This function is EXPERIMENTAL and may be changed or removed in a
// later release.
func NewServerV1(opts ServerOptions) v1reflectiongrpc.ServerReflectionServer {
	if opts.DescriptorResolver == nil {
		opts.DescriptorResolver = protoregistry.GlobalFiles
	}
	if opts.ExtensionResolver == nil {
		opts.ExtensionResolver = protoregistry.GlobalTypes
	}
	return &internal.ServerReflectionServer{
		S:            opts.Services,
		DescResolver: opts.DescriptorResolver,
		ExtResolver:  opts.ExtensionResolver,
	}
}
