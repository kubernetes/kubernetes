/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	context "context"

	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.64.0 or later.
const _ = grpc.SupportPackageIsVersion9

const (
	NodeHealth_WatchResources_FullMethodName = "/v1alpha1.NodeHealth/WatchResources"
)

// NodeHealthClient is the client API for NodeHealth service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type NodeHealthClient interface {
	WatchResources(ctx context.Context, in *WatchResourcesRequest, opts ...grpc.CallOption) (grpc.ServerStreamingClient[WatchResourcesResponse], error)
}

type nodeHealthClient struct {
	cc grpc.ClientConnInterface
}

func NewNodeHealthClient(cc grpc.ClientConnInterface) NodeHealthClient {
	return &nodeHealthClient{cc}
}

func (c *nodeHealthClient) WatchResources(ctx context.Context, in *WatchResourcesRequest, opts ...grpc.CallOption) (grpc.ServerStreamingClient[WatchResourcesResponse], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &NodeHealth_ServiceDesc.Streams[0], NodeHealth_WatchResources_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[WatchResourcesRequest, WatchResourcesResponse]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type NodeHealth_WatchResourcesClient = grpc.ServerStreamingClient[WatchResourcesResponse]

// NodeHealthServer is the server API for NodeHealth service.
// All implementations must embed UnimplementedNodeHealthServer
// for forward compatibility.
type NodeHealthServer interface {
	WatchResources(*WatchResourcesRequest, grpc.ServerStreamingServer[WatchResourcesResponse]) error
	mustEmbedUnimplementedNodeHealthServer()
}

// UnimplementedNodeHealthServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedNodeHealthServer struct{}

func (UnimplementedNodeHealthServer) WatchResources(*WatchResourcesRequest, grpc.ServerStreamingServer[WatchResourcesResponse]) error {
	return status.Errorf(codes.Unimplemented, "method WatchResources not implemented")
}
func (UnimplementedNodeHealthServer) mustEmbedUnimplementedNodeHealthServer() {}
func (UnimplementedNodeHealthServer) testEmbeddedByValue()                    {}

// UnsafeNodeHealthServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to NodeHealthServer will
// result in compilation errors.
type UnsafeNodeHealthServer interface {
	mustEmbedUnimplementedNodeHealthServer()
}

func RegisterNodeHealthServer(s grpc.ServiceRegistrar, srv NodeHealthServer) {
	// If the following call pancis, it indicates UnimplementedNodeHealthServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&NodeHealth_ServiceDesc, srv)
}

func _NodeHealth_WatchResources_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(WatchResourcesRequest)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(NodeHealthServer).WatchResources(m, &grpc.GenericServerStream[WatchResourcesRequest, WatchResourcesResponse]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type NodeHealth_WatchResourcesServer = grpc.ServerStreamingServer[WatchResourcesResponse]

// NodeHealth_ServiceDesc is the grpc.ServiceDesc for NodeHealth service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var NodeHealth_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "v1alpha1.NodeHealth",
	HandlerType: (*NodeHealthServer)(nil),
	Methods:     []grpc.MethodDesc{},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "WatchResources",
			Handler:       _NodeHealth_WatchResources_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "staging/src/k8s.io/kubelet/pkg/apis/dra-health/v1alpha1/api.proto",
}
