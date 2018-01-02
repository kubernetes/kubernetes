// This file contains the generated server side code.
// It's only used for grpclb testing.

package grpclb

import (
	"google.golang.org/grpc"
	lbpb "google.golang.org/grpc/grpclb/grpc_lb_v1"
)

// Server API for LoadBalancer service

type loadBalancerServer interface {
	// Bidirectional rpc to get a list of servers.
	BalanceLoad(*loadBalancerBalanceLoadServer) error
}

func registerLoadBalancerServer(s *grpc.Server, srv loadBalancerServer) {
	s.RegisterService(
		&grpc.ServiceDesc{
			ServiceName: "grpc.lb.v1.LoadBalancer",
			HandlerType: (*loadBalancerServer)(nil),
			Methods:     []grpc.MethodDesc{},
			Streams: []grpc.StreamDesc{
				{
					StreamName:    "BalanceLoad",
					Handler:       balanceLoadHandler,
					ServerStreams: true,
					ClientStreams: true,
				},
			},
			Metadata: "grpclb.proto",
		}, srv)
}

func balanceLoadHandler(srv interface{}, stream grpc.ServerStream) error {
	return srv.(loadBalancerServer).BalanceLoad(&loadBalancerBalanceLoadServer{stream})
}

type loadBalancerBalanceLoadServer struct {
	grpc.ServerStream
}

func (x *loadBalancerBalanceLoadServer) Send(m *lbpb.LoadBalanceResponse) error {
	return x.ServerStream.SendMsg(m)
}

func (x *loadBalancerBalanceLoadServer) Recv() (*lbpb.LoadBalanceRequest, error) {
	m := new(lbpb.LoadBalanceRequest)
	if err := x.ServerStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}
