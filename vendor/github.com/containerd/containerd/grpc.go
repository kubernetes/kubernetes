package containerd

import (
	"github.com/containerd/containerd/namespaces"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

type namespaceInterceptor struct {
	namespace string
}

func (ni namespaceInterceptor) unary(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
	_, ok := namespaces.Namespace(ctx)
	if !ok {
		ctx = namespaces.WithNamespace(ctx, ni.namespace)
	}
	return invoker(ctx, method, req, reply, cc, opts...)
}

func (ni namespaceInterceptor) stream(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
	_, ok := namespaces.Namespace(ctx)
	if !ok {
		ctx = namespaces.WithNamespace(ctx, ni.namespace)
	}

	return streamer(ctx, desc, cc, method, opts...)
}

func newNSInterceptors(ns string) (grpc.UnaryClientInterceptor, grpc.StreamClientInterceptor) {
	ni := namespaceInterceptor{
		namespace: ns,
	}
	return grpc.UnaryClientInterceptor(ni.unary), grpc.StreamClientInterceptor(ni.stream)
}
