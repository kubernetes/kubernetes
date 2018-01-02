package flexvolume

import (
	"github.com/golang/protobuf/ptypes/empty"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

type localAPIClient struct {
	apiServer APIServer
}

func newLocalAPIClient(apiServer APIServer) *localAPIClient {
	return &localAPIClient{apiServer}
}

func (l *localAPIClient) Init(ctx context.Context, request *empty.Empty, _ ...grpc.CallOption) (*empty.Empty, error) {
	return l.apiServer.Init(ctx, request)
}

func (l *localAPIClient) Attach(ctx context.Context, request *AttachRequest, _ ...grpc.CallOption) (*empty.Empty, error) {
	return l.apiServer.Attach(ctx, request)
}

func (l *localAPIClient) Detach(ctx context.Context, request *DetachRequest, _ ...grpc.CallOption) (*empty.Empty, error) {
	return l.apiServer.Detach(ctx, request)
}

func (l *localAPIClient) Mount(ctx context.Context, request *MountRequest, _ ...grpc.CallOption) (*empty.Empty, error) {
	return l.apiServer.Mount(ctx, request)
}

func (l *localAPIClient) Unmount(ctx context.Context, request *UnmountRequest, _ ...grpc.CallOption) (*empty.Empty, error) {
	return l.apiServer.Unmount(ctx, request)
}
