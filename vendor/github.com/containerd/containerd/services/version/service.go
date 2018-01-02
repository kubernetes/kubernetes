package version

import (
	api "github.com/containerd/containerd/api/services/version/v1"
	"github.com/containerd/containerd/plugin"
	ctrdversion "github.com/containerd/containerd/version"
	empty "github.com/golang/protobuf/ptypes/empty"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

var _ api.VersionServer = &service{}

func init() {
	plugin.Register(&plugin.Registration{
		Type:   plugin.GRPCPlugin,
		ID:     "version",
		InitFn: initFunc,
	})
}

func initFunc(ic *plugin.InitContext) (interface{}, error) {
	return &service{}, nil
}

type service struct {
}

func (s *service) Register(server *grpc.Server) error {
	api.RegisterVersionServer(server, s)
	return nil
}

func (s *service) Version(ctx context.Context, _ *empty.Empty) (*api.VersionResponse, error) {
	return &api.VersionResponse{
		Version:  ctrdversion.Version,
		Revision: ctrdversion.Revision,
	}, nil
}
