package containers

import (
	"github.com/boltdb/bolt"
	api "github.com/containerd/containerd/api/services/containers/v1"
	eventsapi "github.com/containerd/containerd/api/services/events/v1"
	"github.com/containerd/containerd/containers"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/events"
	"github.com/containerd/containerd/metadata"
	"github.com/containerd/containerd/plugin"
	"github.com/golang/protobuf/ptypes/empty"
	"github.com/pkg/errors"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func init() {
	plugin.Register(&plugin.Registration{
		Type: plugin.GRPCPlugin,
		ID:   "containers",
		Requires: []plugin.Type{
			plugin.MetadataPlugin,
		},
		InitFn: func(ic *plugin.InitContext) (interface{}, error) {
			m, err := ic.Get(plugin.MetadataPlugin)
			if err != nil {
				return nil, err
			}
			return NewService(m.(*metadata.DB), ic.Events), nil
		},
	})
}

type service struct {
	db        *metadata.DB
	publisher events.Publisher
}

// NewService returns the container GRPC server
func NewService(db *metadata.DB, publisher events.Publisher) api.ContainersServer {
	return &service{db: db, publisher: publisher}
}

func (s *service) Register(server *grpc.Server) error {
	api.RegisterContainersServer(server, s)
	return nil
}

func (s *service) Get(ctx context.Context, req *api.GetContainerRequest) (*api.GetContainerResponse, error) {
	var resp api.GetContainerResponse

	return &resp, errdefs.ToGRPC(s.withStoreView(ctx, func(ctx context.Context, store containers.Store) error {
		container, err := store.Get(ctx, req.ID)
		if err != nil {
			return err
		}
		containerpb := containerToProto(&container)
		resp.Container = containerpb

		return nil
	}))
}

func (s *service) List(ctx context.Context, req *api.ListContainersRequest) (*api.ListContainersResponse, error) {
	var resp api.ListContainersResponse

	return &resp, errdefs.ToGRPC(s.withStoreView(ctx, func(ctx context.Context, store containers.Store) error {
		containers, err := store.List(ctx, req.Filters...)
		if err != nil {
			return err
		}

		resp.Containers = containersToProto(containers)
		return nil
	}))
}

func (s *service) Create(ctx context.Context, req *api.CreateContainerRequest) (*api.CreateContainerResponse, error) {
	var resp api.CreateContainerResponse

	if err := s.withStoreUpdate(ctx, func(ctx context.Context, store containers.Store) error {
		container := containerFromProto(&req.Container)

		created, err := store.Create(ctx, container)
		if err != nil {
			return err
		}

		resp.Container = containerToProto(&created)

		return nil
	}); err != nil {
		return &resp, errdefs.ToGRPC(err)
	}
	if err := s.publisher.Publish(ctx, "/containers/create", &eventsapi.ContainerCreate{
		ID:    resp.Container.ID,
		Image: resp.Container.Image,
		Runtime: &eventsapi.ContainerCreate_Runtime{
			Name:    resp.Container.Runtime.Name,
			Options: resp.Container.Runtime.Options,
		},
	}); err != nil {
		return &resp, err
	}

	return &resp, nil
}

func (s *service) Update(ctx context.Context, req *api.UpdateContainerRequest) (*api.UpdateContainerResponse, error) {
	if req.Container.ID == "" {
		return nil, status.Errorf(codes.InvalidArgument, "Container.ID required")
	}
	var (
		resp      api.UpdateContainerResponse
		container = containerFromProto(&req.Container)
	)

	if err := s.withStoreUpdate(ctx, func(ctx context.Context, store containers.Store) error {
		var fieldpaths []string
		if req.UpdateMask != nil && len(req.UpdateMask.Paths) > 0 {
			for _, path := range req.UpdateMask.Paths {
				fieldpaths = append(fieldpaths, path)
			}
		}

		updated, err := store.Update(ctx, container, fieldpaths...)
		if err != nil {
			return err
		}

		resp.Container = containerToProto(&updated)
		return nil
	}); err != nil {
		return &resp, errdefs.ToGRPC(err)
	}

	if err := s.publisher.Publish(ctx, "/containers/update", &eventsapi.ContainerUpdate{
		ID:          resp.Container.ID,
		Image:       resp.Container.Image,
		Labels:      resp.Container.Labels,
		SnapshotKey: resp.Container.SnapshotKey,
	}); err != nil {
		return &resp, err
	}

	return &resp, nil
}

func (s *service) Delete(ctx context.Context, req *api.DeleteContainerRequest) (*empty.Empty, error) {
	if err := s.withStoreUpdate(ctx, func(ctx context.Context, store containers.Store) error {
		return store.Delete(ctx, req.ID)
	}); err != nil {
		return &empty.Empty{}, errdefs.ToGRPC(err)
	}

	if err := s.publisher.Publish(ctx, "/containers/delete", &eventsapi.ContainerDelete{
		ID: req.ID,
	}); err != nil {
		return &empty.Empty{}, err
	}

	if err := s.db.GarbageCollect(ctx); err != nil {
		return &empty.Empty{}, errdefs.ToGRPC(errors.Wrap(err, "garbage collection failed"))
	}

	return &empty.Empty{}, nil
}

func (s *service) withStore(ctx context.Context, fn func(ctx context.Context, store containers.Store) error) func(tx *bolt.Tx) error {
	return func(tx *bolt.Tx) error { return fn(ctx, metadata.NewContainerStore(tx)) }
}

func (s *service) withStoreView(ctx context.Context, fn func(ctx context.Context, store containers.Store) error) error {
	return s.db.View(s.withStore(ctx, fn))
}

func (s *service) withStoreUpdate(ctx context.Context, fn func(ctx context.Context, store containers.Store) error) error {
	return s.db.Update(s.withStore(ctx, fn))
}
