package namespaces

import (
	"strings"

	"github.com/boltdb/bolt"
	eventsapi "github.com/containerd/containerd/api/services/events/v1"
	api "github.com/containerd/containerd/api/services/namespaces/v1"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/events"
	"github.com/containerd/containerd/metadata"
	"github.com/containerd/containerd/namespaces"
	"github.com/containerd/containerd/plugin"
	"github.com/golang/protobuf/ptypes/empty"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

func init() {
	plugin.Register(&plugin.Registration{
		Type: plugin.GRPCPlugin,
		ID:   "namespaces",
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

var _ api.NamespacesServer = &service{}

// NewService returns the GRPC namespaces server
func NewService(db *metadata.DB, publisher events.Publisher) api.NamespacesServer {
	return &service{
		db:        db,
		publisher: publisher,
	}
}

func (s *service) Register(server *grpc.Server) error {
	api.RegisterNamespacesServer(server, s)
	return nil
}

func (s *service) Get(ctx context.Context, req *api.GetNamespaceRequest) (*api.GetNamespaceResponse, error) {
	var resp api.GetNamespaceResponse

	return &resp, s.withStoreView(ctx, func(ctx context.Context, store namespaces.Store) error {
		labels, err := store.Labels(ctx, req.Name)
		if err != nil {
			return errdefs.ToGRPC(err)
		}

		resp.Namespace = api.Namespace{
			Name:   req.Name,
			Labels: labels,
		}

		return nil
	})
}

func (s *service) List(ctx context.Context, req *api.ListNamespacesRequest) (*api.ListNamespacesResponse, error) {
	var resp api.ListNamespacesResponse

	return &resp, s.withStoreView(ctx, func(ctx context.Context, store namespaces.Store) error {
		namespaces, err := store.List(ctx)
		if err != nil {
			return err
		}

		for _, namespace := range namespaces {
			labels, err := store.Labels(ctx, namespace)
			if err != nil {
				// In general, this should be unlikely, since we are holding a
				// transaction to service this request.
				return errdefs.ToGRPC(err)
			}

			resp.Namespaces = append(resp.Namespaces, api.Namespace{
				Name:   namespace,
				Labels: labels,
			})
		}

		return nil
	})
}

func (s *service) Create(ctx context.Context, req *api.CreateNamespaceRequest) (*api.CreateNamespaceResponse, error) {
	var resp api.CreateNamespaceResponse

	if err := s.withStoreUpdate(ctx, func(ctx context.Context, store namespaces.Store) error {
		if err := store.Create(ctx, req.Namespace.Name, req.Namespace.Labels); err != nil {
			return errdefs.ToGRPC(err)
		}

		for k, v := range req.Namespace.Labels {
			if err := store.SetLabel(ctx, req.Namespace.Name, k, v); err != nil {
				return err
			}
		}

		resp.Namespace = req.Namespace
		return nil
	}); err != nil {
		return &resp, err
	}

	if err := s.publisher.Publish(ctx, "/namespaces/create", &eventsapi.NamespaceCreate{
		Name:   req.Namespace.Name,
		Labels: req.Namespace.Labels,
	}); err != nil {
		return &resp, err
	}

	return &resp, nil

}

func (s *service) Update(ctx context.Context, req *api.UpdateNamespaceRequest) (*api.UpdateNamespaceResponse, error) {
	var resp api.UpdateNamespaceResponse
	if err := s.withStoreUpdate(ctx, func(ctx context.Context, store namespaces.Store) error {
		if req.UpdateMask != nil && len(req.UpdateMask.Paths) > 0 {
			for _, path := range req.UpdateMask.Paths {
				switch {
				case strings.HasPrefix(path, "labels."):
					key := strings.TrimPrefix(path, "labels.")
					if err := store.SetLabel(ctx, req.Namespace.Name, key, req.Namespace.Labels[key]); err != nil {
						return err
					}
				default:
					return grpc.Errorf(codes.InvalidArgument, "cannot update %q field", path)
				}
			}
		} else {
			// clear out the existing labels and then set them to the incoming request.
			// get current set of labels
			labels, err := store.Labels(ctx, req.Namespace.Name)
			if err != nil {
				return errdefs.ToGRPC(err)
			}

			for k := range labels {
				if err := store.SetLabel(ctx, req.Namespace.Name, k, ""); err != nil {
					return err
				}
			}

			for k, v := range req.Namespace.Labels {
				if err := store.SetLabel(ctx, req.Namespace.Name, k, v); err != nil {
					return err
				}

			}
		}

		return nil
	}); err != nil {
		return &resp, err
	}

	if err := s.publisher.Publish(ctx, "/namespaces/update", &eventsapi.NamespaceUpdate{
		Name:   req.Namespace.Name,
		Labels: req.Namespace.Labels,
	}); err != nil {
		return &resp, err
	}

	return &resp, nil
}

func (s *service) Delete(ctx context.Context, req *api.DeleteNamespaceRequest) (*empty.Empty, error) {
	if err := s.withStoreUpdate(ctx, func(ctx context.Context, store namespaces.Store) error {
		return errdefs.ToGRPC(store.Delete(ctx, req.Name))
	}); err != nil {
		return &empty.Empty{}, err
	}
	// set the namespace in the context before publishing the event
	ctx = namespaces.WithNamespace(ctx, req.Name)
	if err := s.publisher.Publish(ctx, "/namespaces/delete", &eventsapi.NamespaceDelete{
		Name: req.Name,
	}); err != nil {
		return &empty.Empty{}, err
	}

	return &empty.Empty{}, nil
}

func (s *service) withStore(ctx context.Context, fn func(ctx context.Context, store namespaces.Store) error) func(tx *bolt.Tx) error {
	return func(tx *bolt.Tx) error { return fn(ctx, metadata.NewNamespaceStore(tx)) }
}

func (s *service) withStoreView(ctx context.Context, fn func(ctx context.Context, store namespaces.Store) error) error {
	return s.db.View(s.withStore(ctx, fn))
}

func (s *service) withStoreUpdate(ctx context.Context, fn func(ctx context.Context, store namespaces.Store) error) error {
	return s.db.Update(s.withStore(ctx, fn))
}
