package leases

import (
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"time"

	"google.golang.org/grpc"

	"github.com/boltdb/bolt"
	api "github.com/containerd/containerd/api/services/leases/v1"
	"github.com/containerd/containerd/metadata"
	"github.com/containerd/containerd/plugin"
	"github.com/golang/protobuf/ptypes/empty"
	"golang.org/x/net/context"
)

func init() {
	plugin.Register(&plugin.Registration{
		Type: plugin.GRPCPlugin,
		ID:   "leases",
		Requires: []plugin.Type{
			plugin.MetadataPlugin,
		},
		InitFn: func(ic *plugin.InitContext) (interface{}, error) {
			m, err := ic.Get(plugin.MetadataPlugin)
			if err != nil {
				return nil, err
			}
			return NewService(m.(*metadata.DB)), nil
		},
	})
}

type service struct {
	db *metadata.DB
}

// NewService returns the GRPC metadata server
func NewService(db *metadata.DB) api.LeasesServer {
	return &service{
		db: db,
	}
}

func (s *service) Register(server *grpc.Server) error {
	api.RegisterLeasesServer(server, s)
	return nil
}

func (s *service) Create(ctx context.Context, r *api.CreateRequest) (*api.CreateResponse, error) {
	lid := r.ID
	if lid == "" {
		lid = generateLeaseID()
	}
	var trans metadata.Lease
	if err := s.db.Update(func(tx *bolt.Tx) error {
		var err error
		trans, err = metadata.NewLeaseManager(tx).Create(ctx, lid, r.Labels)
		return err
	}); err != nil {
		return nil, err
	}
	return &api.CreateResponse{
		Lease: txToGRPC(trans),
	}, nil
}

func (s *service) Delete(ctx context.Context, r *api.DeleteRequest) (*empty.Empty, error) {
	if err := s.db.Update(func(tx *bolt.Tx) error {
		return metadata.NewLeaseManager(tx).Delete(ctx, r.ID)
	}); err != nil {
		return nil, err
	}
	return &empty.Empty{}, nil
}

func (s *service) List(ctx context.Context, r *api.ListRequest) (*api.ListResponse, error) {
	var leases []metadata.Lease
	if err := s.db.View(func(tx *bolt.Tx) error {
		var err error
		leases, err = metadata.NewLeaseManager(tx).List(ctx, false, r.Filters...)
		return err
	}); err != nil {
		return nil, err
	}

	apileases := make([]*api.Lease, len(leases))
	for i := range leases {
		apileases[i] = txToGRPC(leases[i])
	}

	return &api.ListResponse{
		Leases: apileases,
	}, nil
}

func txToGRPC(tx metadata.Lease) *api.Lease {
	return &api.Lease{
		ID:        tx.ID,
		Labels:    tx.Labels,
		CreatedAt: tx.CreatedAt,
		// TODO: Snapshots
		// TODO: Content
	}
}

func generateLeaseID() string {
	t := time.Now()
	var b [3]byte
	// Ignore read failures, just decreases uniqueness
	rand.Read(b[:])
	return fmt.Sprintf("%d-%s", t.Nanosecond(), base64.URLEncoding.EncodeToString(b[:]))
}
