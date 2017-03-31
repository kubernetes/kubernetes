package rootfs

import (
	"syscall"

	"github.com/docker/containerd"
	rootfsapi "github.com/docker/containerd/api/services/rootfs"
	containerd_v1_types "github.com/docker/containerd/api/types/mount"
	"github.com/docker/containerd/content"
	"github.com/docker/containerd/log"
	"github.com/docker/containerd/plugin"
	"github.com/docker/containerd/rootfs"
	"github.com/docker/containerd/snapshot"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

func init() {
	plugin.Register("rootfs-grpc", &plugin.Registration{
		Type: plugin.GRPCPlugin,
		Init: func(ic *plugin.InitContext) (interface{}, error) {
			return NewService(ic.Store, ic.Snapshotter)
		},
	})
}

type Service struct {
	store       *content.Store
	snapshotter snapshot.Snapshotter
}

func NewService(store *content.Store, snapshotter snapshot.Snapshotter) (*Service, error) {
	return &Service{
		store:       store,
		snapshotter: snapshotter,
	}, nil
}

func (s *Service) Register(gs *grpc.Server) error {
	rootfsapi.RegisterRootFSServer(gs, s)
	return nil
}

func (s *Service) Unpack(ctx context.Context, pr *rootfsapi.UnpackRequest) (*rootfsapi.UnpackResponse, error) {
	layers := make([]ocispec.Descriptor, len(pr.Layers))
	for i, l := range pr.Layers {
		layers[i] = ocispec.Descriptor{
			MediaType: l.MediaType,
			Digest:    l.Digest,
			Size:      l.Size_,
		}
	}
	log.G(ctx).Infof("Preparing %#v", layers)
	chainID, err := rootfs.Prepare(ctx, s.snapshotter, mounter{}, layers, s.store.Reader, emptyResolver, noopRegister)
	if err != nil {
		log.G(ctx).Errorf("Rootfs Prepare failed!: %v", err)
		return nil, err
	}
	log.G(ctx).Infof("ChainID %#v", chainID)
	return &rootfsapi.UnpackResponse{
		ChainID: chainID,
	}, nil
}

func (s *Service) Prepare(ctx context.Context, ir *rootfsapi.PrepareRequest) (*rootfsapi.MountResponse, error) {
	mounts, err := rootfs.InitRootFS(ctx, ir.Name, ir.ChainID, ir.Readonly, s.snapshotter, mounter{})
	if err != nil {
		return nil, grpc.Errorf(codes.AlreadyExists, "%v", err)
	}
	return &rootfsapi.MountResponse{
		Mounts: apiMounts(mounts),
	}, nil
}

func (s *Service) Mounts(ctx context.Context, mr *rootfsapi.MountsRequest) (*rootfsapi.MountResponse, error) {
	mounts, err := s.snapshotter.Mounts(ctx, mr.Name)
	if err != nil {
		return nil, err
	}
	return &rootfsapi.MountResponse{
		Mounts: apiMounts(mounts),
	}, nil
}

func apiMounts(mounts []containerd.Mount) []*containerd_v1_types.Mount {
	am := make([]*containerd_v1_types.Mount, len(mounts))
	for i, m := range mounts {
		am[i] = &containerd_v1_types.Mount{
			Type:    m.Type,
			Source:  m.Source,
			Options: m.Options,
		}
	}
	return am
}

type mounter struct{}

func (mounter) Mount(dir string, mounts ...containerd.Mount) error {
	return containerd.MountAll(mounts, dir)
}

func (mounter) Unmount(dir string) error {
	return syscall.Unmount(dir, 0)
}

func emptyResolver(digest.Digest) digest.Digest {
	return digest.Digest("")
}

func noopRegister(digest.Digest, digest.Digest) error {
	return nil
}
