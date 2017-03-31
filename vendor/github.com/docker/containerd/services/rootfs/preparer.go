package rootfs

import (
	"context"

	rootfsapi "github.com/docker/containerd/api/services/rootfs"
	containerd_v1_types "github.com/docker/containerd/api/types/descriptor"
	"github.com/docker/containerd/rootfs"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

func NewUnpackerFromClient(client rootfsapi.RootFSClient) rootfs.Unpacker {
	return remoteUnpacker{
		client: client,
	}
}

type remoteUnpacker struct {
	client rootfsapi.RootFSClient
}

func (rp remoteUnpacker) Unpack(ctx context.Context, layers []ocispec.Descriptor) (digest.Digest, error) {
	pr := rootfsapi.UnpackRequest{
		Layers: make([]*containerd_v1_types.Descriptor, len(layers)),
	}
	for i, l := range layers {
		pr.Layers[i] = &containerd_v1_types.Descriptor{
			MediaType: l.MediaType,
			Digest:    l.Digest,
			Size_:     l.Size,
		}
	}
	resp, err := rp.client.Unpack(ctx, &pr)
	if err != nil {
		return "", err
	}
	return resp.ChainID, nil
}
