// +build !windows

package containerd

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/containerd/containerd/api/types"
	"github.com/containerd/containerd/containers"
	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/images"
	"github.com/containerd/containerd/platforms"
	"github.com/gogo/protobuf/proto"
	protobuf "github.com/gogo/protobuf/types"
	digest "github.com/opencontainers/go-digest"
	"github.com/opencontainers/image-spec/identity"
	"github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
)

// WithCheckpoint allows a container to be created from the checkpointed information
// provided by the descriptor. The image, snapshot, and runtime specifications are
// restored on the container
func WithCheckpoint(im Image, snapshotKey string) NewContainerOpts {
	// set image and rw, and spec
	return func(ctx context.Context, client *Client, c *containers.Container) error {
		var (
			desc  = im.Target()
			id    = desc.Digest
			store = client.ContentStore()
		)
		index, err := decodeIndex(ctx, store, id)
		if err != nil {
			return err
		}
		var rw *v1.Descriptor
		for _, m := range index.Manifests {
			switch m.MediaType {
			case v1.MediaTypeImageLayer:
				fk := m
				rw = &fk
			case images.MediaTypeDockerSchema2Manifest, images.MediaTypeDockerSchema2ManifestList:
				config, err := images.Config(ctx, store, m, platforms.Default())
				if err != nil {
					return errors.Wrap(err, "unable to resolve image config")
				}
				diffIDs, err := images.RootFS(ctx, store, config)
				if err != nil {
					return errors.Wrap(err, "unable to get rootfs")
				}
				setSnapshotterIfEmpty(c)
				if _, err := client.SnapshotService(c.Snapshotter).Prepare(ctx, snapshotKey, identity.ChainID(diffIDs).String()); err != nil {
					if !errdefs.IsAlreadyExists(err) {
						return err
					}
				}
				c.Image = index.Annotations["image.name"]
			case images.MediaTypeContainerd1CheckpointConfig:
				data, err := content.ReadBlob(ctx, store, m.Digest)
				if err != nil {
					return errors.Wrap(err, "unable to read checkpoint config")
				}
				var any protobuf.Any
				if err := proto.Unmarshal(data, &any); err != nil {
					return err
				}
				c.Spec = &any
			}
		}
		if rw != nil {
			// apply the rw snapshot to the new rw layer
			mounts, err := client.SnapshotService(c.Snapshotter).Mounts(ctx, snapshotKey)
			if err != nil {
				return errors.Wrapf(err, "unable to get mounts for %s", snapshotKey)
			}
			if _, err := client.DiffService().Apply(ctx, *rw, mounts); err != nil {
				return errors.Wrap(err, "unable to apply rw diff")
			}
		}
		c.SnapshotKey = snapshotKey
		return nil
	}
}

// WithTaskCheckpoint allows a task to be created with live runtime and memory data from a
// previous checkpoint. Additional software such as CRIU may be required to
// restore a task from a checkpoint
func WithTaskCheckpoint(im Image) NewTaskOpts {
	return func(ctx context.Context, c *Client, info *TaskInfo) error {
		desc := im.Target()
		id := desc.Digest
		index, err := decodeIndex(ctx, c.ContentStore(), id)
		if err != nil {
			return err
		}
		for _, m := range index.Manifests {
			if m.MediaType == images.MediaTypeContainerd1Checkpoint {
				info.Checkpoint = &types.Descriptor{
					MediaType: m.MediaType,
					Size_:     m.Size,
					Digest:    m.Digest,
				}
				return nil
			}
		}
		return fmt.Errorf("checkpoint not found in index %s", id)
	}
}

func decodeIndex(ctx context.Context, store content.Store, id digest.Digest) (*v1.Index, error) {
	var index v1.Index
	p, err := content.ReadBlob(ctx, store, id)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(p, &index); err != nil {
		return nil, err
	}

	return &index, nil
}
