package snapshot

import (
	"context"
	"io"

	snapshotapi "github.com/containerd/containerd/api/services/snapshot/v1"
	"github.com/containerd/containerd/api/types"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/mount"
	"github.com/containerd/containerd/snapshot"
	protobuftypes "github.com/gogo/protobuf/types"
)

// NewSnapshotterFromClient returns a new Snapshotter which communicates
// over a GRPC connection.
func NewSnapshotterFromClient(client snapshotapi.SnapshotsClient, snapshotterName string) snapshot.Snapshotter {
	return &remoteSnapshotter{
		client:          client,
		snapshotterName: snapshotterName,
	}
}

type remoteSnapshotter struct {
	client          snapshotapi.SnapshotsClient
	snapshotterName string
}

func (r *remoteSnapshotter) Stat(ctx context.Context, key string) (snapshot.Info, error) {
	resp, err := r.client.Stat(ctx,
		&snapshotapi.StatSnapshotRequest{
			Snapshotter: r.snapshotterName,
			Key:         key,
		})
	if err != nil {
		return snapshot.Info{}, errdefs.FromGRPC(err)
	}
	return toInfo(resp.Info), nil
}

func (r *remoteSnapshotter) Update(ctx context.Context, info snapshot.Info, fieldpaths ...string) (snapshot.Info, error) {
	resp, err := r.client.Update(ctx,
		&snapshotapi.UpdateSnapshotRequest{
			Snapshotter: r.snapshotterName,
			Info:        fromInfo(info),
			UpdateMask: &protobuftypes.FieldMask{
				Paths: fieldpaths,
			},
		})
	if err != nil {
		return snapshot.Info{}, errdefs.FromGRPC(err)
	}
	return toInfo(resp.Info), nil
}

func (r *remoteSnapshotter) Usage(ctx context.Context, key string) (snapshot.Usage, error) {
	resp, err := r.client.Usage(ctx, &snapshotapi.UsageRequest{
		Snapshotter: r.snapshotterName,
		Key:         key,
	})
	if err != nil {
		return snapshot.Usage{}, errdefs.FromGRPC(err)
	}
	return toUsage(resp), nil
}

func (r *remoteSnapshotter) Mounts(ctx context.Context, key string) ([]mount.Mount, error) {
	resp, err := r.client.Mounts(ctx, &snapshotapi.MountsRequest{
		Snapshotter: r.snapshotterName,
		Key:         key,
	})
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}
	return toMounts(resp.Mounts), nil
}

func (r *remoteSnapshotter) Prepare(ctx context.Context, key, parent string, opts ...snapshot.Opt) ([]mount.Mount, error) {
	var local snapshot.Info
	for _, opt := range opts {
		if err := opt(&local); err != nil {
			return nil, err
		}
	}
	resp, err := r.client.Prepare(ctx, &snapshotapi.PrepareSnapshotRequest{
		Snapshotter: r.snapshotterName,
		Key:         key,
		Parent:      parent,
		Labels:      local.Labels,
	})
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}
	return toMounts(resp.Mounts), nil
}

func (r *remoteSnapshotter) View(ctx context.Context, key, parent string, opts ...snapshot.Opt) ([]mount.Mount, error) {
	var local snapshot.Info
	for _, opt := range opts {
		if err := opt(&local); err != nil {
			return nil, err
		}
	}
	resp, err := r.client.View(ctx, &snapshotapi.ViewSnapshotRequest{
		Snapshotter: r.snapshotterName,
		Key:         key,
		Parent:      parent,
		Labels:      local.Labels,
	})
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}
	return toMounts(resp.Mounts), nil
}

func (r *remoteSnapshotter) Commit(ctx context.Context, name, key string, opts ...snapshot.Opt) error {
	var local snapshot.Info
	for _, opt := range opts {
		if err := opt(&local); err != nil {
			return err
		}
	}
	_, err := r.client.Commit(ctx, &snapshotapi.CommitSnapshotRequest{
		Snapshotter: r.snapshotterName,
		Name:        name,
		Key:         key,
		Labels:      local.Labels,
	})
	return errdefs.FromGRPC(err)
}

func (r *remoteSnapshotter) Remove(ctx context.Context, key string) error {
	_, err := r.client.Remove(ctx, &snapshotapi.RemoveSnapshotRequest{
		Snapshotter: r.snapshotterName,
		Key:         key,
	})
	return errdefs.FromGRPC(err)
}

func (r *remoteSnapshotter) Walk(ctx context.Context, fn func(context.Context, snapshot.Info) error) error {
	sc, err := r.client.List(ctx, &snapshotapi.ListSnapshotsRequest{
		Snapshotter: r.snapshotterName,
	})
	if err != nil {
		return errdefs.FromGRPC(err)
	}
	for {
		resp, err := sc.Recv()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return errdefs.FromGRPC(err)
		}
		if resp == nil {
			return nil
		}
		for _, info := range resp.Info {
			if err := fn(ctx, toInfo(info)); err != nil {
				return err
			}
		}
	}
}

func toKind(kind snapshotapi.Kind) snapshot.Kind {
	if kind == snapshotapi.KindActive {
		return snapshot.KindActive
	}
	if kind == snapshotapi.KindView {
		return snapshot.KindView
	}
	return snapshot.KindCommitted
}

func toInfo(info snapshotapi.Info) snapshot.Info {
	return snapshot.Info{
		Name:    info.Name,
		Parent:  info.Parent,
		Kind:    toKind(info.Kind),
		Created: info.CreatedAt,
		Updated: info.UpdatedAt,
		Labels:  info.Labels,
	}
}

func toUsage(resp *snapshotapi.UsageResponse) snapshot.Usage {
	return snapshot.Usage{
		Inodes: resp.Inodes,
		Size:   resp.Size_,
	}
}

func toMounts(mm []*types.Mount) []mount.Mount {
	mounts := make([]mount.Mount, len(mm))
	for i, m := range mm {
		mounts[i] = mount.Mount{
			Type:    m.Type,
			Source:  m.Source,
			Options: m.Options,
		}
	}
	return mounts
}
