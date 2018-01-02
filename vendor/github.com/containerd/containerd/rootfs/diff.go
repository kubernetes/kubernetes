package rootfs

import (
	"fmt"

	"github.com/containerd/containerd/diff"
	"github.com/containerd/containerd/mount"
	"github.com/containerd/containerd/snapshot"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"golang.org/x/net/context"
)

// Diff creates a layer diff for the given snapshot identifier from the parent
// of the snapshot. A content ref is provided to track the progress of the
// content creation and the provided snapshotter and mount differ are used
// for calculating the diff. The descriptor for the layer diff is returned.
func Diff(ctx context.Context, snapshotID string, sn snapshot.Snapshotter, d diff.Differ, opts ...diff.Opt) (ocispec.Descriptor, error) {
	info, err := sn.Stat(ctx, snapshotID)
	if err != nil {
		return ocispec.Descriptor{}, err
	}

	lowerKey := fmt.Sprintf("%s-parent-view", info.Parent)
	lower, err := sn.View(ctx, lowerKey, info.Parent)
	if err != nil {
		return ocispec.Descriptor{}, err
	}
	defer sn.Remove(ctx, lowerKey)

	var upper []mount.Mount
	if info.Kind == snapshot.KindActive {
		upper, err = sn.Mounts(ctx, snapshotID)
		if err != nil {
			return ocispec.Descriptor{}, err
		}
	} else {
		upperKey := fmt.Sprintf("%s-view", snapshotID)
		upper, err = sn.View(ctx, upperKey, snapshotID)
		if err != nil {
			return ocispec.Descriptor{}, err
		}
		defer sn.Remove(ctx, lowerKey)
	}

	return d.DiffMounts(ctx, lower, upper, opts...)
}
