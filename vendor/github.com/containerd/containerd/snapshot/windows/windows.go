// +build windows

package windows

import (
	"context"

	"github.com/containerd/containerd/mount"
	"github.com/containerd/containerd/plugin"
	"github.com/containerd/containerd/snapshot"
	"github.com/pkg/errors"
)

var (
	// ErrNotImplemented is returned when an action is not implemented
	ErrNotImplemented = errors.New("not implemented")
)

func init() {
	plugin.Register(&plugin.Registration{
		Type: plugin.SnapshotPlugin,
		ID:   "windows",
		InitFn: func(ic *plugin.InitContext) (interface{}, error) {
			return NewSnapshotter(ic.Root)
		},
	})
}

type snapshotter struct {
	root string
}

// NewSnapshotter returns a new windows snapshotter
func NewSnapshotter(root string) (snapshot.Snapshotter, error) {
	return &snapshotter{
		root: root,
	}, nil
}

// Stat returns the info for an active or committed snapshot by name or
// key.
//
// Should be used for parent resolution, existence checks and to discern
// the kind of snapshot.
func (o *snapshotter) Stat(ctx context.Context, key string) (snapshot.Info, error) {
	panic("not implemented")
}

func (o *snapshotter) Update(ctx context.Context, info snapshot.Info, fieldpaths ...string) (snapshot.Info, error) {
	panic("not implemented")
}

func (o *snapshotter) Usage(ctx context.Context, key string) (snapshot.Usage, error) {
	panic("not implemented")
}

func (o *snapshotter) Prepare(ctx context.Context, key, parent string, opts ...snapshot.Opt) ([]mount.Mount, error) {
	panic("not implemented")
}

func (o *snapshotter) View(ctx context.Context, key, parent string, opts ...snapshot.Opt) ([]mount.Mount, error) {
	panic("not implemented")
}

// Mounts returns the mounts for the transaction identified by key. Can be
// called on an read-write or readonly transaction.
//
// This can be used to recover mounts after calling View or Prepare.
func (o *snapshotter) Mounts(ctx context.Context, key string) ([]mount.Mount, error) {
	panic("not implemented")
}

func (o *snapshotter) Commit(ctx context.Context, name, key string, opts ...snapshot.Opt) error {
	panic("not implemented")
}

// Remove abandons the transaction identified by key. All resources
// associated with the key will be removed.
func (o *snapshotter) Remove(ctx context.Context, key string) error {
	panic("not implemented")
}

// Walk the committed snapshots.
func (o *snapshotter) Walk(ctx context.Context, fn func(context.Context, snapshot.Info) error) error {
	panic("not implemented")
}
