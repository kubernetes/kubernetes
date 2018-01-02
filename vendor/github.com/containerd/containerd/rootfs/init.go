package rootfs

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/mount"
	"github.com/containerd/containerd/snapshot"
	digest "github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

var (
	initializers = map[string]initializerFunc{}
)

type initializerFunc func(string) error

// Mounter handles mount and unmount
type Mounter interface {
	Mount(target string, mounts ...mount.Mount) error
	Unmount(target string) error
}

// InitRootFS initializes the snapshot for use as a rootfs
func InitRootFS(ctx context.Context, name string, parent digest.Digest, readonly bool, snapshotter snapshot.Snapshotter, mounter Mounter) ([]mount.Mount, error) {
	_, err := snapshotter.Stat(ctx, name)
	if err == nil {
		return nil, errors.Errorf("rootfs already exists")
	}
	// TODO: ensure not exist error once added to snapshot package

	parentS := parent.String()

	initName := defaultInitializer
	initFn := initializers[initName]
	if initFn != nil {
		parentS, err = createInitLayer(ctx, parentS, initName, initFn, snapshotter, mounter)
		if err != nil {
			return nil, err
		}
	}

	if readonly {
		return snapshotter.View(ctx, name, parentS)
	}

	return snapshotter.Prepare(ctx, name, parentS)
}

func createInitLayer(ctx context.Context, parent, initName string, initFn func(string) error, snapshotter snapshot.Snapshotter, mounter Mounter) (string, error) {
	initS := fmt.Sprintf("%s %s", parent, initName)
	if _, err := snapshotter.Stat(ctx, initS); err == nil {
		return initS, nil
	}
	// TODO: ensure not exist error once added to snapshot package

	// Create tempdir
	td, err := ioutil.TempDir("", "create-init-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(td)

	mounts, err := snapshotter.Prepare(ctx, td, parent)
	if err != nil {
		return "", err
	}
	defer func() {
		if err != nil {
			// TODO: once implemented uncomment
			//if rerr := snapshotter.Remove(ctx, td); rerr != nil {
			//	log.G(ctx).Errorf("Failed to remove snapshot %s: %v", td, merr)
			//}
		}
	}()

	if err = mounter.Mount(td, mounts...); err != nil {
		return "", err
	}

	if err = initFn(td); err != nil {
		if merr := mounter.Unmount(td); merr != nil {
			log.G(ctx).Errorf("Failed to unmount %s: %v", td, merr)
		}
		return "", err
	}

	if err = mounter.Unmount(td); err != nil {
		return "", err
	}

	if err := snapshotter.Commit(ctx, initS, td); err != nil {
		return "", err
	}

	return initS, nil
}
