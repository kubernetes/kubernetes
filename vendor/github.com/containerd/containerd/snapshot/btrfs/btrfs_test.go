// +build linux,!no_btrfs

package btrfs

import (
	"context"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/containerd/containerd/mount"
	"github.com/containerd/containerd/snapshot"
	"github.com/containerd/containerd/snapshot/testsuite"
	"github.com/containerd/containerd/testutil"
	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

func boltSnapshotter(t *testing.T) func(context.Context, string) (snapshot.Snapshotter, func() error, error) {
	mkbtrfs, err := exec.LookPath("mkfs.btrfs")
	if err != nil {
		t.Skipf("could not find mkfs.btrfs: %v", err)
	}

	// TODO: Check for btrfs in /proc/module and skip if not loaded

	return func(ctx context.Context, root string) (snapshot.Snapshotter, func() error, error) {

		deviceName, cleanupDevice, err := testutil.NewLoopback(100 << 20) // 100 MB
		if err != nil {
			return nil, nil, err
		}

		if out, err := exec.Command(mkbtrfs, deviceName).CombinedOutput(); err != nil {
			return nil, nil, errors.Wrapf(err, "failed to make btrfs filesystem (out: %q)", out)
		}
		if out, err := exec.Command("mount", deviceName, root).CombinedOutput(); err != nil {
			return nil, nil, errors.Wrapf(err, "failed to mount device %s (out: %q)", deviceName, out)
		}

		snapshotter, err := NewSnapshotter(root)
		if err != nil {
			return nil, nil, errors.Wrap(err, "failed to create new snapshotter")
		}

		return snapshotter, func() error {
			err := mount.UnmountAll(root, unix.MNT_DETACH)
			if cerr := cleanupDevice(); cerr != nil {
				err = errors.Wrap(cerr, "device cleanup failed")
			}
			return err
		}, nil
	}
}

func TestBtrfs(t *testing.T) {
	testutil.RequiresRoot(t)
	testsuite.SnapshotterSuite(t, "Btrfs", boltSnapshotter(t))
}

func TestBtrfsMounts(t *testing.T) {
	testutil.RequiresRoot(t)
	ctx := context.Background()

	// create temporary directory for mount point
	mountPoint, err := ioutil.TempDir("", "containerd-btrfs-test")
	if err != nil {
		t.Fatal("could not create mount point for btrfs test", err)
	}
	defer os.RemoveAll(mountPoint)
	t.Log("temporary mount point created", mountPoint)

	root, err := ioutil.TempDir(mountPoint, "TestBtrfsPrepare-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(root)

	b, c, err := boltSnapshotter(t)(ctx, root)
	if err != nil {
		t.Fatal(err)
	}
	defer c()

	target := filepath.Join(root, "test")
	mounts, err := b.Prepare(ctx, target, "")
	if err != nil {
		t.Fatal(err)
	}
	t.Log(mounts)

	for _, mount := range mounts {
		if mount.Type != "btrfs" {
			t.Fatalf("wrong mount type: %v != btrfs", mount.Type)
		}

		// assumes the first, maybe incorrect in the future
		if !strings.HasPrefix(mount.Options[0], "subvolid=") {
			t.Fatalf("no subvolid option in %v", mount.Options)
		}
	}

	if err := os.MkdirAll(target, 0755); err != nil {
		t.Fatal(err)
	}
	if err := mount.All(mounts, target); err != nil {
		t.Fatal(err)
	}
	defer testutil.Unmount(t, target)

	// write in some data
	if err := ioutil.WriteFile(filepath.Join(target, "foo"), []byte("content"), 0777); err != nil {
		t.Fatal(err)
	}

	// TODO(stevvooe): We don't really make this with the driver, but that
	// might prove annoying in practice.
	if err := os.MkdirAll(filepath.Join(root, "snapshots"), 0755); err != nil {
		t.Fatal(err)
	}

	if err := b.Commit(ctx, filepath.Join(root, "snapshots/committed"), filepath.Join(root, "test")); err != nil {
		t.Fatal(err)
	}

	target = filepath.Join(root, "test2")
	mounts, err = b.Prepare(ctx, target, filepath.Join(root, "snapshots/committed"))
	if err != nil {
		t.Fatal(err)
	}

	if err := os.MkdirAll(target, 0755); err != nil {
		t.Fatal(err)
	}

	if err := mount.All(mounts, target); err != nil {
		t.Fatal(err)
	}
	defer testutil.Unmount(t, target)

	// TODO(stevvooe): Verify contents of "foo"
	if err := ioutil.WriteFile(filepath.Join(target, "bar"), []byte("content"), 0777); err != nil {
		t.Fatal(err)
	}

	if err := b.Commit(ctx, filepath.Join(root, "snapshots/committed2"), target); err != nil {
		t.Fatal(err)
	}
}
