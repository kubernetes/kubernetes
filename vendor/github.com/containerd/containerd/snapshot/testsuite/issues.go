package testsuite

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/containerd/containerd/fs/fstest"
	"github.com/containerd/containerd/snapshot"
)

// Checks which cover former issues found in older layering models.
//
// NOTE: In older models, applying with tar was used to create read only layers,
// however with the snapshot model read only layers are created just using
// mounts and commits. Read write layers are a separate type of snapshot which
// is not committed, avoiding any confusion in the snapshotter about whether
// a snapshot will be mutated in the future.

// checkLayerFileUpdate tests the update of a single file in an upper layer
// Cause of issue was originally related to tar, snapshot should be able to
// avoid such issues by not relying on tar to create layers.
// See https://github.com/docker/docker/issues/21555
func checkLayerFileUpdate(ctx context.Context, t *testing.T, sn snapshot.Snapshotter, work string) {
	l1Init := fstest.Apply(
		fstest.CreateDir("/etc", 0700),
		fstest.CreateFile("/etc/hosts", []byte("mydomain 10.0.0.1"), 0644),
		fstest.CreateFile("/etc/profile", []byte("PATH=/usr/bin"), 0644),
	)
	l2Init := fstest.Apply(
		fstest.CreateFile("/etc/hosts", []byte("mydomain 10.0.0.2"), 0644),
		fstest.CreateFile("/etc/profile", []byte("PATH=/usr/bin"), 0666),
		fstest.CreateDir("/root", 0700),
		fstest.CreateFile("/root/.bashrc", []byte("PATH=/usr/sbin:/usr/bin"), 0644),
	)

	var sleepTime time.Duration

	// run 5 times to account for sporadic failure
	for i := 0; i < 5; i++ {
		time.Sleep(sleepTime)

		if err := checkSnapshots(ctx, sn, work, l1Init, l2Init); err != nil {
			t.Fatalf("Check snapshots failed: %+v", err)
		}

		// Sleep until next second boundary before running again
		nextTime := time.Now()
		sleepTime = time.Unix(nextTime.Unix()+1, 0).Sub(nextTime)
	}
}

// checkRemoveDirectoryInLowerLayer
// See https://github.com/docker/docker/issues/25244
func checkRemoveDirectoryInLowerLayer(ctx context.Context, t *testing.T, sn snapshot.Snapshotter, work string) {
	l1Init := fstest.Apply(
		fstest.CreateDir("/lib", 0700),
		fstest.CreateFile("/lib/hidden", []byte{}, 0644),
	)
	l2Init := fstest.Apply(
		fstest.RemoveAll("/lib"),
		fstest.CreateDir("/lib", 0700),
		fstest.CreateFile("/lib/not-hidden", []byte{}, 0644),
	)
	l3Init := fstest.Apply(
		fstest.CreateFile("/lib/newfile", []byte{}, 0644),
	)

	if err := checkSnapshots(ctx, sn, work, l1Init, l2Init, l3Init); err != nil {
		t.Fatalf("Check snapshots failed: %+v", err)
	}
}

// checkChown
// See https://github.com/docker/docker/issues/20240 aufs
// See https://github.com/docker/docker/issues/24913 overlay
// see https://github.com/docker/docker/issues/28391 overlay2
func checkChown(ctx context.Context, t *testing.T, sn snapshot.Snapshotter, work string) {
	l1Init := fstest.Apply(
		fstest.CreateDir("/opt", 0700),
		fstest.CreateDir("/opt/a", 0700),
		fstest.CreateDir("/opt/a/b", 0700),
		fstest.CreateFile("/opt/a/b/file.txt", []byte("hello"), 0644),
	)
	l2Init := fstest.Apply(
		fstest.Chown("/opt", 1, 1),
		fstest.Chown("/opt/a", 1, 1),
		fstest.Chown("/opt/a/b", 1, 1),
		fstest.Chown("/opt/a/b/file.txt", 1, 1),
	)

	if err := checkSnapshots(ctx, sn, work, l1Init, l2Init); err != nil {
		t.Fatalf("Check snapshots failed: %+v", err)
	}
}

// checkRename
// https://github.com/docker/docker/issues/25409
func checkRename(ctx context.Context, t *testing.T, sn snapshot.Snapshotter, work string) {
	l1Init := fstest.Apply(
		fstest.CreateDir("/dir1", 0700),
		fstest.CreateDir("/somefiles", 0700),
		fstest.CreateFile("/somefiles/f1", []byte("was here first!"), 0644),
		fstest.CreateFile("/somefiles/f2", []byte("nothing interesting"), 0644),
	)
	l2Init := fstest.Apply(
		fstest.Rename("/dir1", "/dir2"),
		fstest.CreateFile("/somefiles/f1-overwrite", []byte("new content 1"), 0644),
		fstest.Rename("/somefiles/f1-overwrite", "/somefiles/f1"),
		fstest.Rename("/somefiles/f2", "/somefiles/f3"),
	)

	if err := checkSnapshots(ctx, sn, work, l1Init, l2Init); err != nil {
		t.Fatalf("Check snapshots failed: %+v", err)
	}
}

// checkDirectoryPermissionOnCommit
// https://github.com/docker/docker/issues/27298
func checkDirectoryPermissionOnCommit(ctx context.Context, t *testing.T, sn snapshot.Snapshotter, work string) {
	l1Init := fstest.Apply(
		fstest.CreateDir("/dir1", 0700),
		fstest.CreateDir("/dir2", 0700),
		fstest.CreateDir("/dir3", 0700),
		fstest.CreateDir("/dir4", 0700),
		fstest.CreateFile("/dir4/f1", []byte("..."), 0644),
		fstest.CreateDir("/dir5", 0700),
		fstest.CreateFile("/dir5/f1", []byte("..."), 0644),
		fstest.Chown("/dir1", 1, 1),
		fstest.Chown("/dir2", 1, 1),
		fstest.Chown("/dir3", 1, 1),
		fstest.Chown("/dir5", 1, 1),
		fstest.Chown("/dir5/f1", 1, 1),
	)
	l2Init := fstest.Apply(
		fstest.Chown("/dir2", 0, 0),
		fstest.RemoveAll("/dir3"),
		fstest.Chown("/dir4", 1, 1),
		fstest.Chown("/dir4/f1", 1, 1),
	)
	l3Init := fstest.Apply(
		fstest.CreateDir("/dir3", 0700),
		fstest.Chown("/dir3", 1, 1),
		fstest.RemoveAll("/dir5"),
		fstest.CreateDir("/dir5", 0700),
		fstest.Chown("/dir5", 1, 1),
	)

	if err := checkSnapshots(ctx, sn, work, l1Init, l2Init, l3Init); err != nil {
		t.Fatalf("Check snapshots failed: %+v", err)
	}
}

// checkStatInWalk ensures that a stat can be called during a walk
func checkStatInWalk(ctx context.Context, t *testing.T, sn snapshot.Snapshotter, work string) {
	prefix := "stats-in-walk-"
	if err := createNamedSnapshots(ctx, sn, prefix); err != nil {
		t.Fatal(err)
	}

	err := sn.Walk(ctx, func(ctx context.Context, si snapshot.Info) error {
		if !strings.HasPrefix(si.Name, prefix) {
			// Only stat snapshots from this test
			return nil
		}
		si2, err := sn.Stat(ctx, si.Name)
		if err != nil {
			return err
		}

		return checkInfo(si, si2)
	})
	if err != nil {
		t.Fatal(err)
	}
}

func createNamedSnapshots(ctx context.Context, snapshotter snapshot.Snapshotter, ns string) error {
	c1 := fmt.Sprintf("%sc1", ns)
	c2 := fmt.Sprintf("%sc2", ns)
	if _, err := snapshotter.Prepare(ctx, c1+"-a", ""); err != nil {
		return err
	}
	if err := snapshotter.Commit(ctx, c1, c1+"-a"); err != nil {
		return err
	}
	if _, err := snapshotter.Prepare(ctx, c2+"-a", c1); err != nil {
		return err
	}
	if err := snapshotter.Commit(ctx, c2, c2+"-a"); err != nil {
		return err
	}
	if _, err := snapshotter.Prepare(ctx, fmt.Sprintf("%sa1", ns), c2); err != nil {
		return err
	}
	if _, err := snapshotter.View(ctx, fmt.Sprintf("%sv1", ns), c2); err != nil {
		return err
	}
	return nil
}

// More issues to test
//
// checkRemoveAfterCommit
// See https://github.com/docker/docker/issues/24309
//
// checkUnixDomainSockets
// See https://github.com/docker/docker/issues/12080
//
// checkDirectoryInodeStability
// See https://github.com/docker/docker/issues/19647
//
// checkOpenFileInodeStability
// See https://github.com/docker/docker/issues/12327
//
// checkGetCWD
// See https://github.com/docker/docker/issues/19082
//
// checkChmod
// See https://github.com/docker/machine/issues/3327
//
// checkRemoveInWalk
// Allow mutations during walk without deadlocking
