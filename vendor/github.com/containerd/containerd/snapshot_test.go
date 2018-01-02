package containerd

import (
	"context"
	"runtime"
	"testing"

	"github.com/containerd/containerd/snapshot"
	"github.com/containerd/containerd/snapshot/testsuite"
)

func newSnapshotter(ctx context.Context, root string) (snapshot.Snapshotter, func() error, error) {
	client, err := New(address)
	if err != nil {
		return nil, nil, err
	}

	sn := client.SnapshotService(DefaultSnapshotter)

	return sn, func() error {
		return client.Close()
	}, nil
}

func TestSnapshotterClient(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}
	if runtime.GOOS == "windows" {
		t.Skip("snapshots not yet supported on Windows")
	}
	testsuite.SnapshotterSuite(t, "SnapshotterClient", newSnapshotter)
}
