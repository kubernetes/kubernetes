package naive

import (
	"context"
	"testing"

	"github.com/containerd/containerd/snapshot"
	"github.com/containerd/containerd/snapshot/testsuite"
	"github.com/containerd/containerd/testutil"
)

func newSnapshotter(ctx context.Context, root string) (snapshot.Snapshotter, func() error, error) {
	snapshotter, err := NewSnapshotter(root)
	if err != nil {
		return nil, nil, err
	}

	return snapshotter, nil, nil
}

func TestNaive(t *testing.T) {
	testutil.RequiresRoot(t)
	testsuite.SnapshotterSuite(t, "Naive", newSnapshotter)
}
