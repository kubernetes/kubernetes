package containerd

import (
	"context"
	"testing"

	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/content/testsuite"
	"github.com/pkg/errors"
)

func newContentStore(ctx context.Context, root string) (content.Store, func() error, error) {
	client, err := New(address)
	if err != nil {
		return nil, nil, err
	}

	cs := client.ContentStore()

	return cs, func() error {
		statuses, err := cs.ListStatuses(ctx)
		if err != nil {
			return err
		}
		for _, st := range statuses {
			if err := cs.Abort(ctx, st.Ref); err != nil {
				return errors.Wrapf(err, "failed to abort %s", st.Ref)
			}
		}
		return cs.Walk(ctx, func(info content.Info) error {
			return cs.Delete(ctx, info.Digest)
		})

	}, nil
}

func TestContentClient(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}
	testsuite.ContentSuite(t, "ContentClient", newContentStore)
}
