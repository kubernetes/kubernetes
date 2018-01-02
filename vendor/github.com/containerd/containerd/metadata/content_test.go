package metadata

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/boltdb/bolt"
	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/content/local"
	"github.com/containerd/containerd/content/testsuite"
)

func createContentStore(ctx context.Context, root string) (content.Store, func() error, error) {
	// TODO: Use mocked or in-memory store
	cs, err := local.NewStore(root)
	if err != nil {
		return nil, nil, err
	}

	db, err := bolt.Open(filepath.Join(root, "metadata.db"), 0660, nil)
	if err != nil {
		return nil, nil, err
	}

	return NewDB(db, cs, nil).ContentStore(), func() error {
		return db.Close()
	}, nil
}

func TestContent(t *testing.T) {
	testsuite.ContentSuite(t, "metadata", createContentStore)
}
