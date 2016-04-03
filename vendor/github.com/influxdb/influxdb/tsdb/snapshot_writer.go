package tsdb

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/boltdb/bolt"
	"github.com/influxdb/influxdb/snapshot"
)

// NewSnapshotWriter returns a new snapshot.Writer that will write
// metadata and the store's shards to an archive.
func NewSnapshotWriter(meta []byte, store *Store) (*snapshot.Writer, error) {
	// Create snapshot writer.
	sw := snapshot.NewWriter()
	if err := func() error {
		// Create meta file.
		f := &snapshot.File{
			Name:    "meta",
			Size:    int64(len(meta)),
			ModTime: time.Now(),
		}
		sw.Manifest.Files = append(sw.Manifest.Files, *f)
		sw.FileWriters[f.Name] = NopWriteToCloser(bytes.NewReader(meta))

		// Create files for each shard.
		if err := appendShardSnapshotFiles(sw, store); err != nil {
			return fmt.Errorf("create shard snapshot files: %s", err)
		}

		return nil
	}(); err != nil {
		_ = sw.Close()
		return nil, err
	}

	return sw, nil
}

// appendShardSnapshotFiles adds snapshot files for each shard in the store.
func appendShardSnapshotFiles(sw *snapshot.Writer, store *Store) error {
	// Calculate absolute path of store to use for relative shard paths.
	storePath, err := filepath.Abs(store.Path())
	if err != nil {
		return fmt.Errorf("store abs path: %s", err)
	}

	// Create files for each shard.
	for _, shardID := range store.ShardIDs() {
		// Retrieve shard.
		sh := store.Shard(shardID)
		if sh == nil {
			return fmt.Errorf("shard not found: %d", shardID)
		}

		// Calculate relative path from store.
		shardPath, err := filepath.Abs(sh.Path())
		if err != nil {
			return fmt.Errorf("shard abs path: %s", err)
		}
		name, err := filepath.Rel(storePath, shardPath)
		if err != nil {
			return fmt.Errorf("shard rel path: %s", err)
		}

		if err := appendShardSnapshotFile(sw, sh, name); err != nil {
			return fmt.Errorf("append shard: name=%s, err=%s", name, err)
		}
	}

	return nil
}

func appendShardSnapshotFile(sw *snapshot.Writer, sh *Shard, name string) error {
	// Stat the underlying data file to retrieve last modified date.
	fi, err := os.Stat(sh.Path())
	if err != nil {
		return fmt.Errorf("stat shard data file: %s", err)
	}

	// Begin transaction.
	tx, err := sh.db.Begin(false)
	if err != nil {
		return fmt.Errorf("begin: %s", err)
	}

	// Create file.
	f := snapshot.File{
		Name:    name,
		Size:    tx.Size(),
		ModTime: fi.ModTime(),
	}

	// Append to snapshot writer.
	sw.Manifest.Files = append(sw.Manifest.Files, f)
	sw.FileWriters[f.Name] = &boltTxCloser{tx}
	return nil
}

// boltTxCloser wraps a Bolt transaction to implement io.Closer.
type boltTxCloser struct {
	*bolt.Tx
}

// Close rolls back the transaction.
func (tx *boltTxCloser) Close() error { return tx.Rollback() }

// NopWriteToCloser returns an io.WriterTo that implements io.Closer.
func NopWriteToCloser(w io.WriterTo) interface {
	io.WriterTo
	io.Closer
} {
	return &nopWriteToCloser{w}
}

type nopWriteToCloser struct {
	io.WriterTo
}

func (w *nopWriteToCloser) Close() error { return nil }
