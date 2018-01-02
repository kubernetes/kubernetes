package local

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/errdefs"
	"github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

// writer represents a write transaction against the blob store.
type writer struct {
	s         *store
	fp        *os.File // opened data file
	path      string   // path to writer dir
	ref       string   // ref key
	offset    int64
	total     int64
	digester  digest.Digester
	startedAt time.Time
	updatedAt time.Time
}

func (w *writer) Status() (content.Status, error) {
	return content.Status{
		Ref:       w.ref,
		Offset:    w.offset,
		Total:     w.total,
		StartedAt: w.startedAt,
		UpdatedAt: w.updatedAt,
	}, nil
}

// Digest returns the current digest of the content, up to the current write.
//
// Cannot be called concurrently with `Write`.
func (w *writer) Digest() digest.Digest {
	return w.digester.Digest()
}

// Write p to the transaction.
//
// Note that writes are unbuffered to the backing file. When writing, it is
// recommended to wrap in a bufio.Writer or, preferably, use io.CopyBuffer.
func (w *writer) Write(p []byte) (n int, err error) {
	n, err = w.fp.Write(p)
	w.digester.Hash().Write(p[:n])
	w.offset += int64(len(p))
	w.updatedAt = time.Now()
	return n, err
}

func (w *writer) Commit(ctx context.Context, size int64, expected digest.Digest, opts ...content.Opt) error {
	var base content.Info
	for _, opt := range opts {
		if err := opt(&base); err != nil {
			return err
		}
	}

	if w.fp == nil {
		return errors.Wrap(errdefs.ErrFailedPrecondition, "cannot commit on closed writer")
	}

	if err := w.fp.Sync(); err != nil {
		return errors.Wrap(err, "sync failed")
	}

	fi, err := w.fp.Stat()
	if err != nil {
		return errors.Wrap(err, "stat on ingest file failed")
	}

	// change to readonly, more important for read, but provides _some_
	// protection from this point on. We use the existing perms with a mask
	// only allowing reads honoring the umask on creation.
	//
	// This removes write and exec, only allowing read per the creation umask.
	//
	// NOTE: Windows does not support this operation
	if runtime.GOOS != "windows" {
		if err := w.fp.Chmod((fi.Mode() & os.ModePerm) &^ 0333); err != nil {
			return errors.Wrap(err, "failed to change ingest file permissions")
		}
	}

	if size > 0 && size != fi.Size() {
		return errors.Errorf("unexpected commit size %d, expected %d", fi.Size(), size)
	}

	if err := w.fp.Close(); err != nil {
		return errors.Wrap(err, "failed closing ingest")
	}

	dgst := w.digester.Digest()
	if expected != "" && expected != dgst {
		return errors.Errorf("unexpected commit digest %s, expected %s", dgst, expected)
	}

	var (
		ingest = filepath.Join(w.path, "data")
		target = w.s.blobPath(dgst)
	)

	// make sure parent directories of blob exist
	if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
		return err
	}

	// clean up!!
	defer os.RemoveAll(w.path)

	if err := os.Rename(ingest, target); err != nil {
		if os.IsExist(err) {
			// collision with the target file!
			return errors.Wrapf(errdefs.ErrAlreadyExists, "content %v", dgst)
		}
		return err
	}
	commitTime := time.Now()
	if err := os.Chtimes(target, commitTime, commitTime); err != nil {
		return err
	}

	w.fp = nil
	unlock(w.ref)

	if w.s.ls != nil && base.Labels != nil {
		if err := w.s.ls.Set(dgst, base.Labels); err != nil {
			return err
		}
	}

	return nil
}

// Close the writer, flushing any unwritten data and leaving the progress in
// tact.
//
// If one needs to resume the transaction, a new writer can be obtained from
// `Ingester.Writer` using the same key. The write can then be continued
// from it was left off.
//
// To abandon a transaction completely, first call close then `IngestManager.Abort` to
// clean up the associated resources.
func (w *writer) Close() (err error) {
	if w.fp != nil {
		w.fp.Sync()
		err = w.fp.Close()
		w.fp = nil
		unlock(w.ref)
		return
	}

	return nil
}

func (w *writer) Truncate(size int64) error {
	if size != 0 {
		return errors.New("Truncate: unsupported size")
	}
	w.offset = 0
	w.digester.Hash().Reset()
	return w.fp.Truncate(0)
}
