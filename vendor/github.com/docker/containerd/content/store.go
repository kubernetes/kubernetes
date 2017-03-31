package content

import (
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/docker/containerd/log"
	"github.com/nightlyone/lockfile"
	digest "github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

// Store is digest-keyed store for content. All data written into the store is
// stored under a verifiable digest.
//
// Store can generally support multi-reader, single-writer ingest of data,
// including resumable ingest.
type Store struct {
	root string
}

func NewStore(root string) (*Store, error) {
	if err := os.MkdirAll(filepath.Join(root, "ingest"), 0777); err != nil && !os.IsExist(err) {
		return nil, err
	}

	return &Store{
		root: root,
	}, nil
}

func (s *Store) Info(dgst digest.Digest) (Info, error) {
	p := s.blobPath(dgst)
	fi, err := os.Stat(p)
	if err != nil {
		if os.IsNotExist(err) {
			err = ErrNotFound
		}

		return Info{}, err
	}

	return Info{
		Digest:      dgst,
		Size:        fi.Size(),
		CommittedAt: fi.ModTime(),
	}, nil
}

// Open returns an io.ReadCloser for the blob.
//
// TODO(stevvooe): This would work much better as an io.ReaderAt in practice.
// Right now, we are doing type assertion to tease that out, but it won't scale
// well.
func (s *Store) Reader(ctx context.Context, dgst digest.Digest) (io.ReadCloser, error) {
	fp, err := os.Open(s.blobPath(dgst))
	if err != nil {
		if os.IsNotExist(err) {
			err = ErrNotFound
		}
		return nil, err
	}

	return fp, nil
}

// Delete removes a blob by its digest.
//
// While this is safe to do concurrently, safe exist-removal logic must hold
// some global lock on the store.
func (cs *Store) Delete(dgst digest.Digest) error {
	if err := os.RemoveAll(cs.blobPath(dgst)); err != nil {
		if !os.IsNotExist(err) {
			return err
		}

		return ErrNotFound
	}

	return nil
}

// TODO(stevvooe): Allow querying the set of blobs in the blob store.

// WalkFunc defines the callback for a blob walk.
//
// TODO(stevvooe): Remove the file info. Just need size and modtime. Perhaps,
// not a huge deal, considering we have a path, but let's not just let this one
// go without scrutiny.
type WalkFunc func(path string, fi os.FileInfo, dgst digest.Digest) error

func (cs *Store) Walk(fn WalkFunc) error {
	root := filepath.Join(cs.root, "blobs")
	var alg digest.Algorithm
	return filepath.Walk(root, func(path string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !fi.IsDir() && !alg.Available() {
			return nil
		}

		// TODO(stevvooe): There are few more cases with subdirs that should be
		// handled in case the layout gets corrupted. This isn't strict enough
		// an may spew bad data.

		if path == root {
			return nil
		}
		if filepath.Dir(path) == root {
			alg = digest.Algorithm(filepath.Base(path))

			if !alg.Available() {
				alg = ""
				return filepath.SkipDir
			}

			// descending into a hash directory
			return nil
		}

		dgst := digest.NewDigestFromHex(alg.String(), filepath.Base(path))
		if err := dgst.Validate(); err != nil {
			// log error but don't report
			log.L.WithError(err).WithField("path", path).Error("invalid digest for blob path")
			// if we see this, it could mean some sort of corruption of the
			// store or extra paths not expected previously.
		}

		return fn(path, fi, dgst)
	})
}

// Stat returns the current status of a blob by the ingest ref.
func (s *Store) Status(ref string) (Status, error) {
	dp := filepath.Join(s.ingestRoot(ref), "data")
	return s.status(dp)
}

// stat works like stat above except uses the path to the ingest.
func (s *Store) status(ingestPath string) (Status, error) {
	dp := filepath.Join(ingestPath, "data")
	fi, err := os.Stat(dp)
	if err != nil {
		return Status{}, err
	}

	ref, err := readFileString(filepath.Join(ingestPath, "ref"))
	if err != nil {
		return Status{}, err
	}

	var startedAt time.Time
	if st, ok := fi.Sys().(*syscall.Stat_t); ok {
		startedAt = time.Unix(st.Ctim.Sec, st.Ctim.Nsec)
	} else {
		startedAt = fi.ModTime()
	}

	return Status{
		Ref:       ref,
		Offset:    fi.Size(),
		Total:     s.total(ingestPath),
		UpdatedAt: fi.ModTime(),
		StartedAt: startedAt,
	}, nil
}

// total attempts to resolve the total expected size for the write.
func (s *Store) total(ingestPath string) int64 {
	totalS, err := readFileString(filepath.Join(ingestPath, "total"))
	if err != nil {
		return 0
	}

	total, err := strconv.ParseInt(totalS, 10, 64)
	if err != nil {
		// represents a corrupted file, should probably remove.
		return 0
	}

	return total
}

// Writer begins or resumes the active writer identified by ref. If the writer
// is already in use, an error is returned. Only one writer may be in use per
// ref at a time.
//
// The argument `ref` is used to uniquely identify a long-lived writer transaction.
func (s *Store) Writer(ctx context.Context, ref string, total int64, expected digest.Digest) (Writer, error) {
	path, refp, data, lock, err := s.ingestPaths(ref)
	if err != nil {
		return nil, err
	}

	if err := tryLock(lock); err != nil {
		if !os.IsNotExist(errors.Cause(err)) {
			return nil, errors.Wrapf(err, "locking %v failed", ref)
		}

		// if it doesn't exist, we'll make it so below!
	}

	var (
		digester  = digest.Canonical.Digester()
		offset    int64
		startedAt time.Time
		updatedAt time.Time
	)

	// ensure that the ingest path has been created.
	if err := os.Mkdir(path, 0755); err != nil {
		if !os.IsExist(err) {
			return nil, err
		}

		status, err := s.status(path)
		if err != nil {
			return nil, errors.Wrap(err, "failed reading status of resume write")
		}

		if ref != status.Ref {
			// NOTE(stevvooe): This is fairly catastrophic. Either we have some
			// layout corruption or a hash collision for the ref key.
			return nil, errors.Wrapf(err, "ref key does not match: %v != %v", ref, status.Ref)
		}

		if total > 0 && status.Total > 0 && total != status.Total {
			return nil, errors.Errorf("provided total differs from status: %v != %v", total, status.Total)
		}

		// slow slow slow!!, send to goroutine or use resumable hashes
		fp, err := os.Open(data)
		if err != nil {
			return nil, err
		}
		defer fp.Close()

		p := bufPool.Get().([]byte)
		defer bufPool.Put(p)

		offset, err = io.CopyBuffer(digester.Hash(), fp, p)
		if err != nil {
			return nil, err
		}

		updatedAt = status.UpdatedAt
		startedAt = status.StartedAt
		total = status.Total
	} else {
		// the ingest is new, we need to setup the target location.
		// write the ref to a file for later use
		if err := ioutil.WriteFile(refp, []byte(ref), 0666); err != nil {
			return nil, err
		}

		if total > 0 {
			if err := ioutil.WriteFile(filepath.Join(path, "total"), []byte(fmt.Sprint(total)), 0666); err != nil {
				return nil, err
			}
		}

		startedAt = time.Now()
		updatedAt = startedAt
	}

	fp, err := os.OpenFile(data, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		return nil, errors.Wrap(err, "failed to open data file")
	}

	return &writer{
		s:         s,
		fp:        fp,
		lock:      lock,
		ref:       ref,
		path:      path,
		offset:    offset,
		total:     total,
		digester:  digester,
		startedAt: startedAt,
		updatedAt: updatedAt,
	}, nil
}

// Abort an active transaction keyed by ref. If the ingest is active, it will
// be cancelled. Any resources associated with the ingest will be cleaned.
func (s *Store) Abort(ref string) error {
	root := s.ingestRoot(ref)
	if err := os.RemoveAll(root); err != nil {
		if os.IsNotExist(err) {
			return nil
		}

		return err
	}

	return nil
}

func (s *Store) Active() ([]Status, error) {
	fp, err := os.Open(filepath.Join(s.root, "ingest"))
	if err != nil {
		return nil, err
	}

	defer fp.Close()

	fis, err := fp.Readdir(-1)
	if err != nil {
		return nil, err
	}

	var active []Status
	for _, fi := range fis {
		p := filepath.Join(s.root, "ingest", fi.Name())
		stat, err := s.status(p)
		if err != nil {
			if !os.IsNotExist(err) {
				return nil, err
			}

			// TODO(stevvooe): This is a common error if uploads are being
			// completed while making this listing. Need to consider taking a
			// lock on the whole store to coordinate this aspect.
			//
			// Another option is to cleanup downloads asynchronously and
			// coordinate this method with the cleanup process.
			//
			// For now, we just skip them, as they really don't exist.
			continue
		}

		active = append(active, stat)
	}

	return active, nil
}

func (cs *Store) blobPath(dgst digest.Digest) string {
	return filepath.Join(cs.root, "blobs", dgst.Algorithm().String(), dgst.Hex())
}

func (s *Store) ingestRoot(ref string) string {
	dgst := digest.FromString(ref)
	return filepath.Join(s.root, "ingest", dgst.Hex())
}

// ingestPaths are returned, including the lockfile. The paths are the following:
//
// - root: entire ingest directory
// - ref: name of the starting ref, must be unique
// - data: file where data is written
// - lock: lock file location
//
func (s *Store) ingestPaths(ref string) (string, string, string, lockfile.Lockfile, error) {
	var (
		fp = s.ingestRoot(ref)
		rp = filepath.Join(fp, "ref")
		lp = filepath.Join(fp, "lock")
		dp = filepath.Join(fp, "data")
	)

	lock, err := lockfile.New(lp)
	if err != nil {
		return "", "", "", "", errors.Wrapf(err, "error creating lockfile %v", lp)
	}

	return fp, rp, dp, lock, nil
}
