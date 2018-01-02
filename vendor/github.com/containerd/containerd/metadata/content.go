package metadata

import (
	"context"
	"encoding/binary"
	"strings"
	"sync"
	"time"

	"github.com/boltdb/bolt"
	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/filters"
	"github.com/containerd/containerd/labels"
	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/metadata/boltutil"
	"github.com/containerd/containerd/namespaces"
	digest "github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

type contentStore struct {
	content.Store
	db *DB
	l  sync.RWMutex
}

// newContentStore returns a namespaced content store using an existing
// content store interface.
func newContentStore(db *DB, cs content.Store) *contentStore {
	return &contentStore{
		Store: cs,
		db:    db,
	}
}

func (cs *contentStore) Info(ctx context.Context, dgst digest.Digest) (content.Info, error) {
	ns, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return content.Info{}, err
	}

	var info content.Info
	if err := view(ctx, cs.db, func(tx *bolt.Tx) error {
		bkt := getBlobBucket(tx, ns, dgst)
		if bkt == nil {
			return errors.Wrapf(errdefs.ErrNotFound, "content digest %v", dgst)
		}

		info.Digest = dgst
		return readInfo(&info, bkt)
	}); err != nil {
		return content.Info{}, err
	}

	return info, nil
}

func (cs *contentStore) Update(ctx context.Context, info content.Info, fieldpaths ...string) (content.Info, error) {
	ns, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return content.Info{}, err
	}

	cs.l.RLock()
	defer cs.l.RUnlock()

	updated := content.Info{
		Digest: info.Digest,
	}
	if err := update(ctx, cs.db, func(tx *bolt.Tx) error {
		bkt := getBlobBucket(tx, ns, info.Digest)
		if bkt == nil {
			return errors.Wrapf(errdefs.ErrNotFound, "content digest %v", info.Digest)
		}

		if err := readInfo(&updated, bkt); err != nil {
			return errors.Wrapf(err, "info %q", info.Digest)
		}

		if len(fieldpaths) > 0 {
			for _, path := range fieldpaths {
				if strings.HasPrefix(path, "labels.") {
					if updated.Labels == nil {
						updated.Labels = map[string]string{}
					}

					key := strings.TrimPrefix(path, "labels.")
					updated.Labels[key] = info.Labels[key]
					continue
				}

				switch path {
				case "labels":
					updated.Labels = info.Labels
				default:
					return errors.Wrapf(errdefs.ErrInvalidArgument, "cannot update %q field on content info %q", path, info.Digest)
				}
			}
		} else {
			// Set mutable fields
			updated.Labels = info.Labels
		}
		if err := validateInfo(&updated); err != nil {
			return err
		}

		updated.UpdatedAt = time.Now().UTC()
		return writeInfo(&updated, bkt)
	}); err != nil {
		return content.Info{}, err
	}
	return updated, nil
}

func (cs *contentStore) Walk(ctx context.Context, fn content.WalkFunc, fs ...string) error {
	ns, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return err
	}

	filter, err := filters.ParseAll(fs...)
	if err != nil {
		return err
	}

	// TODO: Batch results to keep from reading all info into memory
	var infos []content.Info
	if err := view(ctx, cs.db, func(tx *bolt.Tx) error {
		bkt := getBlobsBucket(tx, ns)
		if bkt == nil {
			return nil
		}

		return bkt.ForEach(func(k, v []byte) error {
			dgst, err := digest.Parse(string(k))
			if err != nil {
				// Not a digest, skip
				return nil
			}
			bbkt := bkt.Bucket(k)
			if bbkt == nil {
				return nil
			}
			info := content.Info{
				Digest: dgst,
			}
			if err := readInfo(&info, bkt.Bucket(k)); err != nil {
				return err
			}
			if filter.Match(adaptContentInfo(info)) {
				infos = append(infos, info)
			}
			return nil
		})
	}); err != nil {
		return err
	}

	for _, info := range infos {
		if err := fn(info); err != nil {
			return err
		}
	}

	return nil
}

func (cs *contentStore) Delete(ctx context.Context, dgst digest.Digest) error {
	ns, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return err
	}

	cs.l.RLock()
	defer cs.l.RUnlock()

	return update(ctx, cs.db, func(tx *bolt.Tx) error {
		bkt := getBlobBucket(tx, ns, dgst)
		if bkt == nil {
			return errors.Wrapf(errdefs.ErrNotFound, "content digest %v", dgst)
		}

		if err := getBlobsBucket(tx, ns).DeleteBucket([]byte(dgst.String())); err != nil {
			return err
		}

		// Mark content store as dirty for triggering garbage collection
		cs.db.dirtyL.Lock()
		cs.db.dirtyCS = true
		cs.db.dirtyL.Unlock()

		return nil
	})
}

func (cs *contentStore) ListStatuses(ctx context.Context, fs ...string) ([]content.Status, error) {
	ns, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return nil, err
	}

	filter, err := filters.ParseAll(fs...)
	if err != nil {
		return nil, err
	}

	brefs := map[string]string{}
	if err := view(ctx, cs.db, func(tx *bolt.Tx) error {
		bkt := getIngestBucket(tx, ns)
		if bkt == nil {
			return nil
		}

		return bkt.ForEach(func(k, v []byte) error {
			// TODO(dmcgowan): match name and potentially labels here
			brefs[string(k)] = string(v)
			return nil
		})
	}); err != nil {
		return nil, err
	}

	statuses := make([]content.Status, 0, len(brefs))
	for k, bref := range brefs {
		status, err := cs.Store.Status(ctx, bref)
		if err != nil {
			if errdefs.IsNotFound(err) {
				continue
			}
			return nil, err
		}
		status.Ref = k

		if filter.Match(adaptContentStatus(status)) {
			statuses = append(statuses, status)
		}
	}

	return statuses, nil

}

func getRef(tx *bolt.Tx, ns, ref string) string {
	bkt := getIngestBucket(tx, ns)
	if bkt == nil {
		return ""
	}
	v := bkt.Get([]byte(ref))
	if len(v) == 0 {
		return ""
	}
	return string(v)
}

func (cs *contentStore) Status(ctx context.Context, ref string) (content.Status, error) {
	ns, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return content.Status{}, err
	}

	var bref string
	if err := view(ctx, cs.db, func(tx *bolt.Tx) error {
		bref = getRef(tx, ns, ref)
		if bref == "" {
			return errors.Wrapf(errdefs.ErrNotFound, "reference %v", ref)
		}

		return nil
	}); err != nil {
		return content.Status{}, err
	}

	st, err := cs.Store.Status(ctx, bref)
	if err != nil {
		return content.Status{}, err
	}
	st.Ref = ref
	return st, nil
}

func (cs *contentStore) Abort(ctx context.Context, ref string) error {
	ns, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return err
	}

	cs.l.RLock()
	defer cs.l.RUnlock()

	return update(ctx, cs.db, func(tx *bolt.Tx) error {
		bkt := getIngestBucket(tx, ns)
		if bkt == nil {
			return errors.Wrapf(errdefs.ErrNotFound, "reference %v", ref)
		}
		bref := string(bkt.Get([]byte(ref)))
		if bref == "" {
			return errors.Wrapf(errdefs.ErrNotFound, "reference %v", ref)
		}
		if err := bkt.Delete([]byte(ref)); err != nil {
			return err
		}

		return cs.Store.Abort(ctx, bref)
	})

}

func (cs *contentStore) Writer(ctx context.Context, ref string, size int64, expected digest.Digest) (content.Writer, error) {
	ns, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return nil, err
	}

	cs.l.RLock()
	defer cs.l.RUnlock()

	var w content.Writer
	if err := update(ctx, cs.db, func(tx *bolt.Tx) error {
		if expected != "" {
			cbkt := getBlobBucket(tx, ns, expected)
			if cbkt != nil {
				return errors.Wrapf(errdefs.ErrAlreadyExists, "content %v", expected)
			}
		}

		bkt, err := createIngestBucket(tx, ns)
		if err != nil {
			return err
		}

		var (
			bref  string
			brefb = bkt.Get([]byte(ref))
		)

		if brefb == nil {
			sid, err := bkt.NextSequence()
			if err != nil {
				return err
			}

			bref = createKey(sid, ns, ref)
			if err := bkt.Put([]byte(ref), []byte(bref)); err != nil {
				return err
			}
		} else {
			bref = string(brefb)
		}

		// Do not use the passed in expected value here since it was
		// already checked against the user metadata. If the content
		// store has the content, it must still be written before
		// linked into the given namespace. It is possible in the future
		// to allow content which exists in content store but not
		// namespace to be linked here and returned an exist error, but
		// this would require more configuration to make secure.
		w, err = cs.Store.Writer(ctx, bref, size, "")
		return err
	}); err != nil {
		return nil, err
	}

	// TODO: keep the expected in the writer to use on commit
	// when no expected is provided there.
	return &namespacedWriter{
		Writer:    w,
		ref:       ref,
		namespace: ns,
		db:        cs.db,
		l:         &cs.l,
	}, nil
}

type namespacedWriter struct {
	content.Writer
	ref       string
	namespace string
	db        transactor
	l         *sync.RWMutex
}

func (nw *namespacedWriter) Commit(ctx context.Context, size int64, expected digest.Digest, opts ...content.Opt) error {
	nw.l.RLock()
	defer nw.l.RUnlock()

	return update(ctx, nw.db, func(tx *bolt.Tx) error {
		bkt := getIngestBucket(tx, nw.namespace)
		if bkt != nil {
			if err := bkt.Delete([]byte(nw.ref)); err != nil {
				return err
			}
		}
		dgst, err := nw.commit(ctx, tx, size, expected, opts...)
		if err != nil {
			return err
		}
		return addContentLease(ctx, tx, dgst)
	})
}

func (nw *namespacedWriter) commit(ctx context.Context, tx *bolt.Tx, size int64, expected digest.Digest, opts ...content.Opt) (digest.Digest, error) {
	var base content.Info
	for _, opt := range opts {
		if err := opt(&base); err != nil {
			return "", err
		}
	}
	if err := validateInfo(&base); err != nil {
		return "", err
	}

	status, err := nw.Writer.Status()
	if err != nil {
		return "", err
	}
	if size != 0 && size != status.Offset {
		return "", errors.Errorf("%q failed size validation: %v != %v", nw.ref, status.Offset, size)
	}
	size = status.Offset

	actual := nw.Writer.Digest()

	if err := nw.Writer.Commit(ctx, size, expected); err != nil {
		if !errdefs.IsAlreadyExists(err) {
			return "", err
		}
		if getBlobBucket(tx, nw.namespace, actual) != nil {
			return "", errors.Wrapf(errdefs.ErrAlreadyExists, "content %v", actual)
		}
	}

	bkt, err := createBlobBucket(tx, nw.namespace, actual)
	if err != nil {
		return "", err
	}

	commitTime := time.Now().UTC()

	sizeEncoded, err := encodeInt(size)
	if err != nil {
		return "", err
	}

	if err := boltutil.WriteTimestamps(bkt, commitTime, commitTime); err != nil {
		return "", err
	}
	if err := boltutil.WriteLabels(bkt, base.Labels); err != nil {
		return "", err
	}
	return actual, bkt.Put(bucketKeySize, sizeEncoded)
}

func (nw *namespacedWriter) Status() (content.Status, error) {
	st, err := nw.Writer.Status()
	if err == nil {
		st.Ref = nw.ref
	}
	return st, err
}

func (cs *contentStore) ReaderAt(ctx context.Context, dgst digest.Digest) (content.ReaderAt, error) {
	if err := cs.checkAccess(ctx, dgst); err != nil {
		return nil, err
	}
	return cs.Store.ReaderAt(ctx, dgst)
}

func (cs *contentStore) checkAccess(ctx context.Context, dgst digest.Digest) error {
	ns, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return err
	}

	return view(ctx, cs.db, func(tx *bolt.Tx) error {
		bkt := getBlobBucket(tx, ns, dgst)
		if bkt == nil {
			return errors.Wrapf(errdefs.ErrNotFound, "content digest %v", dgst)
		}
		return nil
	})
}

func validateInfo(info *content.Info) error {
	for k, v := range info.Labels {
		if err := labels.Validate(k, v); err == nil {
			return errors.Wrapf(err, "info.Labels")
		}
	}

	return nil
}

func readInfo(info *content.Info, bkt *bolt.Bucket) error {
	if err := boltutil.ReadTimestamps(bkt, &info.CreatedAt, &info.UpdatedAt); err != nil {
		return err
	}

	labels, err := boltutil.ReadLabels(bkt)
	if err != nil {
		return err
	}
	info.Labels = labels

	if v := bkt.Get(bucketKeySize); len(v) > 0 {
		info.Size, _ = binary.Varint(v)
	}

	return nil
}

func writeInfo(info *content.Info, bkt *bolt.Bucket) error {
	if err := boltutil.WriteTimestamps(bkt, info.CreatedAt, info.UpdatedAt); err != nil {
		return err
	}

	if err := boltutil.WriteLabels(bkt, info.Labels); err != nil {
		return errors.Wrapf(err, "writing labels for info %v", info.Digest)
	}

	// Write size
	sizeEncoded, err := encodeInt(info.Size)
	if err != nil {
		return err
	}

	return bkt.Put(bucketKeySize, sizeEncoded)
}

func (cs *contentStore) garbageCollect(ctx context.Context) error {
	lt1 := time.Now()
	cs.l.Lock()
	defer func() {
		cs.l.Unlock()
		log.G(ctx).WithField("t", time.Now().Sub(lt1)).Debugf("content garbage collected")
	}()

	seen := map[string]struct{}{}
	if err := cs.db.View(func(tx *bolt.Tx) error {
		v1bkt := tx.Bucket(bucketKeyVersion)
		if v1bkt == nil {
			return nil
		}

		// iterate through each namespace
		v1c := v1bkt.Cursor()

		for k, v := v1c.First(); k != nil; k, v = v1c.Next() {
			if v != nil {
				continue
			}

			cbkt := v1bkt.Bucket(k).Bucket(bucketKeyObjectContent)
			if cbkt == nil {
				continue
			}
			bbkt := cbkt.Bucket(bucketKeyObjectBlob)
			if err := bbkt.ForEach(func(ck, cv []byte) error {
				if cv == nil {
					seen[string(ck)] = struct{}{}
				}
				return nil
			}); err != nil {
				return err
			}
		}

		return nil
	}); err != nil {
		return err
	}

	return cs.Store.Walk(ctx, func(info content.Info) error {
		if _, ok := seen[info.Digest.String()]; !ok {
			if err := cs.Store.Delete(ctx, info.Digest); err != nil {
				return err
			}
			log.G(ctx).WithField("digest", info.Digest).Debug("removed content")
		}
		return nil
	})
}
