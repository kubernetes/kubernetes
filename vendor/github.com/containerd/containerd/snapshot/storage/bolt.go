package storage

import (
	"context"
	"encoding/binary"
	"fmt"
	"strings"
	"time"

	"github.com/boltdb/bolt"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/metadata/boltutil"
	"github.com/containerd/containerd/snapshot"
	"github.com/pkg/errors"
)

var (
	bucketKeyStorageVersion = []byte("v1")
	bucketKeySnapshot       = []byte("snapshots")
	bucketKeyParents        = []byte("parents")

	bucketKeyID     = []byte("id")
	bucketKeyParent = []byte("parent")
	bucketKeyKind   = []byte("kind")
	bucketKeyInodes = []byte("inodes")
	bucketKeySize   = []byte("size")

	// ErrNoTransaction is returned when an operation is attempted with
	// a context which is not inside of a transaction.
	ErrNoTransaction = errors.New("no transaction in context")
)

// parentKey returns a composite key of the parent and child identifiers. The
// parts of the key are separated by a zero byte.
func parentKey(parent, child uint64) []byte {
	b := make([]byte, binary.Size([]uint64{parent, child})+1)
	i := binary.PutUvarint(b, parent)
	j := binary.PutUvarint(b[i+1:], child)
	return b[0 : i+j+1]
}

// parentPrefixKey returns the parent part of the composite key with the
// zero byte separator.
func parentPrefixKey(parent uint64) []byte {
	b := make([]byte, binary.Size(parent)+1)
	i := binary.PutUvarint(b, parent)
	return b[0 : i+1]
}

// getParentPrefix returns the first part of the composite key which
// represents the parent identifier.
func getParentPrefix(b []byte) uint64 {
	parent, _ := binary.Uvarint(b)
	return parent
}

// GetInfo returns the snapshot Info directly from the metadata. Requires a
// context with a storage transaction.
func GetInfo(ctx context.Context, key string) (string, snapshot.Info, snapshot.Usage, error) {
	var (
		id uint64
		su snapshot.Usage
		si = snapshot.Info{
			Name: key,
		}
	)
	err := withSnapshotBucket(ctx, key, func(ctx context.Context, bkt, pbkt *bolt.Bucket) error {
		getUsage(bkt, &su)
		return readSnapshot(bkt, &id, &si)
	})
	if err != nil {
		return "", snapshot.Info{}, snapshot.Usage{}, err
	}

	return fmt.Sprintf("%d", id), si, su, nil
}

// UpdateInfo updates an existing snapshot info's data
func UpdateInfo(ctx context.Context, info snapshot.Info, fieldpaths ...string) (snapshot.Info, error) {
	updated := snapshot.Info{
		Name: info.Name,
	}
	err := withBucket(ctx, func(ctx context.Context, bkt, pbkt *bolt.Bucket) error {
		sbkt := bkt.Bucket([]byte(info.Name))
		if sbkt == nil {
			return errors.Wrap(errdefs.ErrNotFound, "snapshot does not exist")
		}
		if err := readSnapshot(sbkt, nil, &updated); err != nil {
			return err
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
					return errors.Wrapf(errdefs.ErrInvalidArgument, "cannot update %q field on snapshot %q", path, info.Name)
				}
			}
		} else {
			// Set mutable fields
			updated.Labels = info.Labels
		}
		updated.Updated = time.Now().UTC()
		if err := boltutil.WriteTimestamps(sbkt, updated.Created, updated.Updated); err != nil {
			return err
		}

		return boltutil.WriteLabels(sbkt, updated.Labels)
	})
	if err != nil {
		return snapshot.Info{}, err
	}
	return updated, nil
}

// WalkInfo iterates through all metadata Info for the stored snapshots and
// calls the provided function for each. Requires a context with a storage
// transaction.
func WalkInfo(ctx context.Context, fn func(context.Context, snapshot.Info) error) error {
	return withBucket(ctx, func(ctx context.Context, bkt, pbkt *bolt.Bucket) error {
		return bkt.ForEach(func(k, v []byte) error {
			// skip non buckets
			if v != nil {
				return nil
			}
			var (
				sbkt = bkt.Bucket(k)
				si   = snapshot.Info{
					Name: string(k),
				}
			)
			if err := readSnapshot(sbkt, nil, &si); err != nil {
				return err
			}

			return fn(ctx, si)
		})
	})
}

// GetSnapshot returns the metadata for the active or view snapshot transaction
// referenced by the given key. Requires a context with a storage transaction.
func GetSnapshot(ctx context.Context, key string) (s Snapshot, err error) {
	err = withBucket(ctx, func(ctx context.Context, bkt, pbkt *bolt.Bucket) error {
		sbkt := bkt.Bucket([]byte(key))
		if sbkt == nil {
			return errors.Wrap(errdefs.ErrNotFound, "snapshot does not exist")
		}

		s.ID = fmt.Sprintf("%d", readID(sbkt))
		s.Kind = readKind(sbkt)

		if s.Kind != snapshot.KindActive && s.Kind != snapshot.KindView {
			return errors.Wrapf(errdefs.ErrFailedPrecondition, "requested snapshot %v not active or view", key)
		}

		if parentKey := sbkt.Get(bucketKeyParent); len(parentKey) > 0 {
			spbkt := bkt.Bucket(parentKey)
			if spbkt == nil {
				return errors.Wrap(errdefs.ErrNotFound, "parent does not exist")
			}

			s.ParentIDs, err = parents(bkt, spbkt, readID(spbkt))
			if err != nil {
				return errors.Wrap(err, "failed to get parent chain")
			}
		}
		return nil
	})
	if err != nil {
		return Snapshot{}, err
	}

	return
}

// CreateSnapshot inserts a record for an active or view snapshot with the provided parent.
func CreateSnapshot(ctx context.Context, kind snapshot.Kind, key, parent string, opts ...snapshot.Opt) (s Snapshot, err error) {
	switch kind {
	case snapshot.KindActive, snapshot.KindView:
	default:
		return Snapshot{}, errors.Wrapf(errdefs.ErrInvalidArgument, "snapshot type %v invalid; only snapshots of type Active or View can be created", kind)
	}
	var base snapshot.Info
	for _, opt := range opts {
		if err := opt(&base); err != nil {
			return Snapshot{}, err
		}
	}

	err = createBucketIfNotExists(ctx, func(ctx context.Context, bkt, pbkt *bolt.Bucket) error {
		var (
			spbkt *bolt.Bucket
		)
		if parent != "" {
			spbkt = bkt.Bucket([]byte(parent))
			if spbkt == nil {
				return errors.Wrap(errdefs.ErrNotFound, "missing parent bucket")
			}

			if readKind(spbkt) != snapshot.KindCommitted {
				return errors.Wrap(errdefs.ErrInvalidArgument, "parent is not committed snapshot")
			}
		}
		sbkt, err := bkt.CreateBucket([]byte(key))
		if err != nil {
			if err == bolt.ErrBucketExists {
				err = errors.Wrapf(errdefs.ErrAlreadyExists, "snapshot %v", key)
			}
			return err
		}

		id, err := bkt.NextSequence()
		if err != nil {
			return errors.Wrap(err, "unable to get identifier")
		}

		t := time.Now().UTC()
		si := snapshot.Info{
			Parent:  parent,
			Kind:    kind,
			Labels:  base.Labels,
			Created: t,
			Updated: t,
		}
		if err := putSnapshot(sbkt, id, si); err != nil {
			return err
		}

		if spbkt != nil {
			pid := readID(spbkt)

			// Store a backlink from the key to the parent. Store the snapshot name
			// as the value to allow following the backlink to the snapshot value.
			if err := pbkt.Put(parentKey(pid, id), []byte(key)); err != nil {
				return errors.Wrap(err, "failed to write parent link")
			}

			s.ParentIDs, err = parents(bkt, spbkt, pid)
			if err != nil {
				return errors.Wrap(err, "failed to get parent chain")
			}
		}

		s.ID = fmt.Sprintf("%d", id)
		s.Kind = kind
		return nil
	})
	if err != nil {
		return Snapshot{}, err
	}

	return
}

// Remove removes a snapshot from the metastore. The string identifier for the
// snapshot is returned as well as the kind. The provided context must contain a
// writable transaction.
func Remove(ctx context.Context, key string) (string, snapshot.Kind, error) {
	var (
		id uint64
		si snapshot.Info
	)

	if err := withBucket(ctx, func(ctx context.Context, bkt, pbkt *bolt.Bucket) error {
		sbkt := bkt.Bucket([]byte(key))
		if sbkt == nil {
			return errors.Wrapf(errdefs.ErrNotFound, "snapshot %v", key)
		}

		if err := readSnapshot(sbkt, &id, &si); err != nil {
			errors.Wrapf(err, "failed to read snapshot %s", key)
		}

		if pbkt != nil {
			k, _ := pbkt.Cursor().Seek(parentPrefixKey(id))
			if getParentPrefix(k) == id {
				return errors.Wrap(errdefs.ErrFailedPrecondition, "cannot remove snapshot with child")
			}

			if si.Parent != "" {
				spbkt := bkt.Bucket([]byte(si.Parent))
				if spbkt == nil {
					return errors.Wrapf(errdefs.ErrNotFound, "snapshot %v", key)
				}

				if err := pbkt.Delete(parentKey(readID(spbkt), id)); err != nil {
					return errors.Wrap(err, "failed to delete parent link")
				}
			}
		}

		if err := bkt.DeleteBucket([]byte(key)); err != nil {
			return errors.Wrap(err, "failed to delete snapshot")
		}

		return nil
	}); err != nil {
		return "", 0, err
	}

	return fmt.Sprintf("%d", id), si.Kind, nil
}

// CommitActive renames the active snapshot transaction referenced by `key`
// as a committed snapshot referenced by `Name`. The resulting snapshot  will be
// committed and readonly. The `key` reference will no longer be available for
// lookup or removal. The returned string identifier for the committed snapshot
// is the same identifier of the original active snapshot. The provided context
// must contain a writable transaction.
func CommitActive(ctx context.Context, key, name string, usage snapshot.Usage, opts ...snapshot.Opt) (string, error) {
	var (
		id   uint64
		base snapshot.Info
	)
	for _, opt := range opts {
		if err := opt(&base); err != nil {
			return "", err
		}
	}

	if err := withBucket(ctx, func(ctx context.Context, bkt, pbkt *bolt.Bucket) error {
		dbkt, err := bkt.CreateBucket([]byte(name))
		if err != nil {
			if err == bolt.ErrBucketExists {
				err = errdefs.ErrAlreadyExists
			}
			return errors.Wrapf(err, "committed snapshot %v", name)
		}
		sbkt := bkt.Bucket([]byte(key))
		if sbkt == nil {
			return errors.Wrap(errdefs.ErrNotFound, "failed to get active snapshot")
		}

		var si snapshot.Info
		if err := readSnapshot(sbkt, &id, &si); err != nil {
			return errors.Wrap(err, "failed to read snapshot")
		}

		if si.Kind != snapshot.KindActive {
			return errors.Wrapf(errdefs.ErrFailedPrecondition, "snapshot %v is not active", name)
		}
		si.Kind = snapshot.KindCommitted
		si.Created = time.Now().UTC()
		si.Updated = si.Created

		// Replace labels, do not inherit
		si.Labels = base.Labels

		if err := putSnapshot(dbkt, id, si); err != nil {
			return err
		}
		if err := putUsage(dbkt, usage); err != nil {
			return err
		}
		if err := bkt.DeleteBucket([]byte(key)); err != nil {
			return errors.Wrap(err, "failed to delete active")
		}
		if si.Parent != "" {
			spbkt := bkt.Bucket([]byte(si.Parent))
			if spbkt == nil {
				return errors.Wrap(errdefs.ErrNotFound, "missing parent")
			}
			pid := readID(spbkt)

			// Updates parent back link to use new key
			if err := pbkt.Put(parentKey(pid, id), []byte(name)); err != nil {
				return errors.Wrap(err, "failed to update parent link")
			}
		}

		return nil
	}); err != nil {
		return "", err
	}

	return fmt.Sprintf("%d", id), nil
}

func withSnapshotBucket(ctx context.Context, key string, fn func(context.Context, *bolt.Bucket, *bolt.Bucket) error) error {
	tx, ok := ctx.Value(transactionKey{}).(*bolt.Tx)
	if !ok {
		return ErrNoTransaction
	}
	bkt := tx.Bucket(bucketKeyStorageVersion)
	if bkt == nil {
		return errors.Wrap(errdefs.ErrNotFound, "bucket does not exist")
	}
	bkt = bkt.Bucket(bucketKeySnapshot)
	if bkt == nil {
		return errors.Wrap(errdefs.ErrNotFound, "snapshots bucket does not exist")
	}
	bkt = bkt.Bucket([]byte(key))
	if bkt == nil {
		return errors.Wrap(errdefs.ErrNotFound, "snapshot does not exist")
	}

	return fn(ctx, bkt, bkt.Bucket(bucketKeyParents))
}

func withBucket(ctx context.Context, fn func(context.Context, *bolt.Bucket, *bolt.Bucket) error) error {
	tx, ok := ctx.Value(transactionKey{}).(*bolt.Tx)
	if !ok {
		return ErrNoTransaction
	}
	bkt := tx.Bucket(bucketKeyStorageVersion)
	if bkt == nil {
		return errors.Wrap(errdefs.ErrNotFound, "bucket does not exist")
	}
	return fn(ctx, bkt.Bucket(bucketKeySnapshot), bkt.Bucket(bucketKeyParents))
}

func createBucketIfNotExists(ctx context.Context, fn func(context.Context, *bolt.Bucket, *bolt.Bucket) error) error {
	tx, ok := ctx.Value(transactionKey{}).(*bolt.Tx)
	if !ok {
		return ErrNoTransaction
	}

	bkt, err := tx.CreateBucketIfNotExists(bucketKeyStorageVersion)
	if err != nil {
		return errors.Wrap(err, "failed to create version bucket")
	}
	sbkt, err := bkt.CreateBucketIfNotExists(bucketKeySnapshot)
	if err != nil {
		return errors.Wrap(err, "failed to create snapshots bucket")
	}
	pbkt, err := bkt.CreateBucketIfNotExists(bucketKeyParents)
	if err != nil {
		return errors.Wrap(err, "failed to create snapshots bucket")
	}
	return fn(ctx, sbkt, pbkt)
}

func parents(bkt, pbkt *bolt.Bucket, parent uint64) (parents []string, err error) {
	for {
		parents = append(parents, fmt.Sprintf("%d", parent))

		parentKey := pbkt.Get(bucketKeyParent)
		if len(parentKey) == 0 {
			return
		}
		pbkt = bkt.Bucket(parentKey)
		if pbkt == nil {
			return nil, errors.Wrap(errdefs.ErrNotFound, "missing parent")
		}

		parent = readID(pbkt)
	}
}

func readKind(bkt *bolt.Bucket) (k snapshot.Kind) {
	kind := bkt.Get(bucketKeyKind)
	if len(kind) == 1 {
		k = snapshot.Kind(kind[0])
	}
	return
}

func readID(bkt *bolt.Bucket) uint64 {
	id, _ := binary.Uvarint(bkt.Get(bucketKeyID))
	return id
}

func readSnapshot(bkt *bolt.Bucket, id *uint64, si *snapshot.Info) error {
	if id != nil {
		*id = readID(bkt)
	}
	if si != nil {
		si.Kind = readKind(bkt)
		si.Parent = string(bkt.Get(bucketKeyParent))

		if err := boltutil.ReadTimestamps(bkt, &si.Created, &si.Updated); err != nil {
			return err
		}

		labels, err := boltutil.ReadLabels(bkt)
		if err != nil {
			return err
		}
		si.Labels = labels
	}

	return nil
}

func putSnapshot(bkt *bolt.Bucket, id uint64, si snapshot.Info) error {
	idEncoded, err := encodeID(id)
	if err != nil {
		return err
	}

	updates := [][2][]byte{
		{bucketKeyID, idEncoded},
		{bucketKeyKind, []byte{byte(si.Kind)}},
	}
	if si.Parent != "" {
		updates = append(updates, [2][]byte{bucketKeyParent, []byte(si.Parent)})
	}
	for _, v := range updates {
		if err := bkt.Put(v[0], v[1]); err != nil {
			return err
		}
	}
	if err := boltutil.WriteTimestamps(bkt, si.Created, si.Updated); err != nil {
		return err
	}
	return boltutil.WriteLabels(bkt, si.Labels)
}

func getUsage(bkt *bolt.Bucket, usage *snapshot.Usage) {
	usage.Inodes, _ = binary.Varint(bkt.Get(bucketKeyInodes))
	usage.Size, _ = binary.Varint(bkt.Get(bucketKeySize))
}

func putUsage(bkt *bolt.Bucket, usage snapshot.Usage) error {
	for _, v := range []struct {
		key   []byte
		value int64
	}{
		{bucketKeyInodes, usage.Inodes},
		{bucketKeySize, usage.Size},
	} {
		e, err := encodeSize(v.value)
		if err != nil {
			return err
		}
		if err := bkt.Put(v.key, e); err != nil {
			return err
		}
	}
	return nil
}

func encodeSize(size int64) ([]byte, error) {
	var (
		buf         [binary.MaxVarintLen64]byte
		sizeEncoded = buf[:]
	)
	sizeEncoded = sizeEncoded[:binary.PutVarint(sizeEncoded, size)]

	if len(sizeEncoded) == 0 {
		return nil, fmt.Errorf("failed encoding size = %v", size)
	}
	return sizeEncoded, nil
}

func encodeID(id uint64) ([]byte, error) {
	var (
		buf       [binary.MaxVarintLen64]byte
		idEncoded = buf[:]
	)
	idEncoded = idEncoded[:binary.PutUvarint(idEncoded, id)]

	if len(idEncoded) == 0 {
		return nil, fmt.Errorf("failed encoding id = %v", id)
	}
	return idEncoded, nil
}
