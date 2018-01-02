package metadata

import (
	"context"
	"time"

	"github.com/boltdb/bolt"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/leases"
	"github.com/containerd/containerd/metadata/boltutil"
	"github.com/containerd/containerd/namespaces"
	digest "github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

// Lease retains resources to prevent garbage collection before
// the resources can be fully referenced.
type Lease struct {
	ID        string
	CreatedAt time.Time
	Labels    map[string]string

	Content   []string
	Snapshots map[string][]string
}

// LeaseManager manages the create/delete lifecyle of leases
// and also returns existing leases
type LeaseManager struct {
	tx *bolt.Tx
}

// NewLeaseManager creates a new lease manager for managing leases using
// the provided database transaction.
func NewLeaseManager(tx *bolt.Tx) *LeaseManager {
	return &LeaseManager{
		tx: tx,
	}
}

// Create creates a new lease using the provided lease
func (lm *LeaseManager) Create(ctx context.Context, lid string, labels map[string]string) (Lease, error) {
	namespace, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return Lease{}, err
	}

	topbkt, err := createBucketIfNotExists(lm.tx, bucketKeyVersion, []byte(namespace), bucketKeyObjectLeases)
	if err != nil {
		return Lease{}, err
	}

	txbkt, err := topbkt.CreateBucket([]byte(lid))
	if err != nil {
		if err == bolt.ErrBucketExists {
			err = errdefs.ErrAlreadyExists
		}
		return Lease{}, err
	}

	t := time.Now().UTC()
	createdAt, err := t.MarshalBinary()
	if err != nil {
		return Lease{}, err
	}
	if err := txbkt.Put(bucketKeyCreatedAt, createdAt); err != nil {
		return Lease{}, err
	}

	if labels != nil {
		if err := boltutil.WriteLabels(txbkt, labels); err != nil {
			return Lease{}, err
		}
	}

	return Lease{
		ID:        lid,
		CreatedAt: t,
		Labels:    labels,
	}, nil
}

// Delete delets the lease with the provided lease ID
func (lm *LeaseManager) Delete(ctx context.Context, lid string) error {
	namespace, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return err
	}

	topbkt := getBucket(lm.tx, bucketKeyVersion, []byte(namespace), bucketKeyObjectLeases)
	if topbkt == nil {
		return nil
	}
	if err := topbkt.DeleteBucket([]byte(lid)); err != nil && err != bolt.ErrBucketNotFound {
		return err
	}
	return nil
}

// List lists all active leases
func (lm *LeaseManager) List(ctx context.Context, includeResources bool, filter ...string) ([]Lease, error) {
	namespace, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return nil, err
	}

	var leases []Lease

	topbkt := getBucket(lm.tx, bucketKeyVersion, []byte(namespace), bucketKeyObjectLeases)
	if topbkt == nil {
		return leases, nil
	}

	if err := topbkt.ForEach(func(k, v []byte) error {
		if v != nil {
			return nil
		}
		txbkt := topbkt.Bucket(k)

		l := Lease{
			ID: string(k),
		}

		if v := txbkt.Get(bucketKeyCreatedAt); v != nil {
			t := &l.CreatedAt
			if err := t.UnmarshalBinary(v); err != nil {
				return err
			}
		}

		labels, err := boltutil.ReadLabels(txbkt)
		if err != nil {
			return err
		}
		l.Labels = labels

		// TODO: Read Snapshots
		// TODO: Read Content

		leases = append(leases, l)

		return nil
	}); err != nil {
		return nil, err
	}

	return leases, nil
}

func addSnapshotLease(ctx context.Context, tx *bolt.Tx, snapshotter, key string) error {
	lid, ok := leases.Lease(ctx)
	if !ok {
		return nil
	}

	namespace, ok := namespaces.Namespace(ctx)
	if !ok {
		panic("namespace must already be required")
	}

	bkt := getBucket(tx, bucketKeyVersion, []byte(namespace), bucketKeyObjectLeases, []byte(lid))
	if bkt == nil {
		return errors.Wrap(errdefs.ErrNotFound, "lease does not exist")
	}

	bkt, err := bkt.CreateBucketIfNotExists(bucketKeyObjectSnapshots)
	if err != nil {
		return err
	}

	bkt, err = bkt.CreateBucketIfNotExists([]byte(snapshotter))
	if err != nil {
		return err
	}

	return bkt.Put([]byte(key), nil)
}

func addContentLease(ctx context.Context, tx *bolt.Tx, dgst digest.Digest) error {
	lid, ok := leases.Lease(ctx)
	if !ok {
		return nil
	}

	namespace, ok := namespaces.Namespace(ctx)
	if !ok {
		panic("namespace must already be required")
	}

	bkt := getBucket(tx, bucketKeyVersion, []byte(namespace), bucketKeyObjectLeases, []byte(lid))
	if bkt == nil {
		return errors.Wrap(errdefs.ErrNotFound, "lease does not exist")
	}

	bkt, err := bkt.CreateBucketIfNotExists(bucketKeyObjectContent)
	if err != nil {
		return err
	}

	return bkt.Put([]byte(dgst.String()), nil)
}
