// Package storage provides a metadata storage implementation for snapshot
// drivers. Drive implementations are responsible for starting and managing
// transactions using the defined context creator. This storage package uses
// BoltDB for storing metadata. Access to the raw boltdb transaction is not
// provided, but the stored object is provided by the proto subpackage.
package storage

import (
	"context"
	"sync"

	"github.com/boltdb/bolt"
	"github.com/containerd/containerd/snapshot"
	"github.com/pkg/errors"
)

// Transactor is used to finalize an active transaction.
type Transactor interface {
	// Commit commits any changes made during the transaction. On error a
	// caller is expected to clean up any resources which would have relied
	// on data mutated as part of this transaction. Only writable
	// transactions can commit, non-writable must call Rollback.
	Commit() error

	// Rollback rolls back any changes made during the transaction. This
	// must be called on all non-writable transactions and aborted writable
	// transaction.
	Rollback() error
}

// Snapshot hold the metadata for an active or view snapshot transaction. The
// ParentIDs hold the snapshot identifiers for the committed snapshots this
// active or view is based on. The ParentIDs are ordered from the lowest base
// to highest, meaning they should be applied in order from the first index to
// the last index. The last index should always be considered the active
// snapshots immediate parent.
type Snapshot struct {
	Kind      snapshot.Kind
	ID        string
	ParentIDs []string
}

// MetaStore is used to store metadata related to a snapshot driver. The
// MetaStore is intended to store metadata related to name, state and
// parentage. Using the MetaStore is not required to implement a snapshot
// driver but can be used to handle the persistence and transactional
// complexities of a driver implementation.
type MetaStore struct {
	dbfile string

	dbL sync.Mutex
	db  *bolt.DB
}

// NewMetaStore returns a snapshot MetaStore for storage of metadata related to
// a snapshot driver backed by a bolt file database. This implementation is
// strongly consistent and does all metadata changes in a transaction to prevent
// against process crashes causing inconsistent metadata state.
func NewMetaStore(dbfile string) (*MetaStore, error) {
	return &MetaStore{
		dbfile: dbfile,
	}, nil
}

type transactionKey struct{}

// TransactionContext creates a new transaction context. The writable value
// should be set to true for transactions which are expected to mutate data.
func (ms *MetaStore) TransactionContext(ctx context.Context, writable bool) (context.Context, Transactor, error) {
	ms.dbL.Lock()
	if ms.db == nil {
		db, err := bolt.Open(ms.dbfile, 0600, nil)
		if err != nil {
			ms.dbL.Unlock()
			return ctx, nil, errors.Wrap(err, "failed to open database file")
		}
		ms.db = db
	}
	ms.dbL.Unlock()

	tx, err := ms.db.Begin(writable)
	if err != nil {
		return ctx, nil, errors.Wrap(err, "failed to start transaction")
	}

	ctx = context.WithValue(ctx, transactionKey{}, tx)

	return ctx, tx, nil
}

// Close closes the metastore and any underlying database connections
func (ms *MetaStore) Close() error {
	ms.dbL.Lock()
	defer ms.dbL.Unlock()
	if ms.db == nil {
		return nil
	}
	return ms.db.Close()
}
