package storage

import (
	"io"

	"github.com/coreos/etcd/storage/storagepb"
)

type KV interface {
	// Range gets the keys in the range at rangeRev.
	// If rangeRev <=0, range gets the keys at currentRev.
	// If `end` is nil, the request returns the key.
	// If `end` is not nil, it gets the keys in range [key, range_end).
	// Limit limits the number of keys returned.
	// If the required rev is compacted, ErrCompacted will be returned.
	Range(key, end []byte, limit, rangeRev int64) (kvs []storagepb.KeyValue, rev int64, err error)

	// Put puts the given key,value into the store.
	// A put also increases the rev of the store, and generates one event in the event history.
	Put(key, value []byte) (rev int64)

	// DeleteRange deletes the given range from the store.
	// A deleteRange increases the rev of the store if any key in the range exists.
	// The number of key deleted will be returned.
	// It also generates one event for each key delete in the event history.
	// if the `end` is nil, deleteRange deletes the key.
	// if the `end` is not nil, deleteRange deletes the keys in range [key, range_end).
	DeleteRange(key, end []byte) (n, rev int64)

	// TxnBegin begins a txn. Only Txn prefixed operation can be executed, others will be blocked
	// until txn ends. Only one on-going txn is allowed.
	// TxnBegin returns an int64 txn ID.
	// All txn prefixed operations with same txn ID will be done with the same rev.
	TxnBegin() int64
	// TxnEnd ends the on-going txn with txn ID. If the on-going txn ID is not matched, error is returned.
	TxnEnd(txnID int64) error
	TxnRange(txnID int64, key, end []byte, limit, rangeRev int64) (kvs []storagepb.KeyValue, rev int64, err error)
	TxnPut(txnID int64, key, value []byte) (rev int64, err error)
	TxnDeleteRange(txnID int64, key, end []byte) (n, rev int64, err error)

	Compact(rev int64) error

	// Write a snapshot to the given io writer
	Snapshot(w io.Writer) (int64, error)

	Restore() error
	Close() error
}
