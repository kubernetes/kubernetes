// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package mvcc

import (
	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/mvcc/backend"
	"github.com/coreos/etcd/mvcc/mvccpb"
)

type RangeOptions struct {
	Limit int64
	Rev   int64
	Count bool
}

type RangeResult struct {
	KVs   []mvccpb.KeyValue
	Rev   int64
	Count int
}

type KV interface {
	// Rev returns the current revision of the KV.
	Rev() int64

	// FirstRev returns the first revision of the KV.
	// After a compaction, the first revision increases to the compaction
	// revision.
	FirstRev() int64

	// Range gets the keys in the range at rangeRev.
	// The returned rev is the current revision of the KV when the operation is executed.
	// If rangeRev <=0, range gets the keys at currentRev.
	// If `end` is nil, the request returns the key.
	// If `end` is not nil and not empty, it gets the keys in range [key, range_end).
	// If `end` is not nil and empty, it gets the keys greater than or equal to key.
	// Limit limits the number of keys returned.
	// If the required rev is compacted, ErrCompacted will be returned.
	Range(key, end []byte, ro RangeOptions) (r *RangeResult, err error)

	// Put puts the given key, value into the store. Put also takes additional argument lease to
	// attach a lease to a key-value pair as meta-data. KV implementation does not validate the lease
	// id.
	// A put also increases the rev of the store, and generates one event in the event history.
	// The returned rev is the current revision of the KV when the operation is executed.
	Put(key, value []byte, lease lease.LeaseID) (rev int64)

	// DeleteRange deletes the given range from the store.
	// A deleteRange increases the rev of the store if any key in the range exists.
	// The number of key deleted will be returned.
	// The returned rev is the current revision of the KV when the operation is executed.
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
	// TxnRange returns the current revision of the KV when the operation is executed.
	TxnRange(txnID int64, key, end []byte, ro RangeOptions) (r *RangeResult, err error)
	TxnPut(txnID int64, key, value []byte, lease lease.LeaseID) (rev int64, err error)
	TxnDeleteRange(txnID int64, key, end []byte) (n, rev int64, err error)

	// Compact frees all superseded keys with revisions less than rev.
	Compact(rev int64) (<-chan struct{}, error)

	// Hash retrieves the hash of KV state and revision.
	// This method is designed for consistency checking purpose.
	Hash() (hash uint32, revision int64, err error)

	// Commit commits txns into the underlying backend.
	Commit()

	// Restore restores the KV store from a backend.
	Restore(b backend.Backend) error
	Close() error
}

// WatchableKV is a KV that can be watched.
type WatchableKV interface {
	KV
	Watchable
}

// Watchable is the interface that wraps the NewWatchStream function.
type Watchable interface {
	// NewWatchStream returns a WatchStream that can be used to
	// watch events happened or happening on the KV.
	NewWatchStream() WatchStream
}

// ConsistentWatchableKV is a WatchableKV that understands the consistency
// algorithm and consistent index.
// If the consistent index of executing entry is not larger than the
// consistent index of ConsistentWatchableKV, all operations in
// this entry are skipped and return empty response.
type ConsistentWatchableKV interface {
	WatchableKV
	// ConsistentIndex returns the current consistent index of the KV.
	ConsistentIndex() uint64
}
