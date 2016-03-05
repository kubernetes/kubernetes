// Copyright 2015 CoreOS, Inc.
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

package storage

import (
	"encoding/binary"
	"log"

	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/storage/backend"
	"github.com/coreos/etcd/storage/storagepb"
)

var (
	consistentIndexKeyName = []byte("consistent_index")
)

// ConsistentIndexGetter is an interface that wraps the Get method.
// Consistent index is the offset of an entry in a consistent replicated log.
type ConsistentIndexGetter interface {
	// ConsistentIndex returns the consistent index of current executing entry.
	ConsistentIndex() uint64
}

type consistentWatchableStore struct {
	*watchableStore
	// The field is used to get the consistent index of current
	// executing entry.
	// When the store finishes executing current entry, it will
	// put the index got from ConsistentIndexGetter into the
	// underlying backend. This helps to recover consistent index
	// when restoring.
	ig ConsistentIndexGetter

	skip bool // indicate whether or not to skip an operation
}

func New(b backend.Backend, le lease.Lessor, ig ConsistentIndexGetter) ConsistentWatchableKV {
	return newConsistentWatchableStore(b, le, ig)
}

// newConsistentWatchableStore creates a new consistentWatchableStore with the give
// backend.
func newConsistentWatchableStore(b backend.Backend, le lease.Lessor, ig ConsistentIndexGetter) *consistentWatchableStore {
	return &consistentWatchableStore{
		watchableStore: newWatchableStore(b, le),
		ig:             ig,
	}
}

func (s *consistentWatchableStore) Put(key, value []byte, lease lease.LeaseID) (rev int64) {
	id := s.TxnBegin()
	rev, err := s.TxnPut(id, key, value, lease)
	if err != nil {
		log.Panicf("unexpected TxnPut error (%v)", err)
	}
	if err := s.TxnEnd(id); err != nil {
		log.Panicf("unexpected TxnEnd error (%v)", err)
	}
	return rev
}

func (s *consistentWatchableStore) DeleteRange(key, end []byte) (n, rev int64) {
	id := s.TxnBegin()
	n, rev, err := s.TxnDeleteRange(id, key, end)
	if err != nil {
		log.Panicf("unexpected TxnDeleteRange error (%v)", err)
	}
	if err := s.TxnEnd(id); err != nil {
		log.Panicf("unexpected TxnEnd error (%v)", err)
	}
	return n, rev
}

func (s *consistentWatchableStore) TxnBegin() int64 {
	id := s.watchableStore.TxnBegin()

	// If the consistent index of executing entry is not larger than store
	// consistent index, skip all operations in this txn.
	s.skip = s.ig.ConsistentIndex() <= s.consistentIndex()

	if !s.skip {
		// TODO: avoid this unnecessary allocation
		bs := make([]byte, 8)
		binary.BigEndian.PutUint64(bs, s.ig.ConsistentIndex())
		// put the index into the underlying backend
		// tx has been locked in TxnBegin, so there is no need to lock it again
		s.watchableStore.store.tx.UnsafePut(metaBucketName, consistentIndexKeyName, bs)
	}

	return id
}

func (s *consistentWatchableStore) TxnRange(txnID int64, key, end []byte, limit, rangeRev int64) (kvs []storagepb.KeyValue, rev int64, err error) {
	if s.skip {
		return nil, 0, nil
	}
	return s.watchableStore.TxnRange(txnID, key, end, limit, rangeRev)
}

func (s *consistentWatchableStore) TxnPut(txnID int64, key, value []byte, lease lease.LeaseID) (rev int64, err error) {
	if s.skip {
		return 0, nil
	}
	return s.watchableStore.TxnPut(txnID, key, value, lease)
}

func (s *consistentWatchableStore) TxnDeleteRange(txnID int64, key, end []byte) (n, rev int64, err error) {
	if s.skip {
		return 0, 0, nil
	}
	return s.watchableStore.TxnDeleteRange(txnID, key, end)
}

func (s *consistentWatchableStore) TxnEnd(txnID int64) error {
	// reset skip var
	s.skip = false
	return s.watchableStore.TxnEnd(txnID)
}

func (s *consistentWatchableStore) consistentIndex() uint64 {
	// get the index
	// tx has been locked in TxnBegin, so there is no need to lock it again
	_, vs := s.watchableStore.store.tx.UnsafeRange(metaBucketName, consistentIndexKeyName, nil, 0)
	if len(vs) == 0 {
		return 0
	}
	return binary.BigEndian.Uint64(vs[0])
}
