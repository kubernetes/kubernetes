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

package cindex

import (
	"encoding/binary"
	"sync"
	"sync/atomic"

	"go.etcd.io/etcd/server/v3/mvcc/backend"
	"go.etcd.io/etcd/server/v3/mvcc/buckets"
)

type Backend interface {
	BatchTx() backend.BatchTx
	ReadTx() backend.ReadTx
}

// ConsistentIndexer is an interface that wraps the Get/Set/Save method for consistentIndex.
type ConsistentIndexer interface {

	// ConsistentIndex returns the consistent index of current executing entry.
	ConsistentIndex() uint64

	// ConsistentApplyingIndex returns the consistent applying index of current executing entry.
	ConsistentApplyingIndex() (uint64, uint64)

	// UnsafeConsistentIndex is similar to ConsistentIndex, but it doesn't lock the transaction.
	UnsafeConsistentIndex() uint64

	// SetConsistentIndex set the consistent index of current executing entry.
	SetConsistentIndex(v uint64, term uint64)

	// SetConsistentApplyingIndex set the consistent applying index of current executing entry.
	SetConsistentApplyingIndex(v uint64, term uint64)

	// UnsafeSave must be called holding the lock on the tx.
	// It saves consistentIndex to the underlying stable storage.
	UnsafeSave(tx backend.BatchTx)

	// SetBackend set the available backend.BatchTx for ConsistentIndexer.
	SetBackend(be Backend)
}

// consistentIndex implements the ConsistentIndexer interface.
type consistentIndex struct {
	// consistentIndex represents the offset of an entry in a consistent replica log.
	// It caches the "consistent_index" key's value.
	// Accessed through atomics so must be 64-bit aligned.
	consistentIndex uint64
	// term represents the RAFT term of committed entry in a consistent replica log.
	// Accessed through atomics so must be 64-bit aligned.
	// The value is being persisted in the backend since v3.5.
	term uint64

	// applyingIndex and applyingTerm are just temporary cache of the raftpb.Entry.Index
	// and raftpb.Entry.Term, and they are not ready to be persisted yet. They will be
	// saved to consistentIndex and term above in the txPostLockInsideApplyHook.
	applyingIndex uint64
	applyingTerm  uint64

	// be is used for initial read consistentIndex
	be Backend
	// mutex is protecting be.
	mutex sync.Mutex
}

// NewConsistentIndex creates a new consistent index.
// If `be` is nil, it must be set (SetBackend) before first access using `ConsistentIndex()`.
func NewConsistentIndex(be Backend) ConsistentIndexer {
	return &consistentIndex{be: be}
}

func (ci *consistentIndex) ConsistentIndex() uint64 {
	if index := atomic.LoadUint64(&ci.consistentIndex); index > 0 {
		return index
	}
	ci.mutex.Lock()
	defer ci.mutex.Unlock()

	v, term := ReadConsistentIndex(ci.be.ReadTx())
	ci.SetConsistentIndex(v, term)
	return v
}

func (ci *consistentIndex) UnsafeConsistentIndex() uint64 {
	if index := atomic.LoadUint64(&ci.consistentIndex); index > 0 {
		return index
	}

	v, term := unsafeReadConsistentIndex(ci.be.ReadTx())
	ci.SetConsistentIndex(v, term)
	return v
}

func (ci *consistentIndex) SetConsistentIndex(v uint64, term uint64) {
	atomic.StoreUint64(&ci.consistentIndex, v)
	atomic.StoreUint64(&ci.term, term)
}

func (ci *consistentIndex) UnsafeSave(tx backend.BatchTx) {
	index := atomic.LoadUint64(&ci.consistentIndex)
	term := atomic.LoadUint64(&ci.term)
	UnsafeUpdateConsistentIndex(tx, index, term)
}

func (ci *consistentIndex) SetBackend(be Backend) {
	ci.mutex.Lock()
	defer ci.mutex.Unlock()
	ci.be = be
	// After the backend is changed, the first access should re-read it.
	ci.SetConsistentIndex(0, 0)
}

func (ci *consistentIndex) ConsistentApplyingIndex() (uint64, uint64) {
	return atomic.LoadUint64(&ci.applyingIndex), atomic.LoadUint64(&ci.applyingTerm)
}

func (ci *consistentIndex) SetConsistentApplyingIndex(v uint64, term uint64) {
	atomic.StoreUint64(&ci.applyingIndex, v)
	atomic.StoreUint64(&ci.applyingTerm, term)
}

func NewFakeConsistentIndex(index uint64) ConsistentIndexer {
	return &fakeConsistentIndex{index: index}
}

type fakeConsistentIndex struct {
	index uint64
	term  uint64
}

func (f *fakeConsistentIndex) ConsistentIndex() uint64 {
	return atomic.LoadUint64(&f.index)
}
func (f *fakeConsistentIndex) ConsistentApplyingIndex() (uint64, uint64) {
	return atomic.LoadUint64(&f.index), atomic.LoadUint64(&f.term)
}
func (f *fakeConsistentIndex) UnsafeConsistentIndex() uint64 {
	return atomic.LoadUint64(&f.index)
}

func (f *fakeConsistentIndex) SetConsistentIndex(index uint64, term uint64) {
	atomic.StoreUint64(&f.index, index)
	atomic.StoreUint64(&f.term, term)
}
func (f *fakeConsistentIndex) SetConsistentApplyingIndex(index uint64, term uint64) {
	atomic.StoreUint64(&f.index, index)
	atomic.StoreUint64(&f.term, term)
}

func (f *fakeConsistentIndex) UnsafeSave(_ backend.BatchTx) {}
func (f *fakeConsistentIndex) SetBackend(_ Backend)         {}

// UnsafeCreateMetaBucket creates the `meta` bucket (if it does not exists yet).
func UnsafeCreateMetaBucket(tx backend.BatchTx) {
	tx.UnsafeCreateBucket(buckets.Meta)
}

// CreateMetaBucket creates the `meta` bucket (if it does not exists yet).
func CreateMetaBucket(tx backend.BatchTx) {
	tx.LockOutsideApply()
	defer tx.Unlock()
	tx.UnsafeCreateBucket(buckets.Meta)
}

// unsafeGetConsistentIndex loads consistent index & term from given transaction.
// returns 0,0 if the data are not found.
// Term is persisted since v3.5.
func unsafeReadConsistentIndex(tx backend.ReadTx) (uint64, uint64) {
	_, vs := tx.UnsafeRange(buckets.Meta, buckets.MetaConsistentIndexKeyName, nil, 0)
	if len(vs) == 0 {
		return 0, 0
	}
	v := binary.BigEndian.Uint64(vs[0])
	_, ts := tx.UnsafeRange(buckets.Meta, buckets.MetaTermKeyName, nil, 0)
	if len(ts) == 0 {
		return v, 0
	}
	t := binary.BigEndian.Uint64(ts[0])
	return v, t
}

// ReadConsistentIndex loads consistent index and term from given transaction.
// returns 0 if the data are not found.
func ReadConsistentIndex(tx backend.ReadTx) (uint64, uint64) {
	tx.Lock()
	defer tx.Unlock()
	return unsafeReadConsistentIndex(tx)
}

func UnsafeUpdateConsistentIndex(tx backend.BatchTx, index uint64, term uint64) {
	if index == 0 {
		// Never save 0 as it means that we didn't loaded the real index yet.
		return
	}

	bs1 := make([]byte, 8)
	binary.BigEndian.PutUint64(bs1, index)
	// put the index into the underlying backend
	// tx has been locked in TxnBegin, so there is no need to lock it again
	tx.UnsafePut(buckets.Meta, buckets.MetaConsistentIndexKeyName, bs1)
	if term > 0 {
		bs2 := make([]byte, 8)
		binary.BigEndian.PutUint64(bs2, term)
		tx.UnsafePut(buckets.Meta, buckets.MetaTermKeyName, bs2)
	}
}

func UpdateConsistentIndex(tx backend.BatchTx, index uint64, term uint64) {
	tx.LockOutsideApply()
	defer tx.Unlock()
	UnsafeUpdateConsistentIndex(tx, index, term)
}
