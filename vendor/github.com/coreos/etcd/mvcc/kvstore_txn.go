// Copyright 2017 The etcd Authors
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

type storeTxnRead struct {
	s  *store
	tx backend.ReadTx

	firstRev int64
	rev      int64
}

func (s *store) Read() TxnRead {
	s.mu.RLock()
	tx := s.b.ReadTx()
	s.revMu.RLock()
	tx.Lock()
	firstRev, rev := s.compactMainRev, s.currentRev
	s.revMu.RUnlock()
	return newMetricsTxnRead(&storeTxnRead{s, tx, firstRev, rev})
}

func (tr *storeTxnRead) FirstRev() int64 { return tr.firstRev }
func (tr *storeTxnRead) Rev() int64      { return tr.rev }

func (tr *storeTxnRead) Range(key, end []byte, ro RangeOptions) (r *RangeResult, err error) {
	return tr.rangeKeys(key, end, tr.Rev(), ro)
}

func (tr *storeTxnRead) End() {
	tr.tx.Unlock()
	tr.s.mu.RUnlock()
}

type storeTxnWrite struct {
	storeTxnRead
	tx backend.BatchTx
	// beginRev is the revision where the txn begins; it will write to the next revision.
	beginRev int64
	changes  []mvccpb.KeyValue
}

func (s *store) Write() TxnWrite {
	s.mu.RLock()
	tx := s.b.BatchTx()
	tx.Lock()
	tw := &storeTxnWrite{
		storeTxnRead: storeTxnRead{s, tx, 0, 0},
		tx:           tx,
		beginRev:     s.currentRev,
		changes:      make([]mvccpb.KeyValue, 0, 4),
	}
	return newMetricsTxnWrite(tw)
}

func (tw *storeTxnWrite) Rev() int64 { return tw.beginRev }

func (tw *storeTxnWrite) Range(key, end []byte, ro RangeOptions) (r *RangeResult, err error) {
	rev := tw.beginRev
	if len(tw.changes) > 0 {
		rev++
	}
	return tw.rangeKeys(key, end, rev, ro)
}

func (tw *storeTxnWrite) DeleteRange(key, end []byte) (int64, int64) {
	if n := tw.deleteRange(key, end); n != 0 || len(tw.changes) > 0 {
		return n, int64(tw.beginRev + 1)
	}
	return 0, int64(tw.beginRev)
}

func (tw *storeTxnWrite) Put(key, value []byte, lease lease.LeaseID) int64 {
	tw.put(key, value, lease)
	return int64(tw.beginRev + 1)
}

func (tw *storeTxnWrite) End() {
	// only update index if the txn modifies the mvcc state.
	if len(tw.changes) != 0 {
		tw.s.saveIndex(tw.tx)
		// hold revMu lock to prevent new read txns from opening until writeback.
		tw.s.revMu.Lock()
		tw.s.currentRev++
	}
	tw.tx.Unlock()
	if len(tw.changes) != 0 {
		tw.s.revMu.Unlock()
	}
	tw.s.mu.RUnlock()
}

func (tr *storeTxnRead) rangeKeys(key, end []byte, curRev int64, ro RangeOptions) (*RangeResult, error) {
	rev := ro.Rev
	if rev > curRev {
		return &RangeResult{KVs: nil, Count: -1, Rev: curRev}, ErrFutureRev
	}
	if rev <= 0 {
		rev = curRev
	}
	if rev < tr.s.compactMainRev {
		return &RangeResult{KVs: nil, Count: -1, Rev: 0}, ErrCompacted
	}

	revpairs := tr.s.kvindex.Revisions(key, end, int64(rev))
	if len(revpairs) == 0 {
		return &RangeResult{KVs: nil, Count: 0, Rev: curRev}, nil
	}
	if ro.Count {
		return &RangeResult{KVs: nil, Count: len(revpairs), Rev: curRev}, nil
	}

	limit := int(ro.Limit)
	if limit <= 0 || limit > len(revpairs) {
		limit = len(revpairs)
	}

	kvs := make([]mvccpb.KeyValue, limit)
	revBytes := newRevBytes()
	for i, revpair := range revpairs[:len(kvs)] {
		revToBytes(revpair, revBytes)
		_, vs := tr.tx.UnsafeRange(keyBucketName, revBytes, nil, 0)
		if len(vs) != 1 {
			plog.Fatalf("range cannot find rev (%d,%d)", revpair.main, revpair.sub)
		}
		if err := kvs[i].Unmarshal(vs[0]); err != nil {
			plog.Fatalf("cannot unmarshal event: %v", err)
		}
	}
	return &RangeResult{KVs: kvs, Count: len(revpairs), Rev: curRev}, nil
}

func (tw *storeTxnWrite) put(key, value []byte, leaseID lease.LeaseID) {
	rev := tw.beginRev + 1
	c := rev
	oldLease := lease.NoLease

	// if the key exists before, use its previous created and
	// get its previous leaseID
	_, created, ver, err := tw.s.kvindex.Get(key, rev)
	if err == nil {
		c = created.main
		oldLease = tw.s.le.GetLease(lease.LeaseItem{Key: string(key)})
	}

	ibytes := newRevBytes()
	idxRev := revision{main: rev, sub: int64(len(tw.changes))}
	revToBytes(idxRev, ibytes)

	ver = ver + 1
	kv := mvccpb.KeyValue{
		Key:            key,
		Value:          value,
		CreateRevision: c,
		ModRevision:    rev,
		Version:        ver,
		Lease:          int64(leaseID),
	}

	d, err := kv.Marshal()
	if err != nil {
		plog.Fatalf("cannot marshal event: %v", err)
	}

	tw.tx.UnsafeSeqPut(keyBucketName, ibytes, d)
	tw.s.kvindex.Put(key, idxRev)
	tw.changes = append(tw.changes, kv)

	if oldLease != lease.NoLease {
		if tw.s.le == nil {
			panic("no lessor to detach lease")
		}
		err = tw.s.le.Detach(oldLease, []lease.LeaseItem{{Key: string(key)}})
		if err != nil {
			plog.Errorf("unexpected error from lease detach: %v", err)
		}
	}
	if leaseID != lease.NoLease {
		if tw.s.le == nil {
			panic("no lessor to attach lease")
		}
		err = tw.s.le.Attach(leaseID, []lease.LeaseItem{{Key: string(key)}})
		if err != nil {
			panic("unexpected error from lease Attach")
		}
	}
}

func (tw *storeTxnWrite) deleteRange(key, end []byte) int64 {
	rrev := tw.beginRev
	if len(tw.changes) > 0 {
		rrev += 1
	}
	keys, revs := tw.s.kvindex.Range(key, end, rrev)
	if len(keys) == 0 {
		return 0
	}
	for i, key := range keys {
		tw.delete(key, revs[i])
	}
	return int64(len(keys))
}

func (tw *storeTxnWrite) delete(key []byte, rev revision) {
	ibytes := newRevBytes()
	idxRev := revision{main: tw.beginRev + 1, sub: int64(len(tw.changes))}
	revToBytes(idxRev, ibytes)
	ibytes = appendMarkTombstone(ibytes)

	kv := mvccpb.KeyValue{Key: key}

	d, err := kv.Marshal()
	if err != nil {
		plog.Fatalf("cannot marshal event: %v", err)
	}

	tw.tx.UnsafeSeqPut(keyBucketName, ibytes, d)
	err = tw.s.kvindex.Tombstone(key, idxRev)
	if err != nil {
		plog.Fatalf("cannot tombstone an existing key (%s): %v", string(key), err)
	}
	tw.changes = append(tw.changes, kv)

	item := lease.LeaseItem{Key: string(key)}
	leaseID := tw.s.le.GetLease(item)

	if leaseID != lease.NoLease {
		err = tw.s.le.Detach(leaseID, []lease.LeaseItem{item})
		if err != nil {
			plog.Errorf("cannot detach %v", err)
		}
	}
}

func (tw *storeTxnWrite) Changes() []mvccpb.KeyValue { return tw.changes }
