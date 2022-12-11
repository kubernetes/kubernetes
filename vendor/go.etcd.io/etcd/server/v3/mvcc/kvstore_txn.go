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
	"context"
	"fmt"

	"go.etcd.io/etcd/api/v3/mvccpb"
	"go.etcd.io/etcd/pkg/v3/traceutil"
	"go.etcd.io/etcd/server/v3/lease"
	"go.etcd.io/etcd/server/v3/mvcc/backend"
	"go.etcd.io/etcd/server/v3/mvcc/buckets"
	"go.uber.org/zap"
)

type storeTxnRead struct {
	s  *store
	tx backend.ReadTx

	firstRev int64
	rev      int64

	trace *traceutil.Trace
}

func (s *store) Read(mode ReadTxMode, trace *traceutil.Trace) TxnRead {
	s.mu.RLock()
	s.revMu.RLock()
	// For read-only workloads, we use shared buffer by copying transaction read buffer
	// for higher concurrency with ongoing blocking writes.
	// For write/write-read transactions, we use the shared buffer
	// rather than duplicating transaction read buffer to avoid transaction overhead.
	var tx backend.ReadTx
	if mode == ConcurrentReadTxMode {
		tx = s.b.ConcurrentReadTx()
	} else {
		tx = s.b.ReadTx()
	}

	tx.RLock() // RLock is no-op. concurrentReadTx does not need to be locked after it is created.
	firstRev, rev := s.compactMainRev, s.currentRev
	s.revMu.RUnlock()
	return newMetricsTxnRead(&storeTxnRead{s, tx, firstRev, rev, trace})
}

func (tr *storeTxnRead) FirstRev() int64 { return tr.firstRev }
func (tr *storeTxnRead) Rev() int64      { return tr.rev }

func (tr *storeTxnRead) Range(ctx context.Context, key, end []byte, ro RangeOptions) (r *RangeResult, err error) {
	return tr.rangeKeys(ctx, key, end, tr.Rev(), ro)
}

func (tr *storeTxnRead) End() {
	tr.tx.RUnlock() // RUnlock signals the end of concurrentReadTx.
	tr.s.mu.RUnlock()
}

type storeTxnWrite struct {
	storeTxnRead
	tx backend.BatchTx
	// beginRev is the revision where the txn begins; it will write to the next revision.
	beginRev int64
	changes  []mvccpb.KeyValue
}

func (s *store) Write(trace *traceutil.Trace) TxnWrite {
	s.mu.RLock()
	tx := s.b.BatchTx()
	tx.LockInsideApply()
	tw := &storeTxnWrite{
		storeTxnRead: storeTxnRead{s, tx, 0, 0, trace},
		tx:           tx,
		beginRev:     s.currentRev,
		changes:      make([]mvccpb.KeyValue, 0, 4),
	}
	return newMetricsTxnWrite(tw)
}

func (tw *storeTxnWrite) Rev() int64 { return tw.beginRev }

func (tw *storeTxnWrite) Range(ctx context.Context, key, end []byte, ro RangeOptions) (r *RangeResult, err error) {
	rev := tw.beginRev
	if len(tw.changes) > 0 {
		rev++
	}
	return tw.rangeKeys(ctx, key, end, rev, ro)
}

func (tw *storeTxnWrite) DeleteRange(key, end []byte) (int64, int64) {
	if n := tw.deleteRange(key, end); n != 0 || len(tw.changes) > 0 {
		return n, tw.beginRev + 1
	}
	return 0, tw.beginRev
}

func (tw *storeTxnWrite) Put(key, value []byte, lease lease.LeaseID) int64 {
	tw.put(key, value, lease)
	return tw.beginRev + 1
}

func (tw *storeTxnWrite) End() {
	// only update index if the txn modifies the mvcc state.
	if len(tw.changes) != 0 {
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

func (tr *storeTxnRead) rangeKeys(ctx context.Context, key, end []byte, curRev int64, ro RangeOptions) (*RangeResult, error) {
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
	if ro.Count {
		total := tr.s.kvindex.CountRevisions(key, end, rev)
		tr.trace.Step("count revisions from in-memory index tree")
		return &RangeResult{KVs: nil, Count: total, Rev: curRev}, nil
	}
	revpairs, total := tr.s.kvindex.Revisions(key, end, rev, int(ro.Limit))
	tr.trace.Step("range keys from in-memory index tree")
	if len(revpairs) == 0 {
		return &RangeResult{KVs: nil, Count: total, Rev: curRev}, nil
	}

	limit := int(ro.Limit)
	if limit <= 0 || limit > len(revpairs) {
		limit = len(revpairs)
	}

	kvs := make([]mvccpb.KeyValue, limit)
	revBytes := newRevBytes()
	for i, revpair := range revpairs[:len(kvs)] {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("rangeKeys: context cancelled: %w", ctx.Err())
		default:
		}
		revToBytes(revpair, revBytes)
		_, vs := tr.tx.UnsafeRange(buckets.Key, revBytes, nil, 0)
		if len(vs) != 1 {
			tr.s.lg.Fatal(
				"range failed to find revision pair",
				zap.Int64("revision-main", revpair.main),
				zap.Int64("revision-sub", revpair.sub),
				zap.Int64("revision-current", curRev),
				zap.Int64("range-option-rev", ro.Rev),
				zap.Int64("range-option-limit", ro.Limit),
				zap.Binary("key", key),
				zap.Binary("end", end),
				zap.Int("len-revpairs", len(revpairs)),
				zap.Int("len-values", len(vs)),
			)
		}
		if err := kvs[i].Unmarshal(vs[0]); err != nil {
			tr.s.lg.Fatal(
				"failed to unmarshal mvccpb.KeyValue",
				zap.Error(err),
			)
		}
	}
	tr.trace.Step("range keys from bolt db")
	return &RangeResult{KVs: kvs, Count: total, Rev: curRev}, nil
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
		tw.trace.Step("get key's previous created_revision and leaseID")
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
		tw.storeTxnRead.s.lg.Fatal(
			"failed to marshal mvccpb.KeyValue",
			zap.Error(err),
		)
	}

	tw.trace.Step("marshal mvccpb.KeyValue")
	tw.tx.UnsafeSeqPut(buckets.Key, ibytes, d)
	tw.s.kvindex.Put(key, idxRev)
	tw.changes = append(tw.changes, kv)
	tw.trace.Step("store kv pair into bolt db")

	if oldLease != lease.NoLease {
		if tw.s.le == nil {
			panic("no lessor to detach lease")
		}
		err = tw.s.le.Detach(oldLease, []lease.LeaseItem{{Key: string(key)}})
		if err != nil {
			tw.storeTxnRead.s.lg.Error(
				"failed to detach old lease from a key",
				zap.Error(err),
			)
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
	tw.trace.Step("attach lease to kv pair")
}

func (tw *storeTxnWrite) deleteRange(key, end []byte) int64 {
	rrev := tw.beginRev
	if len(tw.changes) > 0 {
		rrev++
	}
	keys, _ := tw.s.kvindex.Range(key, end, rrev)
	if len(keys) == 0 {
		return 0
	}
	for _, key := range keys {
		tw.delete(key)
	}
	return int64(len(keys))
}

func (tw *storeTxnWrite) delete(key []byte) {
	ibytes := newRevBytes()
	idxRev := revision{main: tw.beginRev + 1, sub: int64(len(tw.changes))}
	revToBytes(idxRev, ibytes)

	ibytes = appendMarkTombstone(tw.storeTxnRead.s.lg, ibytes)

	kv := mvccpb.KeyValue{Key: key}

	d, err := kv.Marshal()
	if err != nil {
		tw.storeTxnRead.s.lg.Fatal(
			"failed to marshal mvccpb.KeyValue",
			zap.Error(err),
		)
	}

	tw.tx.UnsafeSeqPut(buckets.Key, ibytes, d)
	err = tw.s.kvindex.Tombstone(key, idxRev)
	if err != nil {
		tw.storeTxnRead.s.lg.Fatal(
			"failed to tombstone an existing key",
			zap.String("key", string(key)),
			zap.Error(err),
		)
	}
	tw.changes = append(tw.changes, kv)

	item := lease.LeaseItem{Key: string(key)}
	leaseID := tw.s.le.GetLease(item)

	if leaseID != lease.NoLease {
		err = tw.s.le.Detach(leaseID, []lease.LeaseItem{item})
		if err != nil {
			tw.storeTxnRead.s.lg.Error(
				"failed to detach old lease from a key",
				zap.Error(err),
			)
		}
	}
}

func (tw *storeTxnWrite) Changes() []mvccpb.KeyValue { return tw.changes }
