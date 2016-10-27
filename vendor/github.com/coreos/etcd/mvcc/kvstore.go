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
	"encoding/binary"
	"errors"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/mvcc/backend"
	"github.com/coreos/etcd/mvcc/mvccpb"
	"github.com/coreos/etcd/pkg/schedule"
	"github.com/coreos/pkg/capnslog"
	"golang.org/x/net/context"
)

var (
	keyBucketName  = []byte("key")
	metaBucketName = []byte("meta")

	// markedRevBytesLen is the byte length of marked revision.
	// The first `revBytesLen` bytes represents a normal revision. The last
	// one byte is the mark.
	markedRevBytesLen      = revBytesLen + 1
	markBytePosition       = markedRevBytesLen - 1
	markTombstone     byte = 't'

	consistentIndexKeyName  = []byte("consistent_index")
	scheduledCompactKeyName = []byte("scheduledCompactRev")
	finishedCompactKeyName  = []byte("finishedCompactRev")

	ErrTxnIDMismatch = errors.New("mvcc: txn id mismatch")
	ErrCompacted     = errors.New("mvcc: required revision has been compacted")
	ErrFutureRev     = errors.New("mvcc: required revision is a future revision")
	ErrCanceled      = errors.New("mvcc: watcher is canceled")

	plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "mvcc")
)

// ConsistentIndexGetter is an interface that wraps the Get method.
// Consistent index is the offset of an entry in a consistent replicated log.
type ConsistentIndexGetter interface {
	// ConsistentIndex returns the consistent index of current executing entry.
	ConsistentIndex() uint64
}

type store struct {
	mu sync.Mutex // guards the following

	ig ConsistentIndexGetter

	b       backend.Backend
	kvindex index

	le lease.Lessor

	currentRev revision
	// the main revision of the last compaction
	compactMainRev int64

	tx    backend.BatchTx
	txnID int64 // tracks the current txnID to verify txn operations

	// bytesBuf8 is a byte slice of length 8
	// to avoid a repetitive allocation in saveIndex.
	bytesBuf8 []byte

	changes   []mvccpb.KeyValue
	fifoSched schedule.Scheduler

	stopc chan struct{}
}

// NewStore returns a new store. It is useful to create a store inside
// mvcc pkg. It should only be used for testing externally.
func NewStore(b backend.Backend, le lease.Lessor, ig ConsistentIndexGetter) *store {
	s := &store{
		b:       b,
		ig:      ig,
		kvindex: newTreeIndex(),

		le: le,

		currentRev:     revision{main: 1},
		compactMainRev: -1,

		bytesBuf8: make([]byte, 8, 8),
		fifoSched: schedule.NewFIFOScheduler(),

		stopc: make(chan struct{}),
	}

	if s.le != nil {
		s.le.SetRangeDeleter(s)
	}

	tx := s.b.BatchTx()
	tx.Lock()
	tx.UnsafeCreateBucket(keyBucketName)
	tx.UnsafeCreateBucket(metaBucketName)
	tx.Unlock()
	s.b.ForceCommit()

	if err := s.restore(); err != nil {
		// TODO: return the error instead of panic here?
		panic("failed to recover store from backend")
	}

	return s
}

func (s *store) Rev() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.currentRev.main
}

func (s *store) FirstRev() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.compactMainRev
}

func (s *store) Put(key, value []byte, lease lease.LeaseID) int64 {
	id := s.TxnBegin()
	s.put(key, value, lease)
	s.txnEnd(id)

	putCounter.Inc()

	return int64(s.currentRev.main)
}

func (s *store) Range(key, end []byte, ro RangeOptions) (r *RangeResult, err error) {
	id := s.TxnBegin()
	kvs, count, rev, err := s.rangeKeys(key, end, ro.Limit, ro.Rev, ro.Count)
	s.txnEnd(id)

	rangeCounter.Inc()

	r = &RangeResult{
		KVs:   kvs,
		Count: count,
		Rev:   rev,
	}

	return r, err
}

func (s *store) DeleteRange(key, end []byte) (n, rev int64) {
	id := s.TxnBegin()
	n = s.deleteRange(key, end)
	s.txnEnd(id)

	deleteCounter.Inc()

	return n, int64(s.currentRev.main)
}

func (s *store) TxnBegin() int64 {
	s.mu.Lock()
	s.currentRev.sub = 0
	s.tx = s.b.BatchTx()
	s.tx.Lock()
	s.saveIndex()

	s.txnID = rand.Int63()
	return s.txnID
}

func (s *store) TxnEnd(txnID int64) error {
	err := s.txnEnd(txnID)
	if err != nil {
		return err
	}

	txnCounter.Inc()
	return nil
}

// txnEnd is used for unlocking an internal txn. It does
// not increase the txnCounter.
func (s *store) txnEnd(txnID int64) error {
	if txnID != s.txnID {
		return ErrTxnIDMismatch
	}

	s.tx.Unlock()
	if s.currentRev.sub != 0 {
		s.currentRev.main += 1
	}
	s.currentRev.sub = 0

	dbTotalSize.Set(float64(s.b.Size()))
	s.mu.Unlock()
	return nil
}

func (s *store) TxnRange(txnID int64, key, end []byte, ro RangeOptions) (r *RangeResult, err error) {
	if txnID != s.txnID {
		return nil, ErrTxnIDMismatch
	}

	kvs, count, rev, err := s.rangeKeys(key, end, ro.Limit, ro.Rev, ro.Count)

	r = &RangeResult{
		KVs:   kvs,
		Count: count,
		Rev:   rev,
	}
	return r, err
}

func (s *store) TxnPut(txnID int64, key, value []byte, lease lease.LeaseID) (rev int64, err error) {
	if txnID != s.txnID {
		return 0, ErrTxnIDMismatch
	}

	s.put(key, value, lease)
	return int64(s.currentRev.main + 1), nil
}

func (s *store) TxnDeleteRange(txnID int64, key, end []byte) (n, rev int64, err error) {
	if txnID != s.txnID {
		return 0, 0, ErrTxnIDMismatch
	}

	n = s.deleteRange(key, end)
	if n != 0 || s.currentRev.sub != 0 {
		rev = int64(s.currentRev.main + 1)
	} else {
		rev = int64(s.currentRev.main)
	}
	return n, rev, nil
}

func (s *store) compactBarrier(ctx context.Context, ch chan struct{}) {
	if ctx == nil || ctx.Err() != nil {
		s.mu.Lock()
		select {
		case <-s.stopc:
		default:
			f := func(ctx context.Context) { s.compactBarrier(ctx, ch) }
			s.fifoSched.Schedule(f)
		}
		s.mu.Unlock()
		return
	}
	close(ch)
}

func (s *store) Compact(rev int64) (<-chan struct{}, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if rev <= s.compactMainRev {
		ch := make(chan struct{})
		f := func(ctx context.Context) { s.compactBarrier(ctx, ch) }
		s.fifoSched.Schedule(f)
		return ch, ErrCompacted
	}
	if rev > s.currentRev.main {
		return nil, ErrFutureRev
	}

	start := time.Now()

	s.compactMainRev = rev

	rbytes := newRevBytes()
	revToBytes(revision{main: rev}, rbytes)

	tx := s.b.BatchTx()
	tx.Lock()
	tx.UnsafePut(metaBucketName, scheduledCompactKeyName, rbytes)
	tx.Unlock()
	// ensure that desired compaction is persisted
	s.b.ForceCommit()

	keep := s.kvindex.Compact(rev)
	ch := make(chan struct{})
	var j = func(ctx context.Context) {
		if ctx.Err() != nil {
			s.compactBarrier(ctx, ch)
			return
		}
		if !s.scheduleCompaction(rev, keep) {
			s.compactBarrier(nil, ch)
			return
		}
		close(ch)
	}

	s.fifoSched.Schedule(j)

	indexCompactionPauseDurations.Observe(float64(time.Since(start) / time.Millisecond))
	return ch, nil
}

func (s *store) Hash() (uint32, int64, error) {
	s.b.ForceCommit()

	s.mu.Lock()
	defer s.mu.Unlock()

	// ignore hash consistent index field for now.
	// consistent index might be changed due to v2 internal sync, which
	// is not controllable by the user.
	ignores := make(map[backend.IgnoreKey]struct{})
	bk := backend.IgnoreKey{Bucket: string(metaBucketName), Key: string(consistentIndexKeyName)}
	ignores[bk] = struct{}{}

	h, err := s.b.Hash(ignores)
	rev := s.currentRev.main
	return h, rev, err
}

func (s *store) Commit() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.tx = s.b.BatchTx()
	s.tx.Lock()
	s.saveIndex()
	s.tx.Unlock()
	s.b.ForceCommit()
}

func (s *store) Restore(b backend.Backend) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	close(s.stopc)
	s.fifoSched.Stop()

	s.b = b
	s.kvindex = newTreeIndex()
	s.currentRev = revision{main: 1}
	s.compactMainRev = -1
	s.tx = b.BatchTx()
	s.txnID = -1
	s.fifoSched = schedule.NewFIFOScheduler()
	s.stopc = make(chan struct{})

	return s.restore()
}

func (s *store) restore() error {
	min, max := newRevBytes(), newRevBytes()
	revToBytes(revision{main: 1}, min)
	revToBytes(revision{main: math.MaxInt64, sub: math.MaxInt64}, max)

	keyToLease := make(map[string]lease.LeaseID)

	// restore index
	tx := s.b.BatchTx()
	tx.Lock()
	_, finishedCompactBytes := tx.UnsafeRange(metaBucketName, finishedCompactKeyName, nil, 0)
	if len(finishedCompactBytes) != 0 {
		s.compactMainRev = bytesToRev(finishedCompactBytes[0]).main
		plog.Printf("restore compact to %d", s.compactMainRev)
	}

	// TODO: limit N to reduce max memory usage
	keys, vals := tx.UnsafeRange(keyBucketName, min, max, 0)
	for i, key := range keys {
		var kv mvccpb.KeyValue
		if err := kv.Unmarshal(vals[i]); err != nil {
			plog.Fatalf("cannot unmarshal event: %v", err)
		}

		rev := bytesToRev(key[:revBytesLen])

		// restore index
		switch {
		case isTombstone(key):
			s.kvindex.Tombstone(kv.Key, rev)
			delete(keyToLease, string(kv.Key))

		default:
			s.kvindex.Restore(kv.Key, revision{kv.CreateRevision, 0}, rev, kv.Version)

			if lid := lease.LeaseID(kv.Lease); lid != lease.NoLease {
				keyToLease[string(kv.Key)] = lid
			} else {
				delete(keyToLease, string(kv.Key))
			}
		}

		// update revision
		s.currentRev = rev
	}

	// keys in the range [compacted revision -N, compaction] might all be deleted due to compaction.
	// the correct revision should be set to compaction revision in the case, not the largest revision
	// we have seen.
	if s.currentRev.main < s.compactMainRev {
		s.currentRev.main = s.compactMainRev
	}

	for key, lid := range keyToLease {
		if s.le == nil {
			panic("no lessor to attach lease")
		}
		err := s.le.Attach(lid, []lease.LeaseItem{{Key: key}})
		if err != nil {
			plog.Errorf("unexpected Attach error: %v", err)
		}
	}

	_, scheduledCompactBytes := tx.UnsafeRange(metaBucketName, scheduledCompactKeyName, nil, 0)
	scheduledCompact := int64(0)
	if len(scheduledCompactBytes) != 0 {
		scheduledCompact = bytesToRev(scheduledCompactBytes[0]).main
		if scheduledCompact <= s.compactMainRev {
			scheduledCompact = 0
		}
	}

	tx.Unlock()

	if scheduledCompact != 0 {
		s.Compact(scheduledCompact)
		plog.Printf("resume scheduled compaction at %d", scheduledCompact)
	}

	return nil
}

func (s *store) Close() error {
	close(s.stopc)
	s.fifoSched.Stop()
	return nil
}

func (a *store) Equal(b *store) bool {
	if a.currentRev != b.currentRev {
		return false
	}
	if a.compactMainRev != b.compactMainRev {
		return false
	}
	return a.kvindex.Equal(b.kvindex)
}

// range is a keyword in Go, add Keys suffix.
func (s *store) rangeKeys(key, end []byte, limit, rangeRev int64, countOnly bool) (kvs []mvccpb.KeyValue, count int, curRev int64, err error) {
	curRev = int64(s.currentRev.main)
	if s.currentRev.sub > 0 {
		curRev += 1
	}

	if rangeRev > curRev {
		return nil, -1, s.currentRev.main, ErrFutureRev
	}
	var rev int64
	if rangeRev <= 0 {
		rev = curRev
	} else {
		rev = rangeRev
	}
	if rev < s.compactMainRev {
		return nil, -1, 0, ErrCompacted
	}

	_, revpairs := s.kvindex.Range(key, end, int64(rev))
	if len(revpairs) == 0 {
		return nil, 0, curRev, nil
	}
	if countOnly {
		return nil, len(revpairs), curRev, nil
	}

	for _, revpair := range revpairs {
		start, end := revBytesRange(revpair)

		_, vs := s.tx.UnsafeRange(keyBucketName, start, end, 0)
		if len(vs) != 1 {
			plog.Fatalf("range cannot find rev (%d,%d)", revpair.main, revpair.sub)
		}

		var kv mvccpb.KeyValue
		if err := kv.Unmarshal(vs[0]); err != nil {
			plog.Fatalf("cannot unmarshal event: %v", err)
		}
		kvs = append(kvs, kv)
		if limit > 0 && len(kvs) >= int(limit) {
			break
		}
	}
	return kvs, len(revpairs), curRev, nil
}

func (s *store) put(key, value []byte, leaseID lease.LeaseID) {
	rev := s.currentRev.main + 1
	c := rev
	oldLease := lease.NoLease

	// if the key exists before, use its previous created and
	// get its previous leaseID
	grev, created, ver, err := s.kvindex.Get(key, rev)
	if err == nil {
		c = created.main
		ibytes := newRevBytes()
		revToBytes(grev, ibytes)
		_, vs := s.tx.UnsafeRange(keyBucketName, ibytes, nil, 0)
		var kv mvccpb.KeyValue
		if err = kv.Unmarshal(vs[0]); err != nil {
			plog.Fatalf("cannot unmarshal value: %v", err)
		}
		oldLease = lease.LeaseID(kv.Lease)
	}

	ibytes := newRevBytes()
	revToBytes(revision{main: rev, sub: s.currentRev.sub}, ibytes)

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

	s.tx.UnsafeSeqPut(keyBucketName, ibytes, d)
	s.kvindex.Put(key, revision{main: rev, sub: s.currentRev.sub})
	s.changes = append(s.changes, kv)
	s.currentRev.sub += 1

	if oldLease != lease.NoLease {
		if s.le == nil {
			panic("no lessor to detach lease")
		}

		err = s.le.Detach(oldLease, []lease.LeaseItem{{Key: string(key)}})
		if err != nil {
			plog.Errorf("unexpected error from lease detach: %v", err)
		}
	}

	if leaseID != lease.NoLease {
		if s.le == nil {
			panic("no lessor to attach lease")
		}

		err = s.le.Attach(leaseID, []lease.LeaseItem{{Key: string(key)}})
		if err != nil {
			panic("unexpected error from lease Attach")
		}
	}
}

func (s *store) deleteRange(key, end []byte) int64 {
	rrev := s.currentRev.main
	if s.currentRev.sub > 0 {
		rrev += 1
	}
	keys, revs := s.kvindex.Range(key, end, rrev)

	if len(keys) == 0 {
		return 0
	}

	for i, key := range keys {
		s.delete(key, revs[i])
	}
	return int64(len(keys))
}

func (s *store) delete(key []byte, rev revision) {
	mainrev := s.currentRev.main + 1

	ibytes := newRevBytes()
	revToBytes(revision{main: mainrev, sub: s.currentRev.sub}, ibytes)
	ibytes = appendMarkTombstone(ibytes)

	kv := mvccpb.KeyValue{
		Key: key,
	}

	d, err := kv.Marshal()
	if err != nil {
		plog.Fatalf("cannot marshal event: %v", err)
	}

	s.tx.UnsafeSeqPut(keyBucketName, ibytes, d)
	err = s.kvindex.Tombstone(key, revision{main: mainrev, sub: s.currentRev.sub})
	if err != nil {
		plog.Fatalf("cannot tombstone an existing key (%s): %v", string(key), err)
	}
	s.changes = append(s.changes, kv)
	s.currentRev.sub += 1

	ibytes = newRevBytes()
	revToBytes(rev, ibytes)
	_, vs := s.tx.UnsafeRange(keyBucketName, ibytes, nil, 0)

	kv.Reset()
	if err = kv.Unmarshal(vs[0]); err != nil {
		plog.Fatalf("cannot unmarshal value: %v", err)
	}

	if lease.LeaseID(kv.Lease) != lease.NoLease {
		err = s.le.Detach(lease.LeaseID(kv.Lease), []lease.LeaseItem{{Key: string(kv.Key)}})
		if err != nil {
			plog.Errorf("cannot detach %v", err)
		}
	}
}

func (s *store) getChanges() []mvccpb.KeyValue {
	changes := s.changes
	s.changes = make([]mvccpb.KeyValue, 0, 4)
	return changes
}

func (s *store) saveIndex() {
	if s.ig == nil {
		return
	}
	tx := s.tx
	bs := s.bytesBuf8
	binary.BigEndian.PutUint64(bs, s.ig.ConsistentIndex())
	// put the index into the underlying backend
	// tx has been locked in TxnBegin, so there is no need to lock it again
	tx.UnsafePut(metaBucketName, consistentIndexKeyName, bs)
}

func (s *store) ConsistentIndex() uint64 {
	// TODO: cache index in a uint64 field?
	tx := s.b.BatchTx()
	tx.Lock()
	defer tx.Unlock()
	_, vs := tx.UnsafeRange(metaBucketName, consistentIndexKeyName, nil, 0)
	if len(vs) == 0 {
		return 0
	}
	return binary.BigEndian.Uint64(vs[0])
}

// appendMarkTombstone appends tombstone mark to normal revision bytes.
func appendMarkTombstone(b []byte) []byte {
	if len(b) != revBytesLen {
		plog.Panicf("cannot append mark to non normal revision bytes")
	}
	return append(b, markTombstone)
}

// isTombstone checks whether the revision bytes is a tombstone.
func isTombstone(b []byte) bool {
	return len(b) == markedRevBytesLen && b[markBytePosition] == markTombstone
}

// revBytesRange returns the range of revision bytes at
// the given revision.
func revBytesRange(rev revision) (start, end []byte) {
	start = newRevBytes()
	revToBytes(rev, start)

	end = newRevBytes()
	endRev := revision{main: rev.main, sub: rev.sub + 1}
	revToBytes(endRev, end)

	return start, end
}
