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
	"crypto/rand"
	"encoding/binary"
	"math"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/mvcc/backend"
	"github.com/coreos/etcd/mvcc/mvccpb"
	"github.com/coreos/etcd/pkg/schedule"
	"github.com/coreos/etcd/pkg/testutil"
)

func TestStoreRev(t *testing.T) {
	b, tmpPath := backend.NewDefaultTmpBackend()
	s := NewStore(b, &lease.FakeLessor{}, nil)
	defer s.Close()
	defer os.Remove(tmpPath)

	for i := 1; i <= 3; i++ {
		s.Put([]byte("foo"), []byte("bar"), lease.NoLease)
		if r := s.Rev(); r != int64(i+1) {
			t.Errorf("#%d: rev = %d, want %d", i, r, i+1)
		}
	}
}

func TestStorePut(t *testing.T) {
	kv := mvccpb.KeyValue{
		Key:            []byte("foo"),
		Value:          []byte("bar"),
		CreateRevision: 1,
		ModRevision:    2,
		Version:        1,
	}
	kvb, err := kv.Marshal()
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		rev revision
		r   indexGetResp
		rr  *rangeResp

		wrev    revision
		wkey    []byte
		wkv     mvccpb.KeyValue
		wputrev revision
	}{
		{
			revision{1, 0},
			indexGetResp{revision{}, revision{}, 0, ErrRevisionNotFound},
			nil,

			revision{1, 1},
			newTestKeyBytes(revision{2, 0}, false),
			mvccpb.KeyValue{
				Key:            []byte("foo"),
				Value:          []byte("bar"),
				CreateRevision: 2,
				ModRevision:    2,
				Version:        1,
				Lease:          1,
			},
			revision{2, 0},
		},
		{
			revision{1, 1},
			indexGetResp{revision{2, 0}, revision{2, 0}, 1, nil},
			&rangeResp{[][]byte{newTestKeyBytes(revision{2, 1}, false)}, [][]byte{kvb}},

			revision{1, 2},
			newTestKeyBytes(revision{2, 1}, false),
			mvccpb.KeyValue{
				Key:            []byte("foo"),
				Value:          []byte("bar"),
				CreateRevision: 2,
				ModRevision:    2,
				Version:        2,
				Lease:          2,
			},
			revision{2, 1},
		},
		{
			revision{2, 0},
			indexGetResp{revision{2, 1}, revision{2, 0}, 2, nil},
			&rangeResp{[][]byte{newTestKeyBytes(revision{2, 1}, false)}, [][]byte{kvb}},

			revision{2, 1},
			newTestKeyBytes(revision{3, 0}, false),
			mvccpb.KeyValue{
				Key:            []byte("foo"),
				Value:          []byte("bar"),
				CreateRevision: 2,
				ModRevision:    3,
				Version:        3,
				Lease:          3,
			},
			revision{3, 0},
		},
	}
	for i, tt := range tests {
		s := newFakeStore()
		b := s.b.(*fakeBackend)
		fi := s.kvindex.(*fakeIndex)

		s.currentRev = tt.rev
		s.tx = b.BatchTx()
		fi.indexGetRespc <- tt.r
		if tt.rr != nil {
			b.tx.rangeRespc <- *tt.rr
		}

		s.put([]byte("foo"), []byte("bar"), lease.LeaseID(i+1))

		data, err := tt.wkv.Marshal()
		if err != nil {
			t.Errorf("#%d: marshal err = %v, want nil", i, err)
		}

		wact := []testutil.Action{
			{"seqput", []interface{}{keyBucketName, tt.wkey, data}},
		}

		if tt.rr != nil {
			wact = []testutil.Action{
				{"seqput", []interface{}{keyBucketName, tt.wkey, data}},
			}
		}

		if g := b.tx.Action(); !reflect.DeepEqual(g, wact) {
			t.Errorf("#%d: tx action = %+v, want %+v", i, g, wact)
		}
		wact = []testutil.Action{
			{"get", []interface{}{[]byte("foo"), tt.wputrev.main}},
			{"put", []interface{}{[]byte("foo"), tt.wputrev}},
		}
		if g := fi.Action(); !reflect.DeepEqual(g, wact) {
			t.Errorf("#%d: index action = %+v, want %+v", i, g, wact)
		}
		if s.currentRev != tt.wrev {
			t.Errorf("#%d: rev = %+v, want %+v", i, s.currentRev, tt.wrev)
		}

		s.Close()
	}
}

func TestStoreRange(t *testing.T) {
	key := newTestKeyBytes(revision{2, 0}, false)
	kv := mvccpb.KeyValue{
		Key:            []byte("foo"),
		Value:          []byte("bar"),
		CreateRevision: 1,
		ModRevision:    2,
		Version:        1,
	}
	kvb, err := kv.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	currev := revision{1, 1}
	wrev := int64(2)

	tests := []struct {
		idxr indexRangeResp
		r    rangeResp
	}{
		{
			indexRangeResp{[][]byte{[]byte("foo")}, []revision{{2, 0}}},
			rangeResp{[][]byte{key}, [][]byte{kvb}},
		},
		{
			indexRangeResp{[][]byte{[]byte("foo"), []byte("foo1")}, []revision{{2, 0}, {3, 0}}},
			rangeResp{[][]byte{key}, [][]byte{kvb}},
		},
	}
	for i, tt := range tests {
		s := newFakeStore()
		b := s.b.(*fakeBackend)
		fi := s.kvindex.(*fakeIndex)

		s.currentRev = currev
		s.tx = b.BatchTx()
		b.tx.rangeRespc <- tt.r
		fi.indexRangeRespc <- tt.idxr

		kvs, _, rev, err := s.rangeKeys([]byte("foo"), []byte("goo"), 1, 0, false)
		if err != nil {
			t.Errorf("#%d: err = %v, want nil", i, err)
		}
		if w := []mvccpb.KeyValue{kv}; !reflect.DeepEqual(kvs, w) {
			t.Errorf("#%d: kvs = %+v, want %+v", i, kvs, w)
		}
		if rev != wrev {
			t.Errorf("#%d: rev = %d, want %d", i, rev, wrev)
		}

		wstart, wend := revBytesRange(tt.idxr.revs[0])
		wact := []testutil.Action{
			{"range", []interface{}{keyBucketName, wstart, wend, int64(0)}},
		}
		if g := b.tx.Action(); !reflect.DeepEqual(g, wact) {
			t.Errorf("#%d: tx action = %+v, want %+v", i, g, wact)
		}
		wact = []testutil.Action{
			{"range", []interface{}{[]byte("foo"), []byte("goo"), wrev}},
		}
		if g := fi.Action(); !reflect.DeepEqual(g, wact) {
			t.Errorf("#%d: index action = %+v, want %+v", i, g, wact)
		}
		if s.currentRev != currev {
			t.Errorf("#%d: current rev = %+v, want %+v", i, s.currentRev, currev)
		}

		s.Close()
	}
}

func TestStoreDeleteRange(t *testing.T) {
	key := newTestKeyBytes(revision{2, 0}, false)
	kv := mvccpb.KeyValue{
		Key:            []byte("foo"),
		Value:          []byte("bar"),
		CreateRevision: 1,
		ModRevision:    2,
		Version:        1,
	}
	kvb, err := kv.Marshal()
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		rev revision
		r   indexRangeResp
		rr  rangeResp

		wkey    []byte
		wrev    revision
		wrrev   int64
		wdelrev revision
	}{
		{
			revision{2, 0},
			indexRangeResp{[][]byte{[]byte("foo")}, []revision{{2, 0}}},
			rangeResp{[][]byte{key}, [][]byte{kvb}},

			newTestKeyBytes(revision{3, 0}, true),
			revision{2, 1},
			2,
			revision{3, 0},
		},
		{
			revision{2, 1},
			indexRangeResp{[][]byte{[]byte("foo")}, []revision{{2, 0}}},
			rangeResp{[][]byte{key}, [][]byte{kvb}},

			newTestKeyBytes(revision{3, 1}, true),
			revision{2, 2},
			3,
			revision{3, 1},
		},
	}
	for i, tt := range tests {
		s := newFakeStore()
		b := s.b.(*fakeBackend)
		fi := s.kvindex.(*fakeIndex)

		s.currentRev = tt.rev
		s.tx = b.BatchTx()
		fi.indexRangeRespc <- tt.r
		b.tx.rangeRespc <- tt.rr

		n := s.deleteRange([]byte("foo"), []byte("goo"))
		if n != 1 {
			t.Errorf("#%d: n = %d, want 1", i, n)
		}

		data, err := (&mvccpb.KeyValue{
			Key: []byte("foo"),
		}).Marshal()
		if err != nil {
			t.Errorf("#%d: marshal err = %v, want nil", i, err)
		}
		wact := []testutil.Action{
			{"seqput", []interface{}{keyBucketName, tt.wkey, data}},
		}
		if g := b.tx.Action(); !reflect.DeepEqual(g, wact) {
			t.Errorf("#%d: tx action = %+v, want %+v", i, g, wact)
		}
		wact = []testutil.Action{
			{"range", []interface{}{[]byte("foo"), []byte("goo"), tt.wrrev}},
			{"tombstone", []interface{}{[]byte("foo"), tt.wdelrev}},
		}
		if g := fi.Action(); !reflect.DeepEqual(g, wact) {
			t.Errorf("#%d: index action = %+v, want %+v", i, g, wact)
		}
		if s.currentRev != tt.wrev {
			t.Errorf("#%d: rev = %+v, want %+v", i, s.currentRev, tt.wrev)
		}
	}
}

func TestStoreCompact(t *testing.T) {
	s := newFakeStore()
	defer s.Close()
	b := s.b.(*fakeBackend)
	fi := s.kvindex.(*fakeIndex)

	s.currentRev = revision{3, 0}
	fi.indexCompactRespc <- map[revision]struct{}{{1, 0}: {}}
	key1 := newTestKeyBytes(revision{1, 0}, false)
	key2 := newTestKeyBytes(revision{2, 0}, false)
	b.tx.rangeRespc <- rangeResp{[][]byte{key1, key2}, nil}

	s.Compact(3)
	s.fifoSched.WaitFinish(1)

	if s.compactMainRev != 3 {
		t.Errorf("compact main rev = %d, want 3", s.compactMainRev)
	}
	end := make([]byte, 8)
	binary.BigEndian.PutUint64(end, uint64(4))
	wact := []testutil.Action{
		{"put", []interface{}{metaBucketName, scheduledCompactKeyName, newTestRevBytes(revision{3, 0})}},
		{"range", []interface{}{keyBucketName, make([]byte, 17), end, int64(10000)}},
		{"delete", []interface{}{keyBucketName, key2}},
		{"put", []interface{}{metaBucketName, finishedCompactKeyName, newTestRevBytes(revision{3, 0})}},
	}
	if g := b.tx.Action(); !reflect.DeepEqual(g, wact) {
		t.Errorf("tx actions = %+v, want %+v", g, wact)
	}
	wact = []testutil.Action{
		{"compact", []interface{}{int64(3)}},
	}
	if g := fi.Action(); !reflect.DeepEqual(g, wact) {
		t.Errorf("index action = %+v, want %+v", g, wact)
	}
}

func TestStoreRestore(t *testing.T) {
	s := newFakeStore()
	b := s.b.(*fakeBackend)
	fi := s.kvindex.(*fakeIndex)

	putkey := newTestKeyBytes(revision{3, 0}, false)
	putkv := mvccpb.KeyValue{
		Key:            []byte("foo"),
		Value:          []byte("bar"),
		CreateRevision: 4,
		ModRevision:    4,
		Version:        1,
	}
	putkvb, err := putkv.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	delkey := newTestKeyBytes(revision{5, 0}, true)
	delkv := mvccpb.KeyValue{
		Key: []byte("foo"),
	}
	delkvb, err := delkv.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	b.tx.rangeRespc <- rangeResp{[][]byte{finishedCompactKeyName}, [][]byte{newTestRevBytes(revision{3, 0})}}
	b.tx.rangeRespc <- rangeResp{[][]byte{putkey, delkey}, [][]byte{putkvb, delkvb}}
	b.tx.rangeRespc <- rangeResp{[][]byte{scheduledCompactKeyName}, [][]byte{newTestRevBytes(revision{3, 0})}}

	s.restore()

	if s.compactMainRev != 3 {
		t.Errorf("compact rev = %d, want 5", s.compactMainRev)
	}
	wrev := revision{5, 0}
	if !reflect.DeepEqual(s.currentRev, wrev) {
		t.Errorf("current rev = %v, want %v", s.currentRev, wrev)
	}
	wact := []testutil.Action{
		{"range", []interface{}{metaBucketName, finishedCompactKeyName, []byte(nil), int64(0)}},
		{"range", []interface{}{keyBucketName, newTestRevBytes(revision{1, 0}), newTestRevBytes(revision{math.MaxInt64, math.MaxInt64}), int64(0)}},
		{"range", []interface{}{metaBucketName, scheduledCompactKeyName, []byte(nil), int64(0)}},
	}
	if g := b.tx.Action(); !reflect.DeepEqual(g, wact) {
		t.Errorf("tx actions = %+v, want %+v", g, wact)
	}

	gens := []generation{
		{created: revision{4, 0}, ver: 2, revs: []revision{{3, 0}, {5, 0}}},
		{created: revision{0, 0}, ver: 0, revs: nil},
	}
	ki := &keyIndex{key: []byte("foo"), modified: revision{5, 0}, generations: gens}
	wact = []testutil.Action{
		{"insert", []interface{}{ki}},
	}
	if g := fi.Action(); !reflect.DeepEqual(g, wact) {
		t.Errorf("index action = %+v, want %+v", g, wact)
	}
}

func TestRestoreContinueUnfinishedCompaction(t *testing.T) {
	b, tmpPath := backend.NewDefaultTmpBackend()
	s0 := NewStore(b, &lease.FakeLessor{}, nil)
	defer os.Remove(tmpPath)

	s0.Put([]byte("foo"), []byte("bar"), lease.NoLease)
	s0.Put([]byte("foo"), []byte("bar1"), lease.NoLease)
	s0.Put([]byte("foo"), []byte("bar2"), lease.NoLease)

	// write scheduled compaction, but not do compaction
	rbytes := newRevBytes()
	revToBytes(revision{main: 2}, rbytes)
	tx := s0.b.BatchTx()
	tx.Lock()
	tx.UnsafePut(metaBucketName, scheduledCompactKeyName, rbytes)
	tx.Unlock()

	s0.Close()

	s1 := NewStore(b, &lease.FakeLessor{}, nil)

	// wait for scheduled compaction to be finished
	time.Sleep(100 * time.Millisecond)

	if _, err := s1.Range([]byte("foo"), nil, RangeOptions{Rev: 1}); err != ErrCompacted {
		t.Errorf("range on compacted rev error = %v, want %v", err, ErrCompacted)
	}
	// check the key in backend is deleted
	revbytes := newRevBytes()
	revToBytes(revision{main: 1}, revbytes)

	// The disk compaction is done asynchronously and requires more time on slow disk.
	// try 5 times for CI with slow IO.
	for i := 0; i < 5; i++ {
		tx = s1.b.BatchTx()
		tx.Lock()
		ks, _ := tx.UnsafeRange(keyBucketName, revbytes, nil, 0)
		tx.Unlock()
		if len(ks) != 0 {
			time.Sleep(100 * time.Millisecond)
			continue
		}
		return
	}

	t.Errorf("key for rev %+v still exists, want deleted", bytesToRev(revbytes))
}

func TestTxnPut(t *testing.T) {
	// assign arbitrary size
	bytesN := 30
	sliceN := 100
	keys := createBytesSlice(bytesN, sliceN)
	vals := createBytesSlice(bytesN, sliceN)

	b, tmpPath := backend.NewDefaultTmpBackend()
	s := NewStore(b, &lease.FakeLessor{}, nil)
	defer cleanup(s, b, tmpPath)

	for i := 0; i < sliceN; i++ {
		id := s.TxnBegin()
		base := int64(i + 2)

		rev, err := s.TxnPut(id, keys[i], vals[i], lease.NoLease)
		if err != nil {
			t.Error("txn put error")
		}
		if rev != base {
			t.Errorf("#%d: rev = %d, want %d", i, rev, base)
		}

		s.TxnEnd(id)
	}
}

func TestTxnBlockBackendForceCommit(t *testing.T) {
	b, tmpPath := backend.NewDefaultTmpBackend()
	s := NewStore(b, &lease.FakeLessor{}, nil)
	defer os.Remove(tmpPath)

	id := s.TxnBegin()

	done := make(chan struct{})
	go func() {
		s.b.ForceCommit()
		done <- struct{}{}
	}()
	select {
	case <-done:
		t.Fatalf("failed to block ForceCommit")
	case <-time.After(100 * time.Millisecond):
	}

	s.TxnEnd(id)
	select {
	case <-done:
	case <-time.After(5 * time.Second): // wait 5 seconds for CI with slow IO
		testutil.FatalStack(t, "failed to execute ForceCommit")
	}
}

// TODO: test attach key to lessor

func newTestRevBytes(rev revision) []byte {
	bytes := newRevBytes()
	revToBytes(rev, bytes)
	return bytes
}

func newTestKeyBytes(rev revision, tombstone bool) []byte {
	bytes := newRevBytes()
	revToBytes(rev, bytes)
	if tombstone {
		bytes = appendMarkTombstone(bytes)
	}
	return bytes
}

// TestStoreHashAfterForceCommit ensures that later Hash call to
// closed backend with ForceCommit does not panic.
func TestStoreHashAfterForceCommit(t *testing.T) {
	be, tmpPath := backend.NewDefaultTmpBackend()
	kv := NewStore(be, &lease.FakeLessor{}, nil)
	defer os.Remove(tmpPath)

	// as in EtcdServer.HardStop
	kv.Close()
	be.Close()

	kv.Hash()
}

func newFakeStore() *store {
	b := &fakeBackend{&fakeBatchTx{
		Recorder:   &testutil.RecorderBuffered{},
		rangeRespc: make(chan rangeResp, 5)}}
	fi := &fakeIndex{
		Recorder:              &testutil.RecorderBuffered{},
		indexGetRespc:         make(chan indexGetResp, 1),
		indexRangeRespc:       make(chan indexRangeResp, 1),
		indexRangeEventsRespc: make(chan indexRangeEventsResp, 1),
		indexCompactRespc:     make(chan map[revision]struct{}, 1),
	}
	return &store{
		b:              b,
		le:             &lease.FakeLessor{},
		kvindex:        fi,
		currentRev:     revision{},
		compactMainRev: -1,
		fifoSched:      schedule.NewFIFOScheduler(),
		stopc:          make(chan struct{}),
	}
}

type rangeResp struct {
	keys [][]byte
	vals [][]byte
}

type fakeBatchTx struct {
	testutil.Recorder
	rangeRespc chan rangeResp
}

func (b *fakeBatchTx) Lock()                          {}
func (b *fakeBatchTx) Unlock()                        {}
func (b *fakeBatchTx) UnsafeCreateBucket(name []byte) {}
func (b *fakeBatchTx) UnsafePut(bucketName []byte, key []byte, value []byte) {
	b.Recorder.Record(testutil.Action{Name: "put", Params: []interface{}{bucketName, key, value}})
}
func (b *fakeBatchTx) UnsafeSeqPut(bucketName []byte, key []byte, value []byte) {
	b.Recorder.Record(testutil.Action{Name: "seqput", Params: []interface{}{bucketName, key, value}})
}
func (b *fakeBatchTx) UnsafeRange(bucketName []byte, key, endKey []byte, limit int64) (keys [][]byte, vals [][]byte) {
	b.Recorder.Record(testutil.Action{Name: "range", Params: []interface{}{bucketName, key, endKey, limit}})
	r := <-b.rangeRespc
	return r.keys, r.vals
}
func (b *fakeBatchTx) UnsafeDelete(bucketName []byte, key []byte) {
	b.Recorder.Record(testutil.Action{Name: "delete", Params: []interface{}{bucketName, key}})
}
func (b *fakeBatchTx) UnsafeForEach(bucketName []byte, visitor func(k, v []byte) error) error {
	return nil
}
func (b *fakeBatchTx) Commit()        {}
func (b *fakeBatchTx) CommitAndStop() {}

type fakeBackend struct {
	tx *fakeBatchTx
}

func (b *fakeBackend) BatchTx() backend.BatchTx                                    { return b.tx }
func (b *fakeBackend) Hash(ignores map[backend.IgnoreKey]struct{}) (uint32, error) { return 0, nil }
func (b *fakeBackend) Size() int64                                                 { return 0 }
func (b *fakeBackend) Snapshot() backend.Snapshot                                  { return nil }
func (b *fakeBackend) ForceCommit()                                                {}
func (b *fakeBackend) Defrag() error                                               { return nil }
func (b *fakeBackend) Close() error                                                { return nil }

type indexGetResp struct {
	rev     revision
	created revision
	ver     int64
	err     error
}

type indexRangeResp struct {
	keys [][]byte
	revs []revision
}

type indexRangeEventsResp struct {
	revs []revision
}

type fakeIndex struct {
	testutil.Recorder
	indexGetRespc         chan indexGetResp
	indexRangeRespc       chan indexRangeResp
	indexRangeEventsRespc chan indexRangeEventsResp
	indexCompactRespc     chan map[revision]struct{}
}

func (i *fakeIndex) Get(key []byte, atRev int64) (rev, created revision, ver int64, err error) {
	i.Recorder.Record(testutil.Action{Name: "get", Params: []interface{}{key, atRev}})
	r := <-i.indexGetRespc
	return r.rev, r.created, r.ver, r.err
}
func (i *fakeIndex) Range(key, end []byte, atRev int64) ([][]byte, []revision) {
	i.Recorder.Record(testutil.Action{Name: "range", Params: []interface{}{key, end, atRev}})
	r := <-i.indexRangeRespc
	return r.keys, r.revs
}
func (i *fakeIndex) Put(key []byte, rev revision) {
	i.Recorder.Record(testutil.Action{Name: "put", Params: []interface{}{key, rev}})
}
func (i *fakeIndex) Tombstone(key []byte, rev revision) error {
	i.Recorder.Record(testutil.Action{Name: "tombstone", Params: []interface{}{key, rev}})
	return nil
}
func (i *fakeIndex) RangeSince(key, end []byte, rev int64) []revision {
	i.Recorder.Record(testutil.Action{Name: "rangeEvents", Params: []interface{}{key, end, rev}})
	r := <-i.indexRangeEventsRespc
	return r.revs
}
func (i *fakeIndex) Compact(rev int64) map[revision]struct{} {
	i.Recorder.Record(testutil.Action{Name: "compact", Params: []interface{}{rev}})
	return <-i.indexCompactRespc
}
func (i *fakeIndex) Equal(b index) bool { return false }

func (i *fakeIndex) Insert(ki *keyIndex) {
	i.Recorder.Record(testutil.Action{Name: "insert", Params: []interface{}{ki}})
}

func createBytesSlice(bytesN, sliceN int) [][]byte {
	rs := [][]byte{}
	for len(rs) != sliceN {
		v := make([]byte, bytesN)
		if _, err := rand.Read(v); err != nil {
			panic(err)
		}
		rs = append(rs, v)
	}
	return rs
}
