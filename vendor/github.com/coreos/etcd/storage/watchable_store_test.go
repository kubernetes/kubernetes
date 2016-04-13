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
	"bytes"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/storage/backend"
	"github.com/coreos/etcd/storage/storagepb"
)

func TestWatch(t *testing.T) {
	b, tmpPath := backend.NewDefaultTmpBackend()
	s := newWatchableStore(b, &lease.FakeLessor{})

	defer func() {
		s.store.Close()
		os.Remove(tmpPath)
	}()

	testKey := []byte("foo")
	testValue := []byte("bar")
	s.Put(testKey, testValue, lease.NoLease)

	w := s.NewWatchStream()
	w.Watch(testKey, nil, 0)

	if !s.synced.contains(string(testKey)) {
		// the key must have had an entry in synced
		t.Errorf("existence = false, want true")
	}
}

func TestNewWatcherCancel(t *testing.T) {
	b, tmpPath := backend.NewDefaultTmpBackend()
	s := newWatchableStore(b, &lease.FakeLessor{})

	defer func() {
		s.store.Close()
		os.Remove(tmpPath)
	}()
	testKey := []byte("foo")
	testValue := []byte("bar")
	s.Put(testKey, testValue, lease.NoLease)

	w := s.NewWatchStream()
	wt := w.Watch(testKey, nil, 0)

	if err := w.Cancel(wt); err != nil {
		t.Error(err)
	}

	if s.synced.contains(string(testKey)) {
		// the key shoud have been deleted
		t.Errorf("existence = true, want false")
	}
}

// TestCancelUnsynced tests if running CancelFunc removes watchers from unsynced.
func TestCancelUnsynced(t *testing.T) {
	b, tmpPath := backend.NewDefaultTmpBackend()

	// manually create watchableStore instead of newWatchableStore
	// because newWatchableStore automatically calls syncWatchers
	// method to sync watchers in unsynced map. We want to keep watchers
	// in unsynced to test if syncWatchers works as expected.
	s := &watchableStore{
		store:    NewStore(b, &lease.FakeLessor{}),
		unsynced: newWatcherGroup(),

		// to make the test not crash from assigning to nil map.
		// 'synced' doesn't get populated in this test.
		synced: newWatcherGroup(),
	}

	defer func() {
		s.store.Close()
		os.Remove(tmpPath)
	}()

	// Put a key so that we can spawn watchers on that key.
	// (testKey in this test). This increases the rev to 1,
	// and later we can we set the watcher's startRev to 1,
	// and force watchers to be in unsynced.
	testKey := []byte("foo")
	testValue := []byte("bar")
	s.Put(testKey, testValue, lease.NoLease)

	w := s.NewWatchStream()

	// arbitrary number for watchers
	watcherN := 100

	// create watcherN of watch ids to cancel
	watchIDs := make([]WatchID, watcherN)
	for i := 0; i < watcherN; i++ {
		// use 1 to keep watchers in unsynced
		watchIDs[i] = w.Watch(testKey, nil, 1)
	}

	for _, idx := range watchIDs {
		if err := w.Cancel(idx); err != nil {
			t.Error(err)
		}
	}

	// After running CancelFunc
	//
	// unsynced should be empty
	// because cancel removes watcher from unsynced
	if size := s.unsynced.size(); size != 0 {
		t.Errorf("unsynced size = %d, want 0", size)
	}
}

// TestSyncWatchers populates unsynced watcher map and tests syncWatchers
// method to see if it correctly sends events to channel of unsynced watchers
// and moves these watchers to synced.
func TestSyncWatchers(t *testing.T) {
	b, tmpPath := backend.NewDefaultTmpBackend()

	s := &watchableStore{
		store:    NewStore(b, &lease.FakeLessor{}),
		unsynced: newWatcherGroup(),
		synced:   newWatcherGroup(),
	}

	defer func() {
		s.store.Close()
		os.Remove(tmpPath)
	}()

	testKey := []byte("foo")
	testValue := []byte("bar")
	s.Put(testKey, testValue, lease.NoLease)

	w := s.NewWatchStream()

	// arbitrary number for watchers
	watcherN := 100

	for i := 0; i < watcherN; i++ {
		// specify rev as 1 to keep watchers in unsynced
		w.Watch(testKey, nil, 1)
	}

	// Before running s.syncWatchers() synced should be empty because we manually
	// populate unsynced only
	sws := s.synced.watcherSetByKey(string(testKey))
	uws := s.unsynced.watcherSetByKey(string(testKey))

	if len(sws) != 0 {
		t.Fatalf("synced[string(testKey)] size = %d, want 0", len(sws))
	}
	// unsynced should not be empty because we manually populated unsynced only
	if len(uws) != watcherN {
		t.Errorf("unsynced size = %d, want %d", len(uws), watcherN)
	}

	// this should move all unsynced watchers to synced ones
	s.syncWatchers()

	sws = s.synced.watcherSetByKey(string(testKey))
	uws = s.unsynced.watcherSetByKey(string(testKey))

	// After running s.syncWatchers(), synced should not be empty because syncwatchers
	// populates synced in this test case
	if len(sws) != watcherN {
		t.Errorf("synced[string(testKey)] size = %d, want %d", len(sws), watcherN)
	}

	// unsynced should be empty because syncwatchers is expected to move all watchers
	// from unsynced to synced in this test case
	if len(uws) != 0 {
		t.Errorf("unsynced size = %d, want 0", len(uws))
	}

	for w := range sws {
		if w.cur != s.Rev() {
			t.Errorf("w.cur = %d, want %d", w.cur, s.Rev())
		}
	}

	if len(w.(*watchStream).ch) != watcherN {
		t.Errorf("watched event size = %d, want %d", len(w.(*watchStream).ch), watcherN)
	}

	evs := (<-w.(*watchStream).ch).Events
	if len(evs) != 1 {
		t.Errorf("len(evs) got = %d, want = 1", len(evs))
	}
	if evs[0].Type != storagepb.PUT {
		t.Errorf("got = %v, want = %v", evs[0].Type, storagepb.PUT)
	}
	if !bytes.Equal(evs[0].Kv.Key, testKey) {
		t.Errorf("got = %s, want = %s", evs[0].Kv.Key, testKey)
	}
	if !bytes.Equal(evs[0].Kv.Value, testValue) {
		t.Errorf("got = %s, want = %s", evs[0].Kv.Value, testValue)
	}
}

// TestWatchCompacted tests a watcher that watches on a compacted revision.
func TestWatchCompacted(t *testing.T) {
	b, tmpPath := backend.NewDefaultTmpBackend()
	s := newWatchableStore(b, &lease.FakeLessor{})

	defer func() {
		s.store.Close()
		os.Remove(tmpPath)
	}()
	testKey := []byte("foo")
	testValue := []byte("bar")

	maxRev := 10
	compactRev := int64(5)
	for i := 0; i < maxRev; i++ {
		s.Put(testKey, testValue, lease.NoLease)
	}
	err := s.Compact(compactRev)
	if err != nil {
		t.Fatalf("failed to compact kv (%v)", err)
	}

	w := s.NewWatchStream()
	wt := w.Watch(testKey, nil, compactRev-1)

	select {
	case resp := <-w.Chan():
		if resp.WatchID != wt {
			t.Errorf("resp.WatchID = %x, want %x", resp.WatchID, wt)
		}
		if resp.CompactRevision == 0 {
			t.Errorf("resp.Compacted = %v, want %v", resp.CompactRevision, compactRev)
		}
	case <-time.After(1 * time.Second):
		t.Fatalf("failed to receive response (timeout)")
	}
}

func TestWatchFutureRev(t *testing.T) {
	b, tmpPath := backend.NewDefaultTmpBackend()
	s := newWatchableStore(b, &lease.FakeLessor{})

	defer func() {
		s.store.Close()
		os.Remove(tmpPath)
	}()

	testKey := []byte("foo")
	testValue := []byte("bar")

	w := s.NewWatchStream()
	wrev := int64(10)
	w.Watch(testKey, nil, wrev)

	for i := 0; i < 10; i++ {
		rev := s.Put(testKey, testValue, lease.NoLease)
		if rev >= wrev {
			break
		}
	}

	select {
	case resp := <-w.Chan():
		if resp.Revision != wrev {
			t.Fatalf("rev = %d, want %d", resp.Revision, wrev)
		}
		if len(resp.Events) != 1 {
			t.Fatalf("failed to get events from the response")
		}
		if resp.Events[0].Kv.ModRevision != wrev {
			t.Fatalf("kv.rev = %d, want %d", resp.Events[0].Kv.ModRevision, wrev)
		}
	case <-time.After(time.Second):
		t.Fatal("failed to receive event in 1 second.")
	}
}

// TestWatchBatchUnsynced tests batching on unsynced watchers
func TestWatchBatchUnsynced(t *testing.T) {
	b, tmpPath := backend.NewDefaultTmpBackend()
	s := newWatchableStore(b, &lease.FakeLessor{})

	oldMaxRevs := watchBatchMaxRevs
	defer func() {
		watchBatchMaxRevs = oldMaxRevs
		s.store.Close()
		os.Remove(tmpPath)
	}()
	batches := 3
	watchBatchMaxRevs = 4

	v := []byte("foo")
	for i := 0; i < watchBatchMaxRevs*batches; i++ {
		s.Put(v, v, lease.NoLease)
	}

	w := s.NewWatchStream()
	w.Watch(v, nil, 1)
	for i := 0; i < batches; i++ {
		if resp := <-w.Chan(); len(resp.Events) != watchBatchMaxRevs {
			t.Fatalf("len(events) = %d, want %d", len(resp.Events), watchBatchMaxRevs)
		}
	}

	s.store.mu.Lock()
	defer s.store.mu.Unlock()
	if size := s.synced.size(); size != 1 {
		t.Errorf("synced size = %d, want 1", size)
	}
}

func TestNewMapwatcherToEventMap(t *testing.T) {
	k0, k1, k2 := []byte("foo0"), []byte("foo1"), []byte("foo2")
	v0, v1, v2 := []byte("bar0"), []byte("bar1"), []byte("bar2")

	ws := []*watcher{{key: k0}, {key: k1}, {key: k2}}

	evs := []storagepb.Event{
		{
			Type: storagepb.PUT,
			Kv:   &storagepb.KeyValue{Key: k0, Value: v0},
		},
		{
			Type: storagepb.PUT,
			Kv:   &storagepb.KeyValue{Key: k1, Value: v1},
		},
		{
			Type: storagepb.PUT,
			Kv:   &storagepb.KeyValue{Key: k2, Value: v2},
		},
	}

	tests := []struct {
		sync []*watcher
		evs  []storagepb.Event

		wwe map[*watcher][]storagepb.Event
	}{
		// no watcher in sync, some events should return empty wwe
		{
			nil,
			evs,
			map[*watcher][]storagepb.Event{},
		},

		// one watcher in sync, one event that does not match the key of that
		// watcher should return empty wwe
		{
			[]*watcher{ws[2]},
			evs[:1],
			map[*watcher][]storagepb.Event{},
		},

		// one watcher in sync, one event that matches the key of that
		// watcher should return wwe with that matching watcher
		{
			[]*watcher{ws[1]},
			evs[1:2],
			map[*watcher][]storagepb.Event{
				ws[1]: evs[1:2],
			},
		},

		// two watchers in sync that watches two different keys, one event
		// that matches the key of only one of the watcher should return wwe
		// with the matching watcher
		{
			[]*watcher{ws[0], ws[2]},
			evs[2:],
			map[*watcher][]storagepb.Event{
				ws[2]: evs[2:],
			},
		},

		// two watchers in sync that watches the same key, two events that
		// match the keys should return wwe with those two watchers
		{
			[]*watcher{ws[0], ws[1]},
			evs[:2],
			map[*watcher][]storagepb.Event{
				ws[0]: evs[:1],
				ws[1]: evs[1:2],
			},
		},
	}

	for i, tt := range tests {
		wg := newWatcherGroup()
		for _, w := range tt.sync {
			wg.add(w)
		}

		gwe := newWatcherBatch(&wg, tt.evs)
		if len(gwe) != len(tt.wwe) {
			t.Errorf("#%d: len(gwe) got = %d, want = %d", i, len(gwe), len(tt.wwe))
		}
		// compare gwe and tt.wwe
		for w, eb := range gwe {
			if len(eb.evs) != len(tt.wwe[w]) {
				t.Errorf("#%d: len(eb.evs) got = %d, want = %d", i, len(eb.evs), len(tt.wwe[w]))
			}
			if !reflect.DeepEqual(eb.evs, tt.wwe[w]) {
				t.Errorf("#%d: reflect.DeepEqual events got = %v, want = true", i, false)
			}
		}
	}
}
