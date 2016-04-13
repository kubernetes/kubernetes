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
	"log"
	"sync"
	"time"

	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/storage/backend"
	"github.com/coreos/etcd/storage/storagepb"
)

const (
	// chanBufLen is the length of the buffered chan
	// for sending out watched events.
	// TODO: find a good buf value. 1024 is just a random one that
	// seems to be reasonable.
	chanBufLen = 1024
)

type watchable interface {
	watch(key, end []byte, startRev int64, id WatchID, ch chan<- WatchResponse) (*watcher, cancelFunc)
	progress(w *watcher)
	rev() int64
}

type watchableStore struct {
	mu sync.Mutex

	*store

	// contains all unsynced watchers that needs to sync with events that have happened
	unsynced watcherGroup

	// contains all synced watchers that are in sync with the progress of the store.
	// The key of the map is the key that the watcher watches on.
	synced watcherGroup

	stopc chan struct{}
	wg    sync.WaitGroup
}

// cancelFunc updates unsynced and synced maps when running
// cancel operations.
type cancelFunc func()

func newWatchableStore(b backend.Backend, le lease.Lessor) *watchableStore {
	s := &watchableStore{
		store:    NewStore(b, le),
		unsynced: newWatcherGroup(),
		synced:   newWatcherGroup(),
		stopc:    make(chan struct{}),
	}
	if s.le != nil {
		// use this store as the deleter so revokes trigger watch events
		s.le.SetRangeDeleter(s)
	}
	s.wg.Add(1)
	go s.syncWatchersLoop()
	return s
}

func (s *watchableStore) Put(key, value []byte, lease lease.LeaseID) (rev int64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	rev = s.store.Put(key, value, lease)
	changes := s.store.getChanges()
	if len(changes) != 1 {
		log.Panicf("unexpected len(changes) != 1 after put")
	}

	ev := storagepb.Event{
		Type: storagepb.PUT,
		Kv:   &changes[0],
	}
	s.notify(rev, []storagepb.Event{ev})
	return rev
}

func (s *watchableStore) DeleteRange(key, end []byte) (n, rev int64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	n, rev = s.store.DeleteRange(key, end)
	changes := s.store.getChanges()

	if len(changes) != int(n) {
		log.Panicf("unexpected len(changes) != n after deleteRange")
	}

	if n == 0 {
		return n, rev
	}

	evs := make([]storagepb.Event, n)
	for i, change := range changes {
		evs[i] = storagepb.Event{
			Type: storagepb.DELETE,
			Kv:   &change}
		evs[i].Kv.ModRevision = rev
	}
	s.notify(rev, evs)
	return n, rev
}

func (s *watchableStore) TxnBegin() int64 {
	s.mu.Lock()
	return s.store.TxnBegin()
}

func (s *watchableStore) TxnEnd(txnID int64) error {
	err := s.store.TxnEnd(txnID)
	if err != nil {
		return err
	}

	changes := s.getChanges()
	if len(changes) == 0 {
		s.mu.Unlock()
		return nil
	}

	rev := s.store.Rev()
	evs := make([]storagepb.Event, len(changes))
	for i, change := range changes {
		switch change.CreateRevision {
		case 0:
			evs[i] = storagepb.Event{
				Type: storagepb.DELETE,
				Kv:   &changes[i]}
			evs[i].Kv.ModRevision = rev
		default:
			evs[i] = storagepb.Event{
				Type: storagepb.PUT,
				Kv:   &changes[i]}
		}
	}

	s.notify(rev, evs)
	s.mu.Unlock()

	return nil
}

func (s *watchableStore) Close() error {
	close(s.stopc)
	s.wg.Wait()
	return s.store.Close()
}

func (s *watchableStore) NewWatchStream() WatchStream {
	watchStreamGauge.Inc()
	return &watchStream{
		watchable: s,
		ch:        make(chan WatchResponse, chanBufLen),
		cancels:   make(map[WatchID]cancelFunc),
		watchers:  make(map[WatchID]*watcher),
	}
}

func (s *watchableStore) watch(key, end []byte, startRev int64, id WatchID, ch chan<- WatchResponse) (*watcher, cancelFunc) {
	s.mu.Lock()
	defer s.mu.Unlock()

	wa := &watcher{
		key: key,
		end: end,
		cur: startRev,
		id:  id,
		ch:  ch,
	}

	s.store.mu.Lock()
	synced := startRev > s.store.currentRev.main || startRev == 0
	if synced {
		wa.cur = s.store.currentRev.main + 1
		if startRev > wa.cur {
			wa.cur = startRev
		}
	}
	s.store.mu.Unlock()
	if synced {
		s.synced.add(wa)
	} else {
		slowWatcherGauge.Inc()
		s.unsynced.add(wa)
	}
	watcherGauge.Inc()

	cancel := cancelFunc(func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		// remove references of the watcher
		if s.unsynced.delete(wa) {
			slowWatcherGauge.Dec()
			watcherGauge.Dec()
			return
		}

		if s.synced.delete(wa) {
			watcherGauge.Dec()
		}
		// If we cannot find it, it should have finished watch.
	})

	return wa, cancel
}

// syncWatchersLoop syncs the watcher in the unsynced map every 100ms.
func (s *watchableStore) syncWatchersLoop() {
	defer s.wg.Done()

	for {
		s.mu.Lock()
		s.syncWatchers()
		s.mu.Unlock()

		select {
		case <-time.After(100 * time.Millisecond):
		case <-s.stopc:
			return
		}
	}
}

// syncWatchers periodically syncs unsynced watchers by: Iterate all unsynced
// watchers to get the minimum revision within its range, skipping the
// watcher if its current revision is behind the compact revision of the
// store. And use this minimum revision to get all key-value pairs. Then send
// those events to watchers.
func (s *watchableStore) syncWatchers() {
	s.store.mu.Lock()
	defer s.store.mu.Unlock()

	if s.unsynced.size() == 0 {
		return
	}

	// in order to find key-value pairs from unsynced watchers, we need to
	// find min revision index, and these revisions can be used to
	// query the backend store of key-value pairs
	curRev := s.store.currentRev.main
	compactionRev := s.store.compactMainRev
	minRev := s.unsynced.scanMinRev(curRev, compactionRev)
	minBytes, maxBytes := newRevBytes(), newRevBytes()
	revToBytes(revision{main: minRev}, minBytes)
	revToBytes(revision{main: curRev + 1}, maxBytes)

	// UnsafeRange returns keys and values. And in boltdb, keys are revisions.
	// values are actual key-value pairs in backend.
	tx := s.store.b.BatchTx()
	tx.Lock()
	revs, vs := tx.UnsafeRange(keyBucketName, minBytes, maxBytes, 0)
	evs := kvsToEvents(&s.unsynced, revs, vs)
	tx.Unlock()

	wb := newWatcherBatch(&s.unsynced, evs)

	for w, eb := range wb {
		select {
		// s.store.Rev also uses Lock, so just return directly
		case w.ch <- WatchResponse{WatchID: w.id, Events: eb.evs, Revision: s.store.currentRev.main}:
			pendingEventsGauge.Add(float64(len(eb.evs)))
		default:
			// TODO: handle the full unsynced watchers.
			// continue to process other watchers for now, the full ones
			// will be processed next time and hopefully it will not be full.
			continue
		}
		if eb.moreRev != 0 {
			w.cur = eb.moreRev
			continue
		}
		w.cur = curRev
		s.synced.add(w)
		s.unsynced.delete(w)
	}

	// bring all un-notified watchers to synced.
	for w := range s.unsynced.watchers {
		if !wb.contains(w) {
			w.cur = curRev
			s.synced.add(w)
			s.unsynced.delete(w)
		}
	}

	slowWatcherGauge.Set(float64(s.unsynced.size()))
}

// kvsToEvents gets all events for the watchers from all key-value pairs
func kvsToEvents(wg *watcherGroup, revs, vals [][]byte) (evs []storagepb.Event) {
	for i, v := range vals {
		var kv storagepb.KeyValue
		if err := kv.Unmarshal(v); err != nil {
			log.Panicf("storage: cannot unmarshal event: %v", err)
		}

		if !wg.contains(string(kv.Key)) {
			continue
		}

		ty := storagepb.PUT
		if isTombstone(revs[i]) {
			ty = storagepb.DELETE
			// patch in mod revision so watchers won't skip
			kv.ModRevision = bytesToRev(revs[i]).main
		}
		evs = append(evs, storagepb.Event{Kv: &kv, Type: ty})
	}
	return evs
}

// notify notifies the fact that given event at the given rev just happened to
// watchers that watch on the key of the event.
func (s *watchableStore) notify(rev int64, evs []storagepb.Event) {
	for w, eb := range newWatcherBatch(&s.synced, evs) {
		if eb.revs != 1 {
			panic("unexpected multiple revisions in notification")
		}
		select {
		case w.ch <- WatchResponse{WatchID: w.id, Events: eb.evs, Revision: s.Rev()}:
			pendingEventsGauge.Add(float64(len(eb.evs)))
		default:
			// move slow watcher to unsynced
			w.cur = rev
			s.unsynced.add(w)
			s.synced.delete(w)
			slowWatcherGauge.Inc()
		}
	}
}

func (s *watchableStore) rev() int64 { return s.store.Rev() }

func (s *watchableStore) progress(w *watcher) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.synced.watchers[w]; ok {
		select {
		case w.ch <- WatchResponse{WatchID: w.id, Revision: s.rev()}:
		default:
			// If the ch is full, this watcher is receiving events.
			// We do not need to send progress at all.
		}
	}
}

type watcher struct {
	// the watcher key
	key []byte
	// end indicates the end of the range to watch.
	// If end is set, the watcher is on a range.
	end []byte

	// cur is the current watcher revision of a unsynced watcher.
	// cur will be updated for unsynced watcher while it is catching up.
	// cur is startRev of a synced watcher.
	// cur will not be updated for synced watcher.
	cur int64
	id  WatchID

	// a chan to send out the watch response.
	// The chan might be shared with other watchers.
	ch chan<- WatchResponse
}
