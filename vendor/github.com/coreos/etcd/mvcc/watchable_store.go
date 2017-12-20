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
	"sync"
	"time"

	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/mvcc/backend"
	"github.com/coreos/etcd/mvcc/mvccpb"
)

const (
	// chanBufLen is the length of the buffered chan
	// for sending out watched events.
	// TODO: find a good buf value. 1024 is just a random one that
	// seems to be reasonable.
	chanBufLen = 1024

	// maxWatchersPerSync is the number of watchers to sync in a single batch
	maxWatchersPerSync = 512
)

type watchable interface {
	watch(key, end []byte, startRev int64, id WatchID, ch chan<- WatchResponse, fcs ...FilterFunc) (*watcher, cancelFunc)
	progress(w *watcher)
	rev() int64
}

type watchableStore struct {
	mu sync.Mutex

	*store

	// victims are watcher batches that were blocked on the watch channel
	victims []watcherBatch
	victimc chan struct{}

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

func New(b backend.Backend, le lease.Lessor, ig ConsistentIndexGetter) ConsistentWatchableKV {
	return newWatchableStore(b, le, ig)
}

func newWatchableStore(b backend.Backend, le lease.Lessor, ig ConsistentIndexGetter) *watchableStore {
	s := &watchableStore{
		store:    NewStore(b, le, ig),
		victimc:  make(chan struct{}, 1),
		unsynced: newWatcherGroup(),
		synced:   newWatcherGroup(),
		stopc:    make(chan struct{}),
	}
	if s.le != nil {
		// use this store as the deleter so revokes trigger watch events
		s.le.SetRangeDeleter(s)
	}
	s.wg.Add(2)
	go s.syncWatchersLoop()
	go s.syncVictimsLoop()
	return s
}

func (s *watchableStore) Put(key, value []byte, lease lease.LeaseID) (rev int64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	rev = s.store.Put(key, value, lease)
	changes := s.store.getChanges()
	if len(changes) != 1 {
		plog.Panicf("unexpected len(changes) != 1 after put")
	}

	ev := mvccpb.Event{
		Type: mvccpb.PUT,
		Kv:   &changes[0],
	}
	s.notify(rev, []mvccpb.Event{ev})
	return rev
}

func (s *watchableStore) DeleteRange(key, end []byte) (n, rev int64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	n, rev = s.store.DeleteRange(key, end)
	changes := s.store.getChanges()

	if len(changes) != int(n) {
		plog.Panicf("unexpected len(changes) != n after deleteRange")
	}

	if n == 0 {
		return n, rev
	}

	evs := make([]mvccpb.Event, n)
	for i := range changes {
		evs[i] = mvccpb.Event{
			Type: mvccpb.DELETE,
			Kv:   &changes[i]}
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
	evs := make([]mvccpb.Event, len(changes))
	for i, change := range changes {
		switch change.CreateRevision {
		case 0:
			evs[i] = mvccpb.Event{
				Type: mvccpb.DELETE,
				Kv:   &changes[i]}
			evs[i].Kv.ModRevision = rev
		default:
			evs[i] = mvccpb.Event{
				Type: mvccpb.PUT,
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

func (s *watchableStore) watch(key, end []byte, startRev int64, id WatchID, ch chan<- WatchResponse, fcs ...FilterFunc) (*watcher, cancelFunc) {
	s.mu.Lock()
	defer s.mu.Unlock()

	wa := &watcher{
		key:    key,
		end:    end,
		minRev: startRev,
		id:     id,
		ch:     ch,
		fcs:    fcs,
	}

	s.store.mu.Lock()
	synced := startRev > s.store.currentRev.main || startRev == 0
	if synced {
		wa.minRev = s.store.currentRev.main + 1
		if startRev > wa.minRev {
			wa.minRev = startRev
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

	return wa, func() { s.cancelWatcher(wa) }
}

// cancelWatcher removes references of the watcher from the watchableStore
func (s *watchableStore) cancelWatcher(wa *watcher) {
	for {
		s.mu.Lock()

		if s.unsynced.delete(wa) {
			slowWatcherGauge.Dec()
			break
		} else if s.synced.delete(wa) {
			break
		} else if wa.compacted {
			break
		}

		if !wa.victim {
			panic("watcher not victim but not in watch groups")
		}

		var victimBatch watcherBatch
		for _, wb := range s.victims {
			if wb[wa] != nil {
				victimBatch = wb
				break
			}
		}
		if victimBatch != nil {
			slowWatcherGauge.Dec()
			delete(victimBatch, wa)
			break
		}

		// victim being processed so not accessible; retry
		s.mu.Unlock()
		time.Sleep(time.Millisecond)
	}

	watcherGauge.Dec()
	s.mu.Unlock()
}

// syncWatchersLoop syncs the watcher in the unsynced map every 100ms.
func (s *watchableStore) syncWatchersLoop() {
	defer s.wg.Done()

	for {
		s.mu.Lock()
		st := time.Now()
		lastUnsyncedWatchers := s.unsynced.size()
		s.syncWatchers()
		unsyncedWatchers := s.unsynced.size()
		s.mu.Unlock()
		syncDuration := time.Since(st)

		waitDuration := 100 * time.Millisecond
		// more work pending?
		if unsyncedWatchers != 0 && lastUnsyncedWatchers > unsyncedWatchers {
			// be fair to other store operations by yielding time taken
			waitDuration = syncDuration
		}

		select {
		case <-time.After(waitDuration):
		case <-s.stopc:
			return
		}
	}
}

// syncVictimsLoop tries to write precomputed watcher responses to
// watchers that had a blocked watcher channel
func (s *watchableStore) syncVictimsLoop() {
	defer s.wg.Done()

	for {
		for s.moveVictims() != 0 {
			// try to update all victim watchers
		}
		s.mu.Lock()
		isEmpty := len(s.victims) == 0
		s.mu.Unlock()

		var tickc <-chan time.Time
		if !isEmpty {
			tickc = time.After(10 * time.Millisecond)
		}

		select {
		case <-tickc:
		case <-s.victimc:
		case <-s.stopc:
			return
		}
	}
}

// moveVictims tries to update watches with already pending event data
func (s *watchableStore) moveVictims() (moved int) {
	s.mu.Lock()
	victims := s.victims
	s.victims = nil
	s.mu.Unlock()

	var newVictim watcherBatch
	for _, wb := range victims {
		// try to send responses again
		for w, eb := range wb {
			// watcher has observed the store up to, but not including, w.minRev
			rev := w.minRev - 1
			if w.send(WatchResponse{WatchID: w.id, Events: eb.evs, Revision: rev}) {
				pendingEventsGauge.Add(float64(len(eb.evs)))
			} else {
				if newVictim == nil {
					newVictim = make(watcherBatch)
				}
				newVictim[w] = eb
				continue
			}
			moved++
		}

		// assign completed victim watchers to unsync/sync
		s.mu.Lock()
		s.store.mu.Lock()
		curRev := s.store.currentRev.main
		for w, eb := range wb {
			if newVictim != nil && newVictim[w] != nil {
				// couldn't send watch response; stays victim
				continue
			}
			w.victim = false
			if eb.moreRev != 0 {
				w.minRev = eb.moreRev
			}
			if w.minRev <= curRev {
				s.unsynced.add(w)
			} else {
				slowWatcherGauge.Dec()
				s.synced.add(w)
			}
		}
		s.store.mu.Unlock()
		s.mu.Unlock()
	}

	if len(newVictim) > 0 {
		s.mu.Lock()
		s.victims = append(s.victims, newVictim)
		s.mu.Unlock()
	}

	return moved
}

// syncWatchers syncs unsynced watchers by:
//	1. choose a set of watchers from the unsynced watcher group
//	2. iterate over the set to get the minimum revision and remove compacted watchers
//	3. use minimum revision to get all key-value pairs and send those events to watchers
//	4. remove synced watchers in set from unsynced group and move to synced group
func (s *watchableStore) syncWatchers() {
	if s.unsynced.size() == 0 {
		return
	}

	s.store.mu.Lock()
	defer s.store.mu.Unlock()

	// in order to find key-value pairs from unsynced watchers, we need to
	// find min revision index, and these revisions can be used to
	// query the backend store of key-value pairs
	curRev := s.store.currentRev.main
	compactionRev := s.store.compactMainRev
	wg, minRev := s.unsynced.choose(maxWatchersPerSync, curRev, compactionRev)
	minBytes, maxBytes := newRevBytes(), newRevBytes()
	revToBytes(revision{main: minRev}, minBytes)
	revToBytes(revision{main: curRev + 1}, maxBytes)

	// UnsafeRange returns keys and values. And in boltdb, keys are revisions.
	// values are actual key-value pairs in backend.
	tx := s.store.b.BatchTx()
	tx.Lock()
	revs, vs := tx.UnsafeRange(keyBucketName, minBytes, maxBytes, 0)
	evs := kvsToEvents(wg, revs, vs)
	tx.Unlock()

	var victims watcherBatch
	wb := newWatcherBatch(wg, evs)
	for w := range wg.watchers {
		w.minRev = curRev + 1

		eb, ok := wb[w]
		if !ok {
			// bring un-notified watcher to synced
			s.synced.add(w)
			s.unsynced.delete(w)
			continue
		}

		if eb.moreRev != 0 {
			w.minRev = eb.moreRev
		}

		if w.send(WatchResponse{WatchID: w.id, Events: eb.evs, Revision: curRev}) {
			pendingEventsGauge.Add(float64(len(eb.evs)))
		} else {
			if victims == nil {
				victims = make(watcherBatch)
			}
			w.victim = true
		}

		if w.victim {
			victims[w] = eb
		} else {
			if eb.moreRev != 0 {
				// stay unsynced; more to read
				continue
			}
			s.synced.add(w)
		}
		s.unsynced.delete(w)
	}
	s.addVictim(victims)

	vsz := 0
	for _, v := range s.victims {
		vsz += len(v)
	}
	slowWatcherGauge.Set(float64(s.unsynced.size() + vsz))
}

// kvsToEvents gets all events for the watchers from all key-value pairs
func kvsToEvents(wg *watcherGroup, revs, vals [][]byte) (evs []mvccpb.Event) {
	for i, v := range vals {
		var kv mvccpb.KeyValue
		if err := kv.Unmarshal(v); err != nil {
			plog.Panicf("cannot unmarshal event: %v", err)
		}

		if !wg.contains(string(kv.Key)) {
			continue
		}

		ty := mvccpb.PUT
		if isTombstone(revs[i]) {
			ty = mvccpb.DELETE
			// patch in mod revision so watchers won't skip
			kv.ModRevision = bytesToRev(revs[i]).main
		}
		evs = append(evs, mvccpb.Event{Kv: &kv, Type: ty})
	}
	return evs
}

// notify notifies the fact that given event at the given rev just happened to
// watchers that watch on the key of the event.
func (s *watchableStore) notify(rev int64, evs []mvccpb.Event) {
	var victim watcherBatch
	for w, eb := range newWatcherBatch(&s.synced, evs) {
		if eb.revs != 1 {
			plog.Panicf("unexpected multiple revisions in notification")
		}

		if w.send(WatchResponse{WatchID: w.id, Events: eb.evs, Revision: rev}) {
			pendingEventsGauge.Add(float64(len(eb.evs)))
		} else {
			// move slow watcher to victims
			w.minRev = rev + 1
			if victim == nil {
				victim = make(watcherBatch)
			}
			w.victim = true
			victim[w] = eb
			s.synced.delete(w)
			slowWatcherGauge.Inc()
		}
	}
	s.addVictim(victim)
}

func (s *watchableStore) addVictim(victim watcherBatch) {
	if victim == nil {
		return
	}
	s.victims = append(s.victims, victim)
	select {
	case s.victimc <- struct{}{}:
	default:
	}
}

func (s *watchableStore) rev() int64 { return s.store.Rev() }

func (s *watchableStore) progress(w *watcher) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.synced.watchers[w]; ok {
		w.send(WatchResponse{WatchID: w.id, Revision: s.rev()})
		// If the ch is full, this watcher is receiving events.
		// We do not need to send progress at all.
	}
}

type watcher struct {
	// the watcher key
	key []byte
	// end indicates the end of the range to watch.
	// If end is set, the watcher is on a range.
	end []byte

	// victim is set when ch is blocked and undergoing victim processing
	victim bool

	// compacted is set when the watcher is removed because of compaction
	compacted bool

	// minRev is the minimum revision update the watcher will accept
	minRev int64
	id     WatchID

	fcs []FilterFunc
	// a chan to send out the watch response.
	// The chan might be shared with other watchers.
	ch chan<- WatchResponse
}

func (w *watcher) send(wr WatchResponse) bool {
	progressEvent := len(wr.Events) == 0

	if len(w.fcs) != 0 {
		ne := make([]mvccpb.Event, 0, len(wr.Events))
		for i := range wr.Events {
			filtered := false
			for _, filter := range w.fcs {
				if filter(wr.Events[i]) {
					filtered = true
					break
				}
			}
			if !filtered {
				ne = append(ne, wr.Events[i])
			}
		}
		wr.Events = ne
	}

	// if all events are filtered out, we should send nothing.
	if !progressEvent && len(wr.Events) == 0 {
		return true
	}
	select {
	case w.ch <- wr:
		return true
	default:
		return false
	}
}
