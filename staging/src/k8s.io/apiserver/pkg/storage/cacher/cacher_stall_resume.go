/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cacher

import (
	"cmp"
	"slices"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	"k8s.io/utils/clock"
)

// Initial values, see KEP-6268.
const (
	// syncScanBudget bounds the history events one pass reads.
	syncScanBudget = 2000
	// syncPushBudget bounds the events one pass pushes. It is soft by the
	// width of one cohort: a scanned event goes to every active member.
	syncPushBudget = syncScanBudget
	// maxWatchersPerSync bounds the members of one pass, as in etcd.
	maxWatchersPerSync = 512
	// syncIntervalOpenCost is the scan budget an interval open takes.
	syncIntervalOpenCost = 64
	// syncPassPeriod is the normal period while any watcher is unsynced.
	syncPassPeriod = 10 * time.Millisecond
	// syncPassIdlePeriod is the period when passes find nothing to do.
	syncPassIdlePeriod = 100 * time.Millisecond
	// A pass that left a member blocked or spent the push budget is followed
	// by syncPassBusyDutyFactor times its duration, at least
	// syncPassBusyPeriod and at most syncPassPeriod.
	syncPassBusyPeriod     = time.Millisecond
	syncPassBusyDutyFactor = 9
)

// cacherStall is the state of the WatchCacheStallResume feature. Unless noted
// otherwise only the dispatcher goroutine touches it.
type cacherStall struct {
	metrics *metrics.StallResumeObservers
	src     *historyCatchUp

	unsynced map[*cacheWatcher]struct{}
	// syncCursor is where the last cohort stopped scanning.
	syncCursor uint64

	// stalled is set when a watcher goes unsynced.
	stalled    bool
	passPeriod time.Duration
	passTimer  clock.Timer
	passArmed  bool
	lastPass   time.Time
	// dispatcherHook runs a test function on the dispatcher goroutine.
	dispatcherHook chan func(runPass syncPassFunc)

	// The fields below live for one pass; their storage is reused.

	// members holds the candidates, sorted by position.
	members []syncMember
	// byScope, byTrigger and triggerFanout index members like the dispatch
	// indexes. Every list is in position order.
	byScope       map[namespacedName][]int
	byTrigger     map[string][]int
	triggerFanout []int
	pushBudget    int
	// lead is the index of the lead of the current cohort, and blockedCount
	// the number of its members that rejected a push.
	lead         int
	blockedCount int
}

func newCacherStall(groupResource schema.GroupResource, cache *watchCache) *cacherStall {
	return &cacherStall{
		metrics:        metrics.NewStallResumeObservers(groupResource),
		src:            &historyCatchUp{cache: cache},
		unsynced:       map[*cacheWatcher]struct{}{},
		passPeriod:     syncPassPeriod,
		dispatcherHook: make(chan func(syncPassFunc)),
		byScope:        map[namespacedName][]int{},
		byTrigger:      map[string][]int{},
	}
}

// syncPassFunc runs one sync pass and reports how many members it served and
// how many watchers it expired.
type syncPassFunc func() (served, expired int)

// syncMember is a candidate of one sync pass.
type syncMember struct {
	w             *cacheWatcher
	startPosition uint64
	served        bool
	// blocked is set once the member rejected a push.
	blocked bool
	// resync is set when the member took all it was offered up to the history end.
	resync bool
}

// scannedEvent is one history event of a cohort scan.
type scannedEvent struct {
	raw *watchCacheEvent
	// wrapped is the dispatch copy all members share.
	wrapped *watchCacheEvent
}

// historyCatchUp reads the watch cache event history for the sync passes.
type historyCatchUp struct {
	cache *watchCache
}

// intervalSince returns the events newer than resourceVersion, or an error if
// the history does not cover it.
func (r *historyCatchUp) intervalSince(resourceVersion uint64) (*watchCacheInterval, error) {
	r.cache.RLock()
	defer r.cache.RUnlock()
	return r.cache.history.GetIntervalLocked(resourceVersion, r.cache.RWMutex.RLocker())
}

// oldest returns the oldest resourceVersion intervalSince can serve.
func (r *historyCatchUp) oldest() (uint64, bool) {
	r.cache.RLock()
	defer r.cache.RUnlock()
	return r.cache.history.oldestServableLocked()
}

// dispatchNonblocking replaces the body of dispatchEvent with the gate on. It
// never waits for a watcher and never terminates one: a watcher with a full
// input becomes unsynced and sync passes serve it from the history.
func (c *Cacher) dispatchNonblocking(event *watchCacheEvent) {
	c.startDispatching(event)
	defer c.finishDispatching()

	if event.Type == watch.Bookmark {
		for _, watcher := range c.watchersBuffer {
			// A bookmark below the position would move the client back.
			if watcher.unsynced || event.ResourceVersion < watcher.position {
				continue
			}
			if watcher.nonblockingAdd(event) {
				watcher.position = event.ResourceVersion
			}
		}
		return
	}

	wcEvent := *event
	setCachingObjects(&wcEvent, c.versioner)
	wcEvent.timeline.MarkAt(metrics.PointWatcherEnqueued, c.clock.Now())
	for _, watcher := range c.watchersBuffer {
		if watcher.unsynced || wcEvent.ResourceVersion <= watcher.position {
			continue
		}
		if watcher.nonblockingAdd(&wcEvent) {
			watcher.position = wcEvent.ResourceVersion
			continue
		}
		watcher.position = wcEvent.ResourceVersion - 1
		watcher.unsynced = true
		c.stall.unsynced[watcher] = struct{}{}
		c.stall.stalled = true
		c.stall.metrics.Stalls.Inc()
	}
}

// syncAfterDispatch runs a sync pass when one is due and keeps the pass timer
// armed while any watcher is unsynced. It returns the channel to select on.
func (c *Cacher) syncAfterDispatch() <-chan time.Time {
	s := c.stall
	rearm := false
	stalled := s.stalled
	if stalled {
		s.stalled = false
		// A stall ends the idle period and keeps a shorter one.
		if s.passPeriod > syncPassPeriod {
			s.passPeriod = syncPassPeriod
			rearm = true
		}
	}
	if len(s.unsynced) == 0 {
		return c.armSyncTimer()
	}
	now := c.clock.Now()
	if now.Sub(s.lastPass) >= s.passPeriod {
		c.syncPass(now)
		if stalled && s.passPeriod == syncPassIdlePeriod {
			// This pass found the new member with a full input.
			s.passPeriod = syncPassPeriod
		}
		return c.armSyncTimer()
	}
	if s.passArmed && !rearm {
		return s.passTimer.C()
	}
	return c.armSyncTimer()
}

// armSyncTimer arms the pass timer for the current period, or stops it when
// nothing is unsynced.
func (c *Cacher) armSyncTimer() <-chan time.Time {
	s := c.stall
	if s.passArmed {
		s.passTimer.Stop()
		s.passArmed = false
	}
	if len(s.unsynced) == 0 {
		return nil
	}
	s.passTimer.Reset(s.passPeriod)
	s.passArmed = true
	return s.passTimer.C()
}

// syncPass serves the unsynced watchers from the history, like etcd's
// syncWatchers. It never holds the Cacher lock while it takes the watch cache
// lock, because Watch takes them in the other order.
func (c *Cacher) syncPass(now time.Time) (served, expired int) {
	s := c.stall
	s.lastPass = now
	oldest, ok := s.src.oldest()
	if !ok {
		return 0, 0
	}
	expired = c.selectSyncCandidates(oldest)
	c.buildSyncIndexes()

	budget := syncScanBudget
	s.pushBudget = syncPushBudget
	for budget >= syncIntervalOpenCost && s.pushBudget > 0 {
		lead := s.pickSyncLead()
		if lead < 0 {
			break
		}
		c.serveSyncCohort(lead, &budget, now)
	}
	s.metrics.DeferredEvents.Add(float64(syncPushBudget - s.pushBudget))

	busy := s.pushBudget <= 0
	c.Lock()
	for i := range s.members {
		m := &s.members[i]
		if m.served {
			served++
		}
		busy = busy || m.blocked
		if !m.resync || m.w.stopped {
			continue
		}
		m.w.unsynced = false
		delete(s.unsynced, m.w)
		s.metrics.CatchupEvents.Observe(float64(m.w.catchupEvents))
		m.w.catchupEvents = 0
	}
	c.Unlock()
	c.finishDispatching()

	clear(s.members)
	s.members = s.members[:0]
	s.passPeriod = nextPassPeriod(served, expired, busy, c.clock.Since(now), s.passPeriod)
	return served, expired
}

// nextPassPeriod picks the period after a pass. A pass with nothing to do
// after a short period does not back off yet: a slow reader that was blocked
// gets syncPassPeriod to free a slot before the idle period.
func nextPassPeriod(served, expired int, busy bool, passDuration, previous time.Duration) time.Duration {
	switch {
	case busy:
		return min(syncPassPeriod, max(syncPassBusyPeriod, syncPassBusyDutyFactor*passDuration))
	case served == 0 && expired == 0 && previous >= syncPassPeriod:
		return syncPassIdlePeriod
	default:
		return syncPassPeriod
	}
}

// selectSyncCandidates starts a pass. It marks the Cacher as dispatching,
// drops the stopped watchers, expires those below oldest and fills s.members
// with the unsynced watchers that have a free input slot, sorted by position.
// It returns the number of expired watchers.
func (c *Cacher) selectSyncCandidates(oldest uint64) int {
	s := c.stall
	var expired []*cacheWatcher
	c.Lock()
	// A client Stop during the pass must not close an input the pass pushes to.
	c.dispatching = true
	for w := range s.unsynced {
		if w.stopped {
			delete(s.unsynced, w)
			continue
		}
		// The same boundary as GetIntervalLocked.
		if w.position+1 < oldest {
			w.expired = true
			if w.live.Load() {
				w.expiredReason = metrics.TerminationReasonResourceExpired
			} else {
				w.expiredReason = metrics.TerminationReasonResourceExpiredInitial
			}
			delete(s.unsynced, w)
			expired = append(expired, w)
			continue
		}
		if len(w.input) < cap(w.input) {
			s.members = append(s.members, syncMember{w: w, startPosition: w.position})
		}
	}
	c.Unlock()

	// A draining stop, deferred to finishDispatching: the watcher goroutine
	// delivers the backlog and then the 410.
	for _, w := range expired {
		w.forget(true)
	}

	if len(s.members) > maxWatchersPerSync {
		// Keep the positions at or above the cursor first, then the lowest
		// ones: the unsigned subtraction wraps.
		slices.SortFunc(s.members, func(a, b syncMember) int {
			return cmp.Compare(a.startPosition-s.syncCursor, b.startPosition-s.syncCursor)
		})
		clear(s.members[maxWatchersPerSync:])
		s.members = s.members[:maxWatchersPerSync]
	}
	slices.SortFunc(s.members, func(a, b syncMember) int {
		return cmp.Compare(a.startPosition, b.startPosition)
	})
	return len(expired)
}

// buildSyncIndexes builds the lookup maps of the pass.
func (c *Cacher) buildSyncIndexes() {
	s := c.stall
	clear(s.byScope)
	clear(s.byTrigger)
	s.triggerFanout = s.triggerFanout[:0]
	for i := range s.members {
		selector := s.members[i].w.selector
		if selector.indexValue != nil {
			s.byTrigger[*selector.indexValue] = append(s.byTrigger[*selector.indexValue], i)
			s.triggerFanout = append(s.triggerFanout, i)
			continue
		}
		s.byScope[selector.scope] = append(s.byScope[selector.scope], i)
	}
}

// pickSyncLead returns the first unserved member at or above the cursor, else
// the first unserved member, or -1. A candidate below the cursor therefore
// waits until no unserved candidate is left at or above it and budget remains;
// only the history window bounds that wait.
func (s *cacherStall) pickSyncLead() int {
	first := -1
	for i := range s.members {
		m := &s.members[i]
		if m.served {
			continue
		}
		if m.startPosition >= s.syncCursor {
			return i
		}
		if first < 0 {
			first = i
		}
	}
	return first
}

// serveSyncCohort opens one history interval at the lead's position and offers
// every scanned event to the cohort: the lead and the unserved members above
// it whose start position is below the scanned event.
func (c *Cacher) serveSyncCohort(lead int, budget *int, now time.Time) {
	s := c.stall
	members := s.members
	*budget -= syncIntervalOpenCost
	lastScannedRV := members[lead].startPosition
	interval, err := s.src.intervalSince(lastScannedRV)
	if err != nil {
		// The history moved past the lead: the next pass expires it.
		members[lead].served = true
		return
	}

	// The cohort is members[lead:end) less the ones already served.
	s.lead, s.blockedCount = lead, 0
	end, active := lead+1, 1
	historyEnd := true
	for ev, err := range interval.All() {
		// Checked with the next event in hand, so a scan that stops on the
		// last event still learns that it reached the history end.
		if err != nil || *budget == 0 || s.pushBudget <= 0 || s.blockedCount == active {
			historyEnd = false
			break
		}
		*budget--
		lastScannedRV = ev.ResourceVersion
		for ; end < len(members) && members[end].startPosition < lastScannedRV; end++ {
			if !members[end].served {
				active++
			}
		}
		c.offerScanned(&scannedEvent{raw: ev}, now)
	}

	for i := lead; i < end; i++ {
		m := &members[i]
		if m.served {
			continue
		}
		m.served = true
		if m.blocked {
			continue
		}
		// Everything up to lastScannedRV was accepted or not selected.
		m.w.position = lastScannedRV
		m.resync = historyEnd
	}
	s.syncCursor = lastScannedRV
}

// offerScanned offers the event to the cohort members that select it.
func (c *Cacher) offerScanned(se *scannedEvent, now time.Time) {
	s := c.stall
	triggerValues, supported := c.triggerValuesThreadUnsafe(se.raw)
	forEachSelectionKey(se.raw.ObjFields["metadata.namespace"], se.raw.ObjFields["metadata.name"], triggerValues, supported,
		func(key namespacedName) { c.offerToAll(s.byScope[key], se, now) },
		func(value string) { c.offerToAll(s.byTrigger[value], se, now) },
		func() { c.offerToAll(s.triggerFanout, se, now) })
}

// offerToAll offers the event to the cohort members among hits. A rejected push
// ends the offering to that member for this pass.
func (c *Cacher) offerToAll(hits []int, se *scannedEvent, now time.Time) {
	s := c.stall
	for _, i := range hits {
		m := &s.members[i]
		if m.startPosition >= se.raw.ResourceVersion {
			return
		}
		if i < s.lead || m.served || m.blocked {
			continue
		}
		if !c.pushScanned(m.w, se, now) {
			m.blocked = true
			s.blockedCount++
		}
	}
}

// pushScanned wraps the event once, like the live path, and pushes the shared
// copy to the watcher.
func (c *Cacher) pushScanned(w *cacheWatcher, se *scannedEvent, now time.Time) bool {
	if se.wrapped == nil {
		wcEvent := *se.raw
		setCachingObjects(&wcEvent, c.versioner)
		wcEvent.timeline.MarkAt(metrics.PointWatcherEnqueued, now)
		se.wrapped = &wcEvent
	}
	if !w.nonblockingAdd(se.wrapped) {
		return false
	}
	w.position = se.raw.ResourceVersion
	w.catchupEvents++
	c.stall.pushBudget--
	return true
}
