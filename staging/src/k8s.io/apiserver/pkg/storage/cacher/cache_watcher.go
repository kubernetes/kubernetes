/*
Copyright 2023 The Kubernetes Authors.

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
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"k8s.io/utils/clock"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/metrics"
	utilflowcontrol "k8s.io/apiserver/pkg/util/flowcontrol"

	"k8s.io/klog/v2"
)

// possible states of the cache watcher
const (
	// cacheWatcherWaitingForBookmark indicates the cacher
	// is waiting for a bookmark event with a specific RV set
	cacheWatcherWaitingForBookmark = iota

	// cacheWatcherBookmarkReceived indicates that the cacher
	// has received a bookmark event with required RV
	cacheWatcherBookmarkReceived

	// cacheWatcherBookmarkSent indicates that the cacher
	// has already sent a bookmark event to a client
	cacheWatcherBookmarkSent
)

// cacheWatcher implements watch.Interface
// this is not thread-safe
type cacheWatcher struct {
	input     chan *watchCacheEvent
	result    chan watch.Event
	done      chan struct{}
	filter    filterWithAttrsFunc
	stopped   bool
	forget    func(bool)
	versioner storage.Versioner
	// The watcher will be closed by server after the deadline,
	// save it here to send bookmark events before that.
	deadline            time.Time
	allowWatchBookmarks bool
	groupResource       schema.GroupResource
	watcherMetrics      *metrics.WatcherMetricsObservers
	clock               clock.Clock

	// human readable identifier that helps assigning cacheWatcher
	// instance with request
	identifier string

	// drainInputBuffer indicates whether we should delay closing this watcher
	// and send all event in the input buffer.
	drainInputBuffer bool

	// bookmarkAfterResourceVersion holds an RV that indicates
	// when we should start delivering bookmark events.
	// If this field holds the value of 0 that means
	// we don't have any special preferences toward delivering bookmark events.
	// Note that this field is used in conjunction with the state field.
	// It should not be changed once the watcher has been started.
	bookmarkAfterResourceVersion uint64

	// stateMutex protects state
	stateMutex sync.Mutex

	// state holds a numeric value indicating the current state of the watcher
	state int

	// stall is non-nil exactly when the WatchCacheStallResume feature gate
	// is enabled (see enableStallResume); when nil the watcher behaves
	// exactly as before the gate existed.
	stall *watcherStall
}

// watcherStall is the per-watcher stall-and-resume state; a watcher holds
// one exactly when the WatchCacheStallResume gate is on, making the nil
// check the single mode representation.
type watcherStall struct {
	// ch is a capacity-1 latch carrying the resourceVersion of the first
	// event of the current stall episode that did not fit into the input
	// channel. The dispatcher pokes it (non-blocking send, on the dispatch
	// goroutine, in program order after the failed nonblockingAdd); the
	// processing goroutine consumes it and catches up from the watch cache
	// history starting just below that resourceVersion. Later misses of the
	// same episode coalesce onto the pending token: they lie above it, so
	// the round it triggers covers them.
	//
	// It is never closed: the dispatcher may poke a watcher that is
	// concurrently stopping, and a send on a closed channel would panic.
	ch chan uint64

	// src serves the catch-up intervals; it reads only the watch cache
	// event history, never the store.
	src *ringCatchUp

	// metrics holds the pre-resolved metric children shared by all
	// watchers of one Cacher.
	metrics *metrics.StallResumeObservers

	// start is the position the live loop was entered with (the resume
	// position before any live delivery); only the watcher goroutine uses it.
	start uint64
}

func newCacheWatcher(
	chanSize int,
	filter filterWithAttrsFunc,
	forget func(bool),
	versioner storage.Versioner,
	deadline time.Time,
	allowWatchBookmarks bool,
	groupResource schema.GroupResource,
	watcherMetrics *metrics.WatcherMetricsObservers,
	clk clock.Clock,
	identifier string,
) *cacheWatcher {
	if clk == nil {
		clk = clock.RealClock{}
	}
	return &cacheWatcher{
		input:               make(chan *watchCacheEvent, chanSize),
		result:              make(chan watch.Event, chanSize),
		done:                make(chan struct{}),
		filter:              filter,
		stopped:             false,
		forget:              forget,
		versioner:           versioner,
		deadline:            deadline,
		allowWatchBookmarks: allowWatchBookmarks,
		groupResource:       groupResource,
		watcherMetrics:      watcherMetrics,
		clock:               clk,
		identifier:          identifier,
	}
}

// enableStallResume switches the watcher to stall-and-resume mode: instead of
// being terminated when its input channel fills up, it is poked and catches
// up from the watch cache history. It must be called before the watcher is
// registered for dispatch.
func (c *cacheWatcher) enableStallResume(src *ringCatchUp, observers *metrics.StallResumeObservers) {
	c.stall = &watcherStall{
		ch:      make(chan uint64, 1),
		src:     src,
		metrics: observers,
	}
}

// poke records that the event at resourceVersion did not fit into this
// watcher's input channel and must instead be served by a catch-up round
// from the watch cache history. It sets the stall latch to that
// resourceVersion if no token is pending; otherwise the pending token
// (an earlier miss) already covers it.
//
// It must run synchronously on the dispatching goroutine, in program order
// after the failed nonblockingAdd, and the latch must stay a capacity-1
// channel: the ordering argument in processStallResume depends on both.
func (c *cacheWatcher) poke(resourceVersion uint64) {
	c.stall.metrics.DeferredEvents.Inc()
	select {
	case c.stall.ch <- resourceVersion:
		// The latch was empty: this failure starts a new stall episode.
		c.stall.metrics.Stalls.Inc()
	default:
		// A token is already pending; the failure coalesces onto it.
	}
}

// consumeLatch takes the pending stall token if there is one, returning the
// resourceVersion of the first missed event of the episode.
func (c *cacheWatcher) consumeLatch() (uint64, bool) {
	select {
	case missed := <-c.stall.ch:
		return missed, true
	default:
		return 0, false
	}
}

// Implements watch.Interface.
func (c *cacheWatcher) ResultChan() <-chan watch.Event {
	return c.result
}

// Implements watch.Interface.
func (c *cacheWatcher) Stop() {
	c.forget(false)
}

// we rely on the fact that stopLocked is actually protected by Cacher.Lock()
func (c *cacheWatcher) stopLocked() {
	if !c.stopped {
		c.stopped = true
		// stop without draining the input channel was requested.
		if !c.drainInputBuffer {
			close(c.done)
		}
		close(c.input)
	}

	// Even if the watcher was already stopped, if it previously was
	// using draining mode and it's not using it now we need to
	// close the done channel now. Otherwise we could leak the
	// processing goroutine if it will be trying to put more objects
	// into result channel, the channel will be full and there will
	// already be noone on the processing the events on the receiving end.
	if !c.drainInputBuffer && !c.isDoneChannelClosedLocked() {
		close(c.done)
	}
}

func (c *cacheWatcher) nonblockingAdd(event *watchCacheEvent) bool {
	// if the bookmarkAfterResourceVersion hasn't been seen
	// we will try to deliver a bookmark event every second.
	// the following check will discard a bookmark event
	// if it is < than the bookmarkAfterResourceVersion
	// so that we don't pollute the input channel
	if event.Type == watch.Bookmark && event.ResourceVersion < c.bookmarkAfterResourceVersion {
		return false
	}
	select {
	case c.input <- event:
		c.markBookmarkAfterRvAsReceived(event)
		return true
	default:
		return false
	}
}

// Nil timer means that add will not block (if it can't send event immediately, it will break the watcher)
//
// Note that bookmark events are never added via the add method only via the nonblockingAdd.
// Changing this behaviour will require moving the markBookmarkAfterRvAsReceived method
func (c *cacheWatcher) add(event *watchCacheEvent, timer *time.Timer) bool {
	// Try to send the event immediately, without blocking.
	if c.nonblockingAdd(event) {
		return true
	}

	closeFunc := func() {
		// This means that we couldn't send event to that watcher.
		// Since we don't want to block on it infinitely,
		// we simply terminate it.
		metrics.TerminatedWatchersCounter.WithLabelValues(c.groupResource.Group, c.groupResource.Resource, metrics.TerminationReasonUnresponsive).Inc()
		// This means that we couldn't send event to that watcher.
		// Since we don't want to block on it infinitely, we simply terminate it.

		// we are graceful = false, when:
		//
		// (a) The bookmarkAfterResourceVersionReceived hasn't been received,
		//     we can safely terminate the watcher. Because the client is waiting
		//     for this specific bookmark, and we even haven't received one.
		// (b) We have seen the bookmarkAfterResourceVersion, and it was sent already to the client.
		//     We can simply terminate the watcher.

		// we are graceful = true, when:
		//
		// (a) We have seen a bookmark, but it hasn't been sent to the client yet.
		//     That means we should drain the input buffer which contains
		//     the bookmarkAfterResourceVersion we want. We do that to make progress
		//     as clients can re-establish a new watch with the given RV and receive
		//     further notifications.
		graceful := func() bool {
			c.stateMutex.Lock()
			defer c.stateMutex.Unlock()
			return c.state == cacheWatcherBookmarkReceived
		}()
		klog.V(1).Infof("Forcing %v watcher close due to unresponsiveness (reason = %v): %v. len(c.input) = %v, len(c.result) = %v, graceful = %v", c.groupResource.String(), metrics.TerminationReasonUnresponsive, c.identifier, len(c.input), len(c.result), graceful)
		c.forget(graceful)
	}

	if timer == nil {
		closeFunc()
		return false
	}

	// OK, block sending, but only until timer fires.
	select {
	case c.input <- event:
		return true
	case <-timer.C:
		closeFunc()
		return false
	}
}

func (c *cacheWatcher) nextBookmarkTime(now time.Time, bookmarkFrequency time.Duration) (time.Time, bool) {
	// We try to send bookmarks:
	//
	// (a) right before the watcher timeout - for now we simply set it 2s before
	//     the deadline
	//
	// (b) roughly every minute
	//
	// (c) immediately when the bookmarkAfterResourceVersion wasn't confirmed
	//     in this scenario the client have already seen (or is in the process of sending)
	//     all initial data and is interested in seeing
	//     a specific RV value (aka. the bookmarkAfterResourceVersion)
	//     since we don't know when the cacher will see the RV we increase frequency
	//
	// (b) gives us periodicity if the watch breaks due to unexpected
	// conditions, (a) ensures that on timeout the watcher is as close to
	// now as possible - this covers 99% of cases.

	if !c.wasBookmarkAfterRvReceived() {
		return time.Time{}, true // schedule immediately
	}

	heartbeatTime := now.Add(bookmarkFrequency)
	if c.deadline.IsZero() {
		// Timeout is set by our client libraries (e.g. reflector) as well as defaulted by
		// apiserver if properly configured. So this shoudln't happen in practice.
		return heartbeatTime, true
	}
	if pretimeoutTime := c.deadline.Add(-2 * time.Second); pretimeoutTime.Before(heartbeatTime) {
		heartbeatTime = pretimeoutTime
	}

	if heartbeatTime.Before(now) {
		return time.Time{}, false
	}
	return heartbeatTime, true
}

// wasBookmarkAfterRvReceived same as wasBookmarkAfterRvReceivedLocked just acquires a lock
func (c *cacheWatcher) wasBookmarkAfterRvReceived() bool {
	c.stateMutex.Lock()
	defer c.stateMutex.Unlock()
	return c.wasBookmarkAfterRvReceivedLocked()
}

// wasBookmarkAfterRvReceivedLocked checks if the given cacheWatcher
// have seen a bookmark event >= bookmarkAfterResourceVersion
func (c *cacheWatcher) wasBookmarkAfterRvReceivedLocked() bool {
	return c.state != cacheWatcherWaitingForBookmark
}

// markBookmarkAfterRvAsReceived indicates that the given cacheWatcher
// have seen a bookmark event >= bookmarkAfterResourceVersion
func (c *cacheWatcher) markBookmarkAfterRvAsReceived(event *watchCacheEvent) {
	if event.Type == watch.Bookmark {
		c.stateMutex.Lock()
		defer c.stateMutex.Unlock()
		if c.wasBookmarkAfterRvReceivedLocked() {
			return
		}
		// bookmark events are scheduled by startDispatchingBookmarkEvents method
		// since we received a bookmark event that means we have
		// converged towards the expected RV and it is okay to update the state so that
		// this cacher can be scheduler for a regular bookmark events
		c.state = cacheWatcherBookmarkReceived
	}
}

// wasBookmarkAfterRvSentLocked checks if a bookmark event
// with an RV >= the bookmarkAfterResourceVersion has been sent by this watcher
func (c *cacheWatcher) wasBookmarkAfterRvSentLocked() bool {
	return c.state == cacheWatcherBookmarkSent
}

// wasBookmarkAfterRvSent same as wasBookmarkAfterRvSentLocked just acquires a lock
func (c *cacheWatcher) wasBookmarkAfterRvSent() bool {
	c.stateMutex.Lock()
	defer c.stateMutex.Unlock()
	return c.wasBookmarkAfterRvSentLocked()
}

// markBookmarkAfterRvSent indicates that the given cacheWatcher
// have sent a bookmark event with an RV >= the bookmarkAfterResourceVersion
//
// this function relies on the fact that the nonblockingAdd method
// won't admit a bookmark event with an RV < the bookmarkAfterResourceVersion
// so the first received bookmark event is considered to match the bookmarkAfterResourceVersion
func (c *cacheWatcher) markBookmarkAfterRvSent(event *watchCacheEvent) {
	// note that bookmark events are not so common so will acquire a lock every ~60 second or so
	if event.Type == watch.Bookmark {
		c.stateMutex.Lock()
		defer c.stateMutex.Unlock()
		if !c.wasBookmarkAfterRvSentLocked() {
			c.state = cacheWatcherBookmarkSent
		}
	}
}

// setBookmarkAfterResourceVersion sets the bookmarkAfterResourceVersion and the state associated with it
func (c *cacheWatcher) setBookmarkAfterResourceVersion(bookmarkAfterResourceVersion uint64) {
	state := cacheWatcherWaitingForBookmark
	if bookmarkAfterResourceVersion == 0 {
		state = cacheWatcherBookmarkSent // if no specific RV was requested we assume no-op
	}
	c.state = state
	c.bookmarkAfterResourceVersion = bookmarkAfterResourceVersion
}

// setDrainInputBufferLocked if set to true indicates that we should delay closing this watcher
// until we send all events residing in the input buffer.
func (c *cacheWatcher) setDrainInputBufferLocked(drain bool) {
	c.drainInputBuffer = drain
}

// isDoneChannelClosed checks if c.done channel is closed
func (c *cacheWatcher) isDoneChannelClosedLocked() bool {
	select {
	case <-c.done:
		return true
	default:
	}
	return false
}

func getMutableObject(object runtime.Object) runtime.Object {
	if _, ok := object.(*cachingObject); ok {
		// It is safe to return without deep-copy, because the underlying
		// object will lazily perform deep-copy on the first try to change
		// any of its fields.
		return object
	}
	return object.DeepCopyObject()
}

func updateResourceVersion(object runtime.Object, versioner storage.Versioner, resourceVersion uint64) {
	if err := versioner.UpdateObject(object, resourceVersion); err != nil {
		utilruntime.HandleError(fmt.Errorf("failure to version api object (%d) %#v: %v", resourceVersion, object, err))
	}
}

func (c *cacheWatcher) convertToWatchEvent(event *watchCacheEvent) *watch.Event {
	if event.Type == watch.Bookmark {
		e := &watch.Event{Type: watch.Bookmark, Object: event.Object.DeepCopyObject()}
		if !c.wasBookmarkAfterRvSent() {
			if err := storage.AnnotateInitialEventsEndBookmark(e.Object); err != nil {
				utilruntime.HandleError(fmt.Errorf("error while accessing object's metadata gr: %v, identifier: %v, obj: %#v, err: %v", c.groupResource, c.identifier, e.Object, err))
				return nil
			}
		}
		return e
	}

	curObjPasses := event.Type != watch.Deleted && c.filter(event.Key, event.ObjLabels, event.ObjFields, event.Object)
	oldObjPasses := false
	if event.PrevObject != nil {
		oldObjPasses = c.filter(event.Key, event.PrevObjLabels, event.PrevObjFields, event.PrevObject)
	}
	if !curObjPasses && !oldObjPasses {
		// Watcher is not interested in that object.
		return nil
	}

	switch {
	case curObjPasses && !oldObjPasses:
		return &watch.Event{Type: watch.Added, Object: getMutableObject(event.Object)}
	case curObjPasses && oldObjPasses:
		return &watch.Event{Type: watch.Modified, Object: getMutableObject(event.Object)}
	case !curObjPasses && oldObjPasses:
		// return a delete event with the previous object content, but with the event's resource version
		oldObj := getMutableObject(event.PrevObject)
		// We know that if oldObj is cachingObject (which can only be set via
		// setCachingObjects), its resourceVersion is already set correctly and
		// we don't need to update it. However, since cachingObject efficiently
		// handles noop updates, we avoid this microoptimization here.
		updateResourceVersion(oldObj, c.versioner, event.ResourceVersion)
		return &watch.Event{Type: watch.Deleted, Object: oldObj}
	}

	return nil
}

// NOTE: sendWatchCacheEvent is assumed to not modify <event> !!!
func (c *cacheWatcher) sendWatchCacheEvent(event *watchCacheEvent) (builtAt, sentAt time.Time) {
	watchEvent := c.convertToWatchEvent(event)
	if watchEvent == nil {
		// Watcher is not interested in that object.
		return time.Time{}, time.Time{}
	}
	builtAt = c.clock.Now()

	// We need to ensure that if we put event X to the c.result, all
	// previous events were already put into it before, no matter whether
	// c.done is close or not.
	// Thus we cannot simply select from c.done and c.result and this
	// would give us non-determinism.
	// At the same time, we don't want to block infinitely on putting
	// to c.result, when c.done is already closed.
	//
	// This ensures that with c.done already close, we at most once go
	// into the next select after this. With that, no matter which
	// statement we choose there, we will deliver only consecutive
	// events.
	select {
	case <-c.done:
		return time.Time{}, time.Time{}
	default:
	}

	select {
	case c.result <- *watchEvent:
		c.markBookmarkAfterRvSent(event)
		sentAt = c.clock.Now()
	case <-c.done:
	}
	return builtAt, sentAt
}

// streamInterval sends every event of cacheInterval to the result channel,
// advancing *resourceVersion to the highest resourceVersion of any event the
// interval yielded (including events the watcher's filter drops). It returns
// the number of events the interval yielded. A non-nil error means the
// interval has been invalidated (the watch cache history moved past it) and
// can no longer serve events; the returned count is then meaningless.
//
// catchUpRound is true when the interval is a catch-up round from the watch
// cache history (as opposed to the initial interval), in which case the
// delivery latency of the events it serves is recorded like a live delivery
// and the round is abandoned promptly once the watcher is stopped.
func (c *cacheWatcher) streamInterval(cacheInterval *watchCacheInterval, resourceVersion *uint64, catchUpRound bool) (int, error) {
	eventCount := 0
	for {
		if catchUpRound {
			// Don't keep walking a (possibly long) history segment once the
			// watcher is stopped; the initial interval keeps its behavior.
			select {
			case <-c.done:
				return eventCount, nil
			default:
			}
		}
		event, err := cacheInterval.Next()
		if err != nil {
			return eventCount, err
		}
		if event == nil {
			return eventCount, nil
		}
		builtAt, sentAt := c.sendWatchCacheEvent(event)
		if catchUpRound && !sentAt.IsZero() {
			c.observeDispatchMetrics(event, builtAt, sentAt)
		}

		// With some events already sent, update resourceVersion so that
		// events that were buffered and not yet processed won't be delivered
		// to this watcher second time causing going back in time.
		//
		// There is one case where events are not necessary ordered by
		// resourceVersion, being a case of watching from resourceVersion=0,
		// which at the beginning returns the state of each objects.
		// For the purpose of it, we need to max it with the resource version
		// that we have so far.
		if event.ResourceVersion > *resourceVersion {
			*resourceVersion = event.ResourceVersion
		}
		eventCount++
	}
}

func (c *cacheWatcher) processInterval(ctx context.Context, cacheInterval *watchCacheInterval, resourceVersion uint64) {
	defer utilruntime.HandleCrashWithContext(ctx)
	defer close(c.result)
	defer c.Stop()

	// Check how long we are processing initEvents.
	// As long as these are not processed, we are not processing
	// any incoming events, so if it takes long, we may actually
	// block all watchers for some time.
	// TODO: From the logs it seems that there happens processing
	// times even up to 1s which is very long. However, this doesn't
	// depend that much on the number of initEvents. E.g. from the
	// 2000-node Kubemark run we have logs like this, e.g.:
	// ... processing 13862 initEvents took 66.808689ms
	// ... processing 14040 initEvents took 993.532539ms
	// We should understand what is blocking us in those cases (e.g.
	// is it lack of CPU, network, or sth else) and potentially
	// consider increase size of result buffer in those cases.
	const initProcessThreshold = 500 * time.Millisecond
	startTime := time.Now()

	// cacheInterval may be created from a version being more fresh than requested
	// (e.g. for NotOlderThan semantic). In such a case, we need to prevent watch event
	// with lower resourceVersion from being delivered to ensure watch contract.
	if cacheInterval.resourceVersion > resourceVersion {
		resourceVersion = cacheInterval.resourceVersion
	}

	initEventCount, err := c.streamInterval(cacheInterval, &resourceVersion, false)
	if err != nil {
		// The interval can no longer serve events: either the watch cache
		// history has moved past it (errCacheIntervalInvalidated) or reading
		// the initial state failed.
		if c.stall != nil && errors.Is(err, errCacheIntervalInvalidated) {
			// The watch cache history moved past the interval while the
			// client was still consuming it, i.e. the client fell more than
			// the history window behind: tell it honestly to re-list.
			c.terminate(err, resourceVersion, false)
			return
		}
		// Initially we considered sending an "out-of-history"
		// Error event in this case, but because historically
		// such events weren't sent out of the watchCache, we
		// decided not to. This is still ok, because on watch
		// closure, the watcher will try to re-instantiate the
		// watch and then will get an explicit "out-of-history"
		// window. There is potential for optimization, but for
		// now, in order to be on the safe side and not break
		// custom clients, the cost of it is something that we
		// are fully accepting.
		klog.Warningf("couldn't retrieve watch event to serve: %v", err)
		return
	}

	if initEventCount > 0 {
		metrics.InitCounter.WithLabelValues(c.groupResource.Group, c.groupResource.Resource).Add(float64(initEventCount))
	}
	processingTime := time.Since(startTime)
	if processingTime > initProcessThreshold {
		klog.V(2).Infof("processing %d initEvents of %s (%s) took %v", initEventCount, c.groupResource, c.identifier, processingTime)
	}

	// send bookmark after sending all events in cacheInterval for watchlist request
	if cacheInterval.initialEventsEndBookmark != nil {
		c.sendWatchCacheEvent(cacheInterval.initialEventsEndBookmark)
	}
	c.process(ctx, resourceVersion)
}

func (c *cacheWatcher) process(ctx context.Context, resourceVersion uint64) {
	// At this point we already start processing incoming watch events.
	// However, the init event can still be processed because their serialization
	// and sending to the client happens asynchrnously.
	// TODO: As describe in the KEP, we would like to estimate that by delaying
	//   the initialization signal proportionally to the number of events to
	//   process, but we're leaving this to the tuning phase.
	utilflowcontrol.WatchInitialized(ctx)

	if c.stall != nil {
		c.processStallResume(ctx, resourceVersion)
		return
	}

	for {
		select {
		case event, ok := <-c.input:
			if !ok {
				return
			}
			// only send events newer than resourceVersion
			// or a bookmark event with an RV equal to resourceVersion
			// if we haven't sent one to the client
			if event.ResourceVersion > resourceVersion || (event.Type == watch.Bookmark && event.ResourceVersion == resourceVersion && !c.wasBookmarkAfterRvSent()) {
				builtAt, sentAt := c.sendWatchCacheEvent(event)
				c.observeDispatchMetrics(event, builtAt, sentAt)
			}
		case <-ctx.Done():
			return
		}
	}
}

func (c *cacheWatcher) observeDispatchMetrics(event *watchCacheEvent, builtAt, sentAt time.Time) {
	if event.Type == watch.Bookmark || sentAt.IsZero() || event.RecordTime.IsZero() {
		return
	}
	if !event.CacheReceived.IsZero() {
		c.watcherMetrics.ObserveStage(metrics.StageStorageToCache, event.CacheReceived.Sub(event.RecordTime))
	}
	c.watcherMetrics.ObserveStage(metrics.StageCacheToWatcher, sentAt.Sub(builtAt))
	c.watcherMetrics.ObserveStage(metrics.StageTotal, sentAt.Sub(event.RecordTime))
}

// processStallResume is the stall/resume replacement for process's live
// loop: delivery is filtered by a resume position so events can arrive
// twice (once from a catch-up round, once from input) without being sent
// twice, and the stall latch wakes the watcher to catch up from the watch
// cache history when dispatch could not feed its input.
func (c *cacheWatcher) processStallResume(ctx context.Context, resourceVersion uint64) {
	c.stall.start = resourceVersion
	// reachedLive tells whether this watcher ever consumed the live stream
	// (an event from input or a completed catch-up round); it selects the
	// termination reason if the watcher later ages out of the history.
	reachedLive := false

	for {
		select {
		case event, ok := <-c.input:
			if !ok {
				return
			}
			// Peek the stall latch BEFORE delivering anything received
			// from input (bookmarks included).
			//
			// INVARIANT: every event that failed to enter this watcher's
			// input channel is served by a catch-up round before any
			// event at or above its resourceVersion is delivered (the
			// backlog below it that did enter input is delivered first,
			// by resume). This holds
			// because poke() runs synchronously on the dispatching
			// goroutine right after the failed nonblockingAdd (so a token is
			// pending before any later event lands in input), and because
			// the latch is a channel: even a poke that coalesces onto an
			// existing token leaves that token pending until we consume
			// it here, and our subsequent read of the history under its
			// lock covers everything appended before the failed dispatch.
			// Do not move poke() off the dispatching goroutine and do not
			// replace the channel with an atomic flag.
			if missed, pending := c.consumeLatch(); pending {
				leftover, ok := c.resume(missed, event, &resourceVersion, reachedLive)
				if !ok {
					return
				}
				reachedLive = true
				if leftover != nil {
					c.deliverLive(leftover, &resourceVersion)
				}
				continue
			}
			reachedLive = true
			c.deliverLive(event, &resourceVersion)
		case missed := <-c.stall.ch:
			// A failed enqueue while input was quiet wakes us up here.
			//
			// Do not delete this arm as redundant with the peek above: the
			// dispatcher's failed nonblockingAdd and its poke are not
			// atomic, and in that window this goroutine can drain input to
			// empty and park in this select. Without this arm a quiet
			// scope-filtered watcher would then serve the miss only when
			// (if ever) its next event arrives.
			leftover, ok := c.resume(missed, nil, &resourceVersion, reachedLive)
			if !ok {
				return
			}
			reachedLive = true
			if leftover != nil {
				c.deliverLive(leftover, &resourceVersion)
			}
		case <-ctx.Done():
			return
		}
	}
}

// deliverLive applies the live-stream delivery rules to one event received
// from input, filtering by and advancing the resume position.
func (c *cacheWatcher) deliverLive(event *watchCacheEvent, resourceVersion *uint64) {
	if event.Type == watch.Bookmark {
		// Deliver bookmarks beyond our position, and at it unless the
		// position is still the starting one and no bookmark for it is
		// awaited and unsent (as process does with the gate off: a watch
		// started at RV gets no bookmark at exactly RV, and the
		// initial-events-end bookmark sent by processInterval is not
		// repeated). A bookmark below the position was leapt over by a
		// catch-up round and would move the client's last seen
		// resourceVersion backwards.
		if event.ResourceVersion > *resourceVersion || (event.ResourceVersion == *resourceVersion && (*resourceVersion > c.stall.start || !c.wasBookmarkAfterRvSent())) {
			c.deliverEvent(event)
		}
		// A bookmark from input may advance the position: every object
		// event destined for this watcher at or below its RV either entered
		// input before it (channel FIFO) or failed to and latched a token
		// before the bookmark was enqueued, and the latch is always served
		// before an input event is delivered.
		if event.ResourceVersion > *resourceVersion {
			*resourceVersion = event.ResourceVersion
		}
		return
	}
	// Object events carry strictly increasing resourceVersions (the watch
	// cache history appends in etcd revision order), so a received event at
	// or below our position is one a catch-up round already served from the
	// history: drop it.
	if event.ResourceVersion > *resourceVersion {
		*resourceVersion = event.ResourceVersion
		c.deliverEvent(event)
	}
}

// resume serves a consumed stall token whose first missed event is at
// resourceVersion missed. Events below the miss that did fit into input were
// all enqueued before the poke (dispatch is sequential), so they are already
// in the channel: they are delivered first, in order (starting with inHand,
// the event the caller may already have received), so that raising the
// position to the miss cannot skip them. The position then becomes
// max(position, missed-1) and a catch-up round streams the history from
// there; a 410 therefore means the missed event itself has aged out of the
// history, never merely that a scope-filtered watcher's last delivery was
// long ago. It returns the first post-miss event taken from input, if any,
// for the caller to deliver after the round, and false if the watcher must
// exit.
func (c *cacheWatcher) resume(missed uint64, inHand *watchCacheEvent, resourceVersion *uint64, reachedLive bool) (*watchCacheEvent, bool) {
	pending := inHand
	for {
		if pending == nil {
			select {
			case event, ok := <-c.input:
				if !ok {
					return nil, false
				}
				pending = event
			default:
			}
			if pending == nil {
				break
			}
		}
		if pending.ResourceVersion >= missed {
			break
		}
		c.deliverLive(pending, resourceVersion)
		pending = nil
	}
	if missed > 0 && missed-1 > *resourceVersion {
		*resourceVersion = missed - 1
	}
	if !c.catchUp(resourceVersion, reachedLive) {
		return nil, false
	}
	return pending, true
}

// deliverEvent sends a live event to the client and records its delivery
// latency.
func (c *cacheWatcher) deliverEvent(event *watchCacheEvent) {
	builtAt, sentAt := c.sendWatchCacheEvent(event)
	if sentAt.IsZero() {
		return
	}
	c.observeDispatchMetrics(event, builtAt, sentAt)
}

// catchUp streams the watch cache history from *resourceVersion (the resume
// position established by resume) into the result channel, then returns to
// the live stream. It returns true only for a completed round (the caller
// then marks the watcher live). It returns false if the watcher must exit:
// the position has aged out of the history (a 410 was sent to the client,
// with reachedLive selecting the termination reason) or it was stopped
// meanwhile (the round is abandoned uncounted).
//
// The stall latch was consumed by the caller before this call, so every
// enqueue failure whose poke preceded that consume happened before this
// call reads the history and is therefore covered by the returned interval;
// a failure whose poke follows the consume latches a fresh token, which
// triggers the next round.
func (c *cacheWatcher) catchUp(resourceVersion *uint64, reachedLive bool) bool {
	interval, err := c.stall.src.intervalSince(*resourceVersion)
	if err != nil {
		c.terminate(err, *resourceVersion, reachedLive)
		return false
	}
	processed, err := c.streamInterval(interval, resourceVersion, true)
	if err != nil {
		// The history moved past our interval while we were streaming it:
		// the client is too slow to ever catch up.
		c.terminate(err, *resourceVersion, reachedLive)
		return false
	}
	select {
	case <-c.done:
		// Stopped mid-round: the round was abandoned, not completed, and
		// there is nothing left to do.
		return false
	default:
	}
	c.stall.metrics.CatchupRounds.Inc()
	c.stall.metrics.CatchupEvents.Observe(float64(processed))
	klog.V(4).InfoS("Watcher caught up from the watch cache history", "groupResource", c.groupResource, "identifier", c.identifier, "position", *resourceVersion, "events", processed)
	return true
}

// terminate ends this watch because its resume position is no longer covered
// by the watch cache history: the client gets one in-stream 410 (Expired)
// ERROR event, exactly like a compaction on a direct etcd watch, and the
// deferred close of the result channel in processInterval ends the stream.
func (c *cacheWatcher) terminate(err error, position uint64, reachedLive bool) {
	select {
	case <-c.done:
		// The watcher was already stopped by its client (Stop, deadline,
		// shutdown): its expiry is not a termination for cause and nobody is
		// reading the ERROR event; do not count or log it.
		return
	default:
	}
	reason := metrics.TerminationReasonResourceExpiredInitial
	counter := c.stall.metrics.TerminatedExpiredInitial
	if reachedLive {
		reason = metrics.TerminationReasonResourceExpired
		counter = c.stall.metrics.TerminatedExpired
	}
	counter.Inc()
	klog.V(2).InfoS("Terminating watcher: resume position no longer in the watch cache history",
		"groupResource", c.groupResource, "identifier", c.identifier, "position", position, "reason", reason, "err", err)

	// intervalSince normally yields apierrors.NewResourceExpired (a
	// StatusError); mid-stream interval invalidation (and any other plain
	// error) is reported as the equivalent 410 Status built here.
	var statusErr *apierrors.StatusError
	if !errors.As(err, &statusErr) {
		statusErr = apierrors.NewResourceExpired(fmt.Sprintf("too old resource version: %d", position))
	}
	// The 410 is the stream's final event: the caller returns into
	// processInterval's deferred close of the result channel.
	select {
	case c.result <- watch.Event{Type: watch.Error, Object: &statusErr.ErrStatus}:
	case <-c.done:
	}
}
