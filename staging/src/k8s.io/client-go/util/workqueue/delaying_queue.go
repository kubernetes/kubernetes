/*
Copyright 2016 The Kubernetes Authors.

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

package workqueue

import (
	"container/heap"
	"sync"
	"time"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/utils/clock"
)

// DelayingInterface is an Interface that can Add an item at a later time. This makes it easier to
// requeue items after failures without ending up in a hot-loop.
// A Delaying queue can be thought of two separate queues that work together - an 'active' queue of
// items which we want to process ASAP and a 'waiting' queue of items which must wait for a period of time
// to elapse before they are popped from the 'waiting' queue and added to the 'active' queue.
type DelayingInterface interface {
	Interface
	// AddAfter adds an item to the workqueue after the indicated duration has passed
	AddAfter(item interface{}, duration time.Duration)
}

// ExpandedDelayingInterface provides additional control over the behaviour of the delaying behaviour and allows
// us to keep the original interface intact for legacy clients.
type ExpandedDelayingInterface interface {
	DelayingInterface
	// AddWithOptions allows full control over the delaying queue features.
	AddWithOptions(item interface{}, opts ExpandedDelayingOptions)
	// DoneWaiting will remove an item from the 'waiting' queue.
	// If you want to cancel 'active' queued items you need to call Done().
	DoneWaiting(item interface{})
}

// ExpandedDelayingOptions toggle implementation specific options for use with ExpandedDelayingInterface
type ExpandedDelayingOptions struct {
	// Duration specifies for how long this item should be delayed before being added back into the 'active' queue.
	Duration time.Duration
	// Waiting specifies how you want the waitingLoop to treat items that are already present in the 'waiting' queue.
	WhenWaiting AlreadyWaitingBehaviour
	// PermitActiveAndWaiting overrides the default queue behaviour which ensures that an item is not allowed to be simultaneously
	// queued in both the 'active' and 'waiting' queues.
	// Setting PermitActiveAndWaiting to 'true' results in the following: -
	// 1. A item with a delay (Duration > 0) can be queued to 'waiting' when the same item is in the 'active' queue.
	// 2. A 'waiting' item will remain queued if the same item is subsequently queued to the 'active' queue with this option set.
	PermitActiveAndWaiting bool
	// Synchronous defines whether to wait for an item to be processed into the queue (or be dropped) before returning.
	Synchronous bool
}

// AlreadyWaitingBehaviour defines different choices of how to handle a delayed add when there is already the same item in the 'waiting' queue.
type AlreadyWaitingBehaviour int

const (
	// TakeShorter will modify the waiting item's wait time only if the incoming item has as shorter delay duration.
	TakeShorter AlreadyWaitingBehaviour = iota
	// TakeLonger will modify the waiting item's wait time only if the incoming item has a longer delay duration.
	TakeLonger
	// TakeIncoming will always adjust the delay of the waiting item with the delay of the incoming item unless they are equal.
	TakeIncoming
	// TakeExisting will never modify the waiting item's wait time, so effectively works as an "add only if the item is not already in the 'waiting' queue."
	TakeExisting
)

// DelayingQueueConfig specifies optional configurations to customize a DelayingInterface.
type DelayingQueueConfig struct {
	// Name for the queue. If unnamed, the metrics will not be registered.
	Name string

	// MetricsProvider optionally allows specifying a metrics provider to use for the queue
	// instead of the global provider.
	MetricsProvider MetricsProvider

	// Clock optionally allows injecting a real or fake clock for testing purposes.
	Clock clock.WithTicker

	// Queue optionally allows injecting custom queue Interface instead of the default one.
	Queue Interface
}

// NewDelayingQueue constructs a new workqueue with delayed queuing ability.
// NewDelayingQueue does not emit metrics. For use with a MetricsProvider, please use
// NewDelayingQueueWithConfig instead and specify a name.
func NewDelayingQueue() *delayingType {
	return NewDelayingQueueWithConfig(DelayingQueueConfig{})
}

// NewDelayingQueueWithConfig constructs a new workqueue with options to
// customize different properties.
func NewDelayingQueueWithConfig(config DelayingQueueConfig) *delayingType {
	if config.Clock == nil {
		config.Clock = clock.RealClock{}
	}

	if config.Queue == nil {
		config.Queue = NewWithConfig(QueueConfig{
			Name:            config.Name,
			MetricsProvider: config.MetricsProvider,
			Clock:           config.Clock,
		})
	}

	return newDelayingQueue(config.Clock, config.Queue, config.Name, config.MetricsProvider)
}

// NewDelayingQueueWithCustomQueue constructs a new workqueue with ability to
// inject custom queue Interface instead of the default one
// Deprecated: Use NewDelayingQueueWithConfig instead.
func NewDelayingQueueWithCustomQueue(q Interface, name string) DelayingInterface {
	return NewDelayingQueueWithConfig(DelayingQueueConfig{
		Name:  name,
		Queue: q,
	})
}

// NewNamedDelayingQueue constructs a new named workqueue with delayed queuing ability.
// Deprecated: Use NewDelayingQueueWithConfig instead.
func NewNamedDelayingQueue(name string) DelayingInterface {
	return NewDelayingQueueWithConfig(DelayingQueueConfig{Name: name})
}

// NewDelayingQueueWithCustomClock constructs a new named workqueue
// with ability to inject real or fake clock for testing purposes.
// Deprecated: Use NewDelayingQueueWithConfig instead.
func NewDelayingQueueWithCustomClock(clock clock.WithTicker, name string) *delayingType {
	return NewDelayingQueueWithConfig(DelayingQueueConfig{
		Name:  name,
		Clock: clock,
	})
}

func newDelayingQueue(clock clock.WithTicker, q Interface, name string, provider MetricsProvider) *delayingType {
	ret := &delayingType{
		Interface:          q,
		clock:              clock,
		nextReadyAtTimerCh: make(<-chan time.Time),
		heartbeat:          clock.NewTicker(maxWait),
		stopCh:             make(chan struct{}),
		waitingForAddCh:    make(chan *waitFor, 1000),
		metrics:            newRetryMetrics(name, provider),
	}

	go ret.waitingLoop()
	return ret
}

// delayingType wraps an Interface and provides delayed re-enqueuing
type delayingType struct {
	Interface

	// afterFill - a list of waitFor that require processing after the queue has filled and nextReadyAt
	// has been re-calculated.
	afterFill []*waitFor

	// clock tracks time for delayed firing
	clock clock.Clock

	// headReadyTime tracks the next time the head of the 'waiting' queue should become active.
	headReadyTime time.Time

	// heartbeat ensures we wait no more than maxWait before firing
	heartbeat clock.Ticker

	// metrics counts the number of retries
	metrics retryMetrics

	// nextReadyAtTimer is a timer used to create nextReadyAt signals when an item matures ready
	// for queuing in the 'active' queue.
	nextReadyAtTimer clock.Timer

	// nextReadyAtTimerCh is the channel to return the signal from the timer.const
	// We set it explicitly so that we can set it to 'never' by detaching it from the timer.
	nextReadyAtTimerCh <-chan time.Time

	// stopCh lets us signal a shutdown to the waitingLoop
	stopCh chan struct{}
	// stopOnce guarantees we only signal shutdown a single time
	stopOnce sync.Once

	// waitingForAddCh is a buffered channel that feeds waitingForAdd
	waitingForAddCh chan *waitFor
}

var _ DelayingInterface = &delayingType{}
var _ ExpandedDelayingInterface = &delayingType{}
var _ Interface = &delayingType{}

// waitForAction defines what we are asking the queue to do (which is usually queue an item but can be other things)
type waitForAction int

const (
	// waitForActionQueue - queue the item in the delaying queue
	waitForActionQueue waitForAction = iota
	// waitForActionDoneWaiting - forget the item provided if 'waiting'
	waitForActionDoneWaiting
	// waitForActionIsWaiting - report back on an item existing in the 'waiting' queue.
	waitForActionIsWaiting
	// waitForActionLenWaiting - report back on the waiting queue length
	waitForActionLenWaiting
	// waitForActionNextReady - report back on the next item to be ready.
	waitForActionNextReady
	// waitForActionSyncFill - close return channel when the 'waiting' queue is filled
	// and the nextReadyAt timer has been updated.
	waitForActionSyncFill
)

// waitFor holds the data to add and the time it should be added
type waitFor struct {
	action  waitForAction
	data    t
	readyAt time.Time
	// index in the priority queue (heap)
	index int
	// options defines how the caller would like the item to behave.
	options ExpandedDelayingOptions
	// returnCh allows the caller to block pending the processing and to also return
	// a response value which can represent a stat such as existence or 'waiting' queue length.
	returnCh chan waitingLoopResponse
}

// waitForPriorityQueue implements a priority queue for waitFor items.
//
// waitForPriorityQueue implements heap.Interface. The item occurring next in
// time (i.e., the item with the smallest readyAt) is at the root (index 0).
// Peek returns this minimum item at index 0. Pop returns the minimum item after
// it has been removed from the queue and placed at index Len()-1 by
// container/heap. Push adds an item at index Len(), and container/heap
// percolates it into the correct location.
type waitForPriorityQueue []*waitFor

func (pq waitForPriorityQueue) Len() int {
	return len(pq)
}
func (pq waitForPriorityQueue) Less(i, j int) bool {
	return pq[i].readyAt.Before(pq[j].readyAt)
}
func (pq waitForPriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

// Push adds an item to the queue. Push should not be called directly; instead,
// use `heap.Push`.
func (pq *waitForPriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*waitFor)
	item.index = n
	*pq = append(*pq, item)
}

// Pop removes an item from the queue. Pop should not be called directly;
// instead, use `heap.Pop`.
func (pq *waitForPriorityQueue) Pop() interface{} {
	n := len(*pq)
	item := (*pq)[n-1]
	item.index = -1
	*pq = (*pq)[0:(n - 1)]
	return item
}

// Peek returns the item at the beginning of the queue, without removing the
// item or otherwise mutating the queue. It is safe to call directly.
func (pq waitForPriorityQueue) Peek() interface{} {
	return pq[0]
}

// ShutDown stops the queue. After the queue drains, the returned shutdown bool
// on Get() will be true. This method may be invoked more than once.
func (q *delayingType) ShutDown() {
	q.stopOnce.Do(func() {
		q.Interface.ShutDown()
		close(q.stopCh)
		q.heartbeat.Stop()
	})
}

// AddAfter adds the given item to the work queue after the given delay
func (q *delayingType) AddAfter(item interface{}, duration time.Duration) {
	// don't add if we're already shutting down
	if q.ShuttingDown() {
		return
	}

	q.metrics.retry()

	// immediately add things with no delay
	if duration <= 0 {
		q.Add(item)
		return
	}

	select {
	case <-q.stopCh:
		// unblock if ShutDown() is called
	case q.waitingForAddCh <- &waitFor{data: item, readyAt: q.clock.Now().Add(duration)}:
	}
}

// AddWithOptions gives callers more control over the queueing behaviour.
// Calls are asynchronous unless option Synchronous is set.
func (q *delayingType) AddWithOptions(item interface{}, opts ExpandedDelayingOptions) {
	q.sendItemToWaitingLoop(item, opts, waitForActionQueue)
}

// ForgetWaiting removes an item from the delayed queue, effectively cancelling its future processing.
func (q *delayingType) DoneWaiting(item interface{}) {
	q.sendItemToWaitingLoop(item, ExpandedDelayingOptions{}, waitForActionDoneWaiting)
}

// isWaiting returns a bool indicating whether the given item is queued in the 'waiting' queue.
// The client can call IsQueued() in order to determine whether an item is queued in the 'active' queue.
func (q *delayingType) isWaiting(item interface{}) (bool, time.Duration) {
	result := q.sendItemToWaitingLoop(item, ExpandedDelayingOptions{Synchronous: true}, waitForActionIsWaiting)
	if result.exists != nil && result.nextReady != nil {
		return *result.exists, *result.nextReady
	}
	return false, 0
}

// lenWaiting returns an int representing the number of items queued in the 'waiting' queue.
// The client can call Len() in order to determine the number of items in the 'active' queue.
func (q *delayingType) lenWaiting() int {
	result := q.sendItemToWaitingLoop(nil, ExpandedDelayingOptions{Synchronous: true}, waitForActionLenWaiting)
	if result.length != nil {
		return *result.length
	}
	return 0
}

// nextReady returns the item at the head of the 'waiting' queue and how long it has left to wait.
func (q *delayingType) nextReady() (interface{}, time.Duration) {
	result := q.sendItemToWaitingLoop(nil, ExpandedDelayingOptions{Synchronous: true}, waitForActionNextReady)
	if result.nextReady != nil {
		return result.item, *result.nextReady
	}
	return nil, 0
}

// syncFill blocks until the delaying queue has drained all items from its feeder channel, processed
// them into their appropriate queue and updated the nextReadyAt time as necessary.
func (q *delayingType) syncFill() {
	q.sendItemToWaitingLoop(nil, ExpandedDelayingOptions{Synchronous: true}, waitForActionSyncFill)
}

// sendItemToWaitingLoop facilitates the different interface methods and performs the actual add, sending,
// forgetting an item, by sending an appropriate message through to the waitingLoop.
func (q *delayingType) sendItemToWaitingLoop(item interface{}, opts ExpandedDelayingOptions, action waitForAction) waitingLoopResponse {
	var result waitingLoopResponse

	// don't add if we're already shutting down
	if q.Interface.ShuttingDown() {
		return result
	}

	// Add the item into the delaying queue (either the 'active' or 'waiting' queue)
	waitEntry := &waitFor{
		action:  action,
		data:    item,
		readyAt: q.clock.Now().Add(opts.Duration),
		options: opts,
	}

	// For synchronous calls we will create a channel that we can wait to be released (closed).
	var cb chan waitingLoopResponse
	if opts.Synchronous {
		cb = make(chan waitingLoopResponse)
		waitEntry.returnCh = cb
	}

	// Send the entry to the waitingLoop...
	select {
	case <-q.stopCh:
		// unblock if ShutDown() is called
	case q.waitingForAddCh <- waitEntry:
		// wait for call-back if we asked for one...
		if opts.Synchronous {
			select {
			case <-q.stopCh:
			case result = <-cb:
			}
		}
	}
	return result
}

// waitingLoopResponse describes the message that is returned from the waiting loop when a response
// is requested/required.  It contains the different answer types that may be in the response.
type waitingLoopResponse struct {
	item      interface{}
	exists    *bool
	length    *int
	nextReady *time.Duration
}

// maxWait keeps a max bound on the wait time. It's just insurance against weird things happening.
// Checking the queue every 10 seconds isn't expensive and we know that we'll never end up with an
// expired item sitting for more than 10 seconds.
const maxWait = 10 * time.Second

// waitingLoop runs until the workqueue is shutdown and keeps a check on the list of items to be added.
func (q *delayingType) waitingLoop() {
	defer utilruntime.HandleCrash()

	waitingForQueue := &waitForPriorityQueue{}
	heap.Init(waitingForQueue)

	waitingEntryByData := map[t]*waitFor{}

	for {
		if q.Interface.ShuttingDown() {
			return
		}

		// Add ready entries
		for waitingForQueue.Len() > 0 {
			entry := waitingForQueue.Peek().(*waitFor)
			if entry.readyAt.After(q.clock.Now()) {
				break
			}
			entry = heap.Pop(waitingForQueue).(*waitFor)
			q.Interface.Add(entry.data)
			delete(waitingEntryByData, entry.data)
		}

		// Set up a wait for the first item's readyAt (if one exists)
		q.calculateNextReadyAt(waitingForQueue)
		q.handleReportingAndSync(waitingForQueue, waitingEntryByData)

		select {
		case <-q.stopCh:
			return

		case <-q.heartbeat.C():
			// continue the loop, which will add ready items

		case <-q.nextReadyAtTimerCh:
			// continue the loop, which will add ready items

		case waitEntry := <-q.waitingForAddCh:
			q.processArrivingWaitForEntry(waitingForQueue, waitingEntryByData, waitEntry)

			drained := false
			for !drained {
				select {
				case waitEntry := <-q.waitingForAddCh:
					q.processArrivingWaitForEntry(waitingForQueue, waitingEntryByData, waitEntry)
				default:
					drained = true
				}
			}
		}
	}
}

// insert adds the entry to the 'waiting' queue, or updates the readyAt if it already exists in the queue
// It returns a boolean to indicate that the entry was inserted or not.
func insert(q *waitForPriorityQueue, knownEntries map[t]*waitFor, entry *waitFor) {
	// if the entry already exists, update the time according to the entry.options.WhenWaiting
	existing, exists := knownEntries[entry.data]
	if exists {
		var replace bool
		switch entry.options.WhenWaiting {
		case TakeShorter:
			replace = existing.readyAt.After(entry.readyAt)
		case TakeLonger:
			replace = existing.readyAt.Before(entry.readyAt)
		case TakeIncoming:
			replace = !existing.readyAt.Equal(entry.readyAt)
		}
		if replace {
			existing.readyAt = entry.readyAt
			existing.options = entry.options
			heap.Fix(q, existing.index)
		}
		return
	}

	heap.Push(q, entry)
	knownEntries[entry.data] = entry
}

// calculateNextReadyAt recalculates the nextReadyAtTime as follows:
// - removes the timer when the 'waiting' queue is empty.
// - resets the timer when the head 'waiting' item has changed.
// - creates a new timer when 1+ items are queued to an empty 'waiting' queue.
func (q *delayingType) calculateNextReadyAt(w *waitForPriorityQueue) {
	var head *waitFor

	// early escape when the timer has not changed
	if w.Len() > 0 {
		head = w.Peek().(*waitFor)
		if q.headReadyTime.Equal(head.readyAt) {
			return
		}
	}

	// reset timer
	if q.nextReadyAtTimer != nil {
		q.nextReadyAtTimerCh = make(<-chan time.Time)
		q.nextReadyAtTimer.Stop()
		q.nextReadyAtTimer = nil
		q.headReadyTime = time.Time{}
	}

	// create new timer
	if head != nil {
		now := q.clock.Now()
		q.nextReadyAtTimer = q.clock.NewTimer(head.readyAt.Sub(now))
		q.nextReadyAtTimerCh = q.nextReadyAtTimer.C()
		q.headReadyTime = head.readyAt
	}
}

// handleReportingAndSync loops through the inspection requests and syncs and loops through providing the answers
// and closing the return channel.
func (q *delayingType) handleReportingAndSync(w *waitForPriorityQueue, knownEntries map[t]*waitFor) {
	if len(q.afterFill) == 0 {
		return
	}

	for _, req := range q.afterFill {
		r := waitingLoopResponse{}
		switch req.action {
		case waitForActionIsWaiting:
			_, exists := knownEntries[req.data]
			r.exists = &exists
			nr := req.readyAt.Sub(q.clock.Now())
			r.nextReady = &nr
		case waitForActionLenWaiting:
			l := w.Len()
			r.length = &l
		case waitForActionNextReady:
			if w.Len() > 0 {
				r.item = w.Peek().(*waitFor).data
				nr := q.headReadyTime.Sub(q.clock.Now())
				r.nextReady = &nr
			}
		case waitForActionSyncFill:
		}
		req.returnCh <- r
		close(req.returnCh)
	}
	q.afterFill = nil
}

// processArrivingWaitForEntry receives messages received from clients a takes the appropriate queuing based
// upon the entry.action requested.
// When performing the waitForActionQueue action it behaves as follows:
//   - An entry is dropped if it is already actively queued (much like Add() on the Interface)
//   - Delayed items are inserted into the 'waiting' queue according to their Waiting option (TakeShorter by default)
//   - Immediate items will be dropped when the TakeLonger Waiting option is set and they are already present in 'waiting'.
//   - Immediate items will pre-empt/remove items in the 'waiting' queue unless the PermitActiveAndWaiting option is set.
func (q *delayingType) processArrivingWaitForEntry(w *waitForPriorityQueue, knownEntries map[t]*waitFor, entry *waitFor) {
	// handle the different message actions
	switch entry.action {
	case waitForActionIsWaiting, waitForActionLenWaiting, waitForActionNextReady, waitForActionSyncFill:
		// all of these actions are processed in the call handleReportingAndSync()
		if entry.returnCh == nil {
			return
		}
		q.afterFill = append(q.afterFill, entry)
		return
	case waitForActionDoneWaiting:
		// remove entry from the 'waiting' queue
		existing, exists := knownEntries[entry.data]
		if exists {
			// remove the item from the 'waiting' queue
			heap.Remove(w, existing.index)
			delete(knownEntries, entry.data)
		}
		return
	case waitForActionQueue:
		// add entry to either 'active' or 'waiting' queues (dependent on delay and settings)
		if entry.returnCh != nil {
			defer close(entry.returnCh)
		}

		// Perform de-duplication between the 'waiting' queue against items in the 'active' queue unless
		// disabled by PermitActiveAndWaiting.
		if !entry.options.PermitActiveAndWaiting && q.Interface.IsQueued(entry.data) {
			return
		}

		// drop if the incoming item if the caller wishes to keep the existing item if it exists.
		existing, exists := knownEntries[entry.data]
		if exists && entry.options.WhenWaiting == TakeExisting {
			return
		}

		// delayed add (to 'waiting' queue)
		if entry.readyAt.After(q.clock.Now()) {
			// inform the metrics of the retry (moved from AddAfter())
			q.metrics.retry()

			// insert the entry into the 'waiting' queue with appropriate insertion strategy defined by options.Waiting
			insert(w, knownEntries, entry)
			return
		}

		// immediate add (to 'active' queue)
		if exists {
			// items in the 'waiting' queue must be longer than an immediate add, if we TakeLonger we must drop this add.
			if entry.options.WhenWaiting == TakeLonger {
				return
			}
			// This covers both the TakeShorter and TakeIncoming cases because this is incoming and must be the shortest
			// right now too.  We will remove/pre-empt the item in the 'waiting' queue unless disabled.
			if !entry.options.PermitActiveAndWaiting {
				heap.Remove(w, existing.index)
				delete(knownEntries, entry.data)
			}
		}

		// Perform the immediate add to the 'active' queue.
		q.Interface.Add(entry.data)
	}
}
