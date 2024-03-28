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
type DelayingInterface interface {
	Interface
	// AddAfter adds an item to the workqueue after the indicated duration has passed.
	// If the item is already scheduled for an earlier add then nothing more is scheduled.
	// If the item is already scheduled for a later add then the scheduled add is moved up
	// to the requested time.
	// If the item is already in the FIFO then no addition is scheduled.
	AddAfter(item interface{}, duration time.Duration)
}

type DelayingPeekableQueue interface {
	PeekableQueue
	DelayingInterface

	// Has reveals whether the given item is either present in either the FIFO
	// or waiting for a specific time to be added to the FIFO
	Has(item interface{}) bool
}

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
	Queue PeekableQueue
}

// NewDelayingQueue constructs a new workqueue with delayed queuing ability.
// NewDelayingQueue does not emit metrics. For use with a MetricsProvider, please use
// NewDelayingQueueWithConfig instead and specify a name.
func NewDelayingQueue() DelayingPeekableQueue {
	return NewDelayingQueueWithConfig(DelayingQueueConfig{})
}

// NewDelayingQueueWithConfig constructs a new workqueue with options to
// customize different properties.
func NewDelayingQueueWithConfig(config DelayingQueueConfig) DelayingPeekableQueue {
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
func NewDelayingQueueWithCustomQueue(q PeekableQueue, name string) DelayingPeekableQueue {
	return NewDelayingQueueWithConfig(DelayingQueueConfig{
		Name:  name,
		Queue: q,
	})
}

// NewNamedDelayingQueue constructs a new named workqueue with delayed queuing ability.
// Deprecated: Use NewDelayingQueueWithConfig instead.
func NewNamedDelayingQueue(name string) DelayingPeekableQueue {
	return NewDelayingQueueWithConfig(DelayingQueueConfig{Name: name})
}

// NewDelayingQueueWithCustomClock constructs a new named workqueue
// with ability to inject real or fake clock for testing purposes.
// Deprecated: Use NewDelayingQueueWithConfig instead.
func NewDelayingQueueWithCustomClock(clock clock.WithTicker, name string) DelayingPeekableQueue {
	return NewDelayingQueueWithConfig(DelayingQueueConfig{
		Name:  name,
		Clock: clock,
	})
}

func newDelayingQueue(clock clock.WithTicker, q PeekableQueue, name string, provider MetricsProvider) *delayingType {
	ret := createDelayingQueue(clock, q, name, provider)
	go ret.waitingLoop()
	return ret
}

func createDelayingQueue(clock clock.WithTicker, q PeekableQueue, name string, provider MetricsProvider) *delayingType {
	ret := &delayingType{
		PeekableQueue:            q,
		clock:                    clock,
		heartbeat:                clock.NewTicker(maxWait),
		stopCh:                   make(chan struct{}),
		waitingForAddCh:          make(chan *waitFor, 1000),
		waitingEntryByDataLocked: map[t]*waitFor{},
		metrics:                  newRetryMetrics(name, provider),
	}
	return ret
}

// delayingType wraps an Interface and provides delayed re-enquing
type delayingType struct {
	PeekableQueue

	// clock tracks time for delayed firing
	clock clock.Clock

	// stopCh lets us signal a shutdown to the waiting loop
	stopCh chan struct{}
	// stopOnce guarantees we only signal shutdown a single time
	stopOnce sync.Once

	// heartbeat ensures we wait no more than maxWait before firing
	heartbeat clock.Ticker

	// waitingForAddCh is a buffered channel that feeds waitingForQueue
	waitingForAddCh chan *waitFor

	// waitingEntryByDataLocked is an index into waitingForQueue.
	// Access only while holding the PeekableQueue's lock.
	waitingEntryByDataLocked map[t]*waitFor

	// metrics counts the number of retries
	metrics retryMetrics
}

// waitFor holds the data to add and the time it should be added
type waitFor struct {
	data    t
	readyAt time.Time
	// index in the priority queue (heap)
	index int

	// deletedLocked indicates that this should NOT actually be added.
	// Access only while holding the PeekableQueue's lock.
	deletedLocked bool
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
		q.PeekableQueue.ShutDown()
		close(q.stopCh)
		q.heartbeat.Stop()
	})
}

func (q *delayingType) Has(item interface{}) bool {
	var had bool
	q.PeekableQueue.ConditionalAdd(item, func(shuttingDown, queueHas bool) bool {
		had = queueHas
		if !had {
			_, had = q.waitingEntryByDataLocked[item]

		}
		return false
	})
	return had
}

func (q *delayingType) Add(item interface{}) {
	q.ConditionalAdd(item, func(bool, bool) bool { return true })
}

func (q *delayingType) ConditionalAdd(item interface{}, wantAddLocked func(shuttingDown, has bool) bool) {
	q.PeekableQueue.ConditionalAdd(item, func(shuttingDown, queueHas bool) bool {
		addToQueue := wantAddLocked(shuttingDown, queueHas)
		if addToQueue {
			waitEntry, isWaiting := q.waitingEntryByDataLocked[item]
			if isWaiting {
				waitEntry.deletedLocked = true
			}
		}
		return addToQueue
	})
}

// AddAfter adds the given item to the work queue after the given delay
func (q *delayingType) AddAfter(item interface{}, duration time.Duration) {
	var wasShuttingDown, had, added bool
	q.PeekableQueue.ConditionalAdd(item, func(shuttingDown, has bool) bool {
		wasShuttingDown = shuttingDown
		had = has
		added = (!has) && (!shuttingDown) && duration <= 0
		return added
	})
	// don't add if we're already shutting down
	if wasShuttingDown {
		return
	}

	q.metrics.retry()

	// immediately add things with no delay
	if had || added {
		return
	}

	select {
	case <-q.stopCh:
		// unblock if ShutDown() is called
	case q.waitingForAddCh <- &waitFor{data: item, readyAt: q.clock.Now().Add(duration)}:
	}
}

// maxWait keeps a max bound on the wait time. It's just insurance against weird things happening.
// Checking the queue every 10 seconds isn't expensive and we know that we'll never end up with an
// expired item sitting for more than 10 seconds.
const maxWait = 10 * time.Second

// waitingLoop runs until the workqueue is shutdown and keeps a check on the list of items to be added.
func (q *delayingType) waitingLoop() {
	defer utilruntime.HandleCrash()

	// Make a placeholder channel to use when there are no items in our list
	never := make(<-chan time.Time)

	// Make a timer that expires when the item at the head of the waiting queue is ready
	var nextReadyAtTimer clock.Timer

	waitingForQueue := &waitForPriorityQueue{}
	heap.Init(waitingForQueue)

	addWaitFor := func(waitEntry *waitFor) {
		q.PeekableQueue.ConditionalAdd(waitEntry.data, func(shuttingDown, has bool) bool {
			if shuttingDown || has {
				return false
			}
			if waitEntry.readyAt.After(q.clock.Now()) {
				insert(waitingForQueue, q.waitingEntryByDataLocked, waitEntry)
				return false
			}
			return true
		})
	}
	for {
		if q.PeekableQueue.ShuttingDown() {
			return
		}

		now := q.clock.Now()

		// Add ready entries
		for waitingForQueue.Len() > 0 {
			entry := waitingForQueue.Peek().(*waitFor)
			if entry.readyAt.After(now) {
				break
			}

			entry = heap.Pop(waitingForQueue).(*waitFor)
			q.PeekableQueue.ConditionalAdd(entry.data, func(shuttingDown bool, has bool) bool {
				delete(q.waitingEntryByDataLocked, entry.data)
				return !entry.deletedLocked
			})
		}

		// Set up a wait for the first item's readyAt (if one exists)
		nextReadyAt := never
		if waitingForQueue.Len() > 0 {
			if nextReadyAtTimer != nil {
				nextReadyAtTimer.Stop()
			}
			entry := waitingForQueue.Peek().(*waitFor)
			nextReadyAtTimer = q.clock.NewTimer(entry.readyAt.Sub(now))
			nextReadyAt = nextReadyAtTimer.C()
		}

		select {
		case <-q.stopCh:
			return

		case <-q.heartbeat.C():
			// continue the loop, which will add ready items

		case <-nextReadyAt:
			// continue the loop, which will add ready items

		case waitEntry := <-q.waitingForAddCh:
			addWaitFor(waitEntry)

			drained := false
			for !drained {
				select {
				case waitEntry := <-q.waitingForAddCh:
					addWaitFor(waitEntry)
				default:
					drained = true
				}
			}
		}
	}
}

// insert adds the entry to the priority queue, or updates the readyAt if it already exists in the queue.
// Call this ONLY from the goroutine running q.waitingLoop.
func insert(q *waitForPriorityQueue, knownEntries map[t]*waitFor, entry *waitFor) {
	// if the entry already exists, update the time only if it would cause the item to be queued sooner
	existing, exists := knownEntries[entry.data]
	if exists {
		if existing.readyAt.After(entry.readyAt) {
			existing.readyAt = entry.readyAt
			heap.Fix(q, existing.index)
		}

		return
	}

	heap.Push(q, entry)
	knownEntries[entry.data] = entry
}
