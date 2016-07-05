/*
Copyright 2015 The Kubernetes Authors.

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

package queue

import (
	"container/heap"
	"runtime"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/util/sets"
)

type qitem struct {
	value    interface{}
	priority Priority
	index    int
	readd    func(item *qitem) // re-add the value of the item to the queue
}

// A priorityQueue implements heap.Interface and holds qitems.
type priorityQueue []*qitem

func (pq priorityQueue) Len() int { return len(pq) }

func (pq priorityQueue) Less(i, j int) bool {
	return pq[i].priority.ts.Before(pq[j].priority.ts)
}

func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *priorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*qitem)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *priorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	item.index = -1 // for safety
	*pq = old[0 : n-1]
	return item
}

// concurrency-safe, deadline-oriented queue that returns items after their
// delay period has expired.
type DelayQueue struct {
	queue priorityQueue
	lock  sync.RWMutex
	cond  sync.Cond
}

func NewDelayQueue() *DelayQueue {
	q := &DelayQueue{}
	q.cond.L = &q.lock
	return q
}

func (q *DelayQueue) Add(d Delayed) {
	deadline := extractFromDelayed(d)

	q.lock.Lock()
	defer q.lock.Unlock()

	// readd using the original deadline computed from the original delay
	var readd func(*qitem)
	readd = func(qp *qitem) {
		q.lock.Lock()
		defer q.lock.Unlock()
		heap.Push(&q.queue, &qitem{
			value:    d,
			priority: deadline,
			readd:    readd,
		})
		q.cond.Broadcast()
	}
	heap.Push(&q.queue, &qitem{
		value:    d,
		priority: deadline,
		readd:    readd,
	})
	q.cond.Broadcast()
}

// If there's a deadline reported by d.Deadline() then `d` is added to the
// queue and this func returns true.
func (q *DelayQueue) Offer(d Deadlined) bool {
	deadline, ok := extractFromDeadlined(d)
	if ok {
		q.lock.Lock()
		defer q.lock.Unlock()
		heap.Push(&q.queue, &qitem{
			value:    d,
			priority: deadline,
			readd: func(qp *qitem) {
				q.Offer(qp.value.(Deadlined))
			},
		})
		q.cond.Broadcast()
	}
	return ok
}

// wait for the delay of the next item in the queue to expire, blocking if
// there are no items in the queue. does not guarantee first-come-first-serve
// ordering with respect to clients.
func (q *DelayQueue) Pop() interface{} {
	// doesn't implement cancellation, will always return a non-nil value
	return q.pop(func() *qitem {
		q.lock.Lock()
		defer q.lock.Unlock()
		for q.queue.Len() == 0 {
			q.cond.Wait()
		}
		x := heap.Pop(&q.queue)
		item := x.(*qitem)
		return item
	}, nil)
}

func finishWaiting(cond *sync.Cond, waitFinished <-chan struct{}) {
	runtime.Gosched()
	select {
	// avoid creating a timer if we can help it...
	case <-waitFinished:
		return
	default:
		const spinTimeout = 100 * time.Millisecond
		t := time.NewTimer(spinTimeout)
		defer t.Stop()
		for {
			runtime.Gosched()
			cond.Broadcast()
			select {
			case <-waitFinished:
				return
			case <-t.C:
				t.Reset(spinTimeout)
			}
		}
	}
}

// returns a non-nil value from the queue, or else nil if/when cancelled; if cancel
// is nil then cancellation is disabled and this func must return a non-nil value.
func (q *DelayQueue) pop(next func() *qitem, cancel <-chan struct{}) interface{} {
	var ch chan struct{}
	for {
		item := next()
		if item == nil {
			// cancelled
			return nil
		}
		x := item.value
		waitingPeriod := item.priority.ts.Sub(time.Now())
		if waitingPeriod >= 0 {
			// listen for calls to Add() while we're waiting for the deadline
			if ch == nil {
				ch = make(chan struct{}, 1)
			}
			go func() {
				q.lock.Lock()
				defer q.lock.Unlock()
				q.cond.Wait()
				ch <- struct{}{}
			}()
			select {
			case <-cancel:
				item.readd(item)
				finishWaiting(&q.cond, ch)
				return nil
			case <-ch:
				// we may no longer have the earliest deadline, re-try
				item.readd(item)
				continue
			case <-time.After(waitingPeriod):
				// noop
			case <-item.priority.notify:
				// noop
			}
		}
		return x
	}
}

// If multiple adds/updates of a single item happen while an item is in the
// queue before it has been processed, it will only be processed once, and
// when it is processed, the most recent version will be processed. Items are
// popped in order of their priority, currently controlled by a delay or
// deadline assigned to each item in the queue.
type DelayFIFO struct {
	// internal deadline-based priority queue
	delegate *DelayQueue
	// We depend on the property that items in the set are in the queue and vice versa.
	items          map[string]*qitem
	deadlinePolicy DeadlinePolicy
}

func (q *DelayFIFO) lock() {
	q.delegate.lock.Lock()
}

func (q *DelayFIFO) unlock() {
	q.delegate.lock.Unlock()
}

func (q *DelayFIFO) rlock() {
	q.delegate.lock.RLock()
}

func (q *DelayFIFO) runlock() {
	q.delegate.lock.RUnlock()
}

func (q *DelayFIFO) queue() *priorityQueue {
	return &q.delegate.queue
}

func (q *DelayFIFO) cond() *sync.Cond {
	return &q.delegate.cond
}

// Add inserts an item, and puts it in the queue. The item is only enqueued
// if it doesn't already exist in the set.
func (q *DelayFIFO) Add(d UniqueDelayed, rp ReplacementPolicy) {
	deadline := extractFromDelayed(d)
	id := d.GetUID()
	var adder func(*qitem)
	adder = func(*qitem) {
		q.add(id, deadline, d, KeepExisting, adder)
	}
	q.add(id, deadline, d, rp, adder)
}

func (q *DelayFIFO) Offer(d UniqueDeadlined, rp ReplacementPolicy) bool {
	if deadline, ok := extractFromDeadlined(d); ok {
		id := d.GetUID()
		q.add(id, deadline, d, rp, func(qp *qitem) { q.Offer(qp.value.(UniqueDeadlined), KeepExisting) })
		return true
	}
	return false
}

func (q *DelayFIFO) add(id string, deadline Priority, value interface{}, rp ReplacementPolicy, adder func(*qitem)) {
	q.lock()
	defer q.unlock()
	if item, exists := q.items[id]; !exists {
		item = &qitem{
			value:    value,
			priority: deadline,
			readd:    adder,
		}
		heap.Push(q.queue(), item)
		q.items[id] = item
	} else {
		// this is an update of an existing item
		item.value = rp.replacementValue(item.value, value)
		item.priority = q.deadlinePolicy.nextDeadline(item.priority, deadline)
		heap.Fix(q.queue(), item.index)
	}
	q.cond().Broadcast()
}

// Delete removes an item. It doesn't add it to the queue, because
// this implementation assumes the consumer only cares about the objects,
// not their priority order.
func (f *DelayFIFO) Delete(id string) {
	f.lock()
	defer f.unlock()
	delete(f.items, id)
}

// List returns a list of all the items.
func (f *DelayFIFO) List() []UniqueID {
	f.rlock()
	defer f.runlock()
	list := make([]UniqueID, 0, len(f.items))
	for _, item := range f.items {
		list = append(list, item.value.(UniqueDelayed))
	}
	return list
}

// ContainedIDs returns a stringset.StringSet containing all IDs of the stored items.
// This is a snapshot of a moment in time, and one should keep in mind that
// other go routines can add or remove items after you call this.
func (c *DelayFIFO) ContainedIDs() sets.String {
	c.rlock()
	defer c.runlock()
	set := sets.String{}
	for id := range c.items {
		set.Insert(id)
	}
	return set
}

// Get returns the requested item, or sets exists=false.
func (f *DelayFIFO) Get(id string) (UniqueID, bool) {
	f.rlock()
	defer f.runlock()
	if item, exists := f.items[id]; exists {
		return item.value.(UniqueID), true
	}
	return nil, false
}

// Variant of DelayQueue.Pop() for UniqueDelayed items
func (q *DelayFIFO) Await(timeout time.Duration) UniqueID {
	var (
		cancel = make(chan struct{})
		ch     = make(chan interface{}, 1)
		t      = time.NewTimer(timeout)
	)
	defer t.Stop()

	go func() { ch <- q.pop(cancel) }()

	var x interface{}
	select {
	case <-t.C:
		close(cancel)
		x = <-ch
	case x = <-ch:
		// noop
	}
	if x != nil {
		return x.(UniqueID)
	}
	return nil
}

// Pop blocks until either there is an item available to dequeue or else the specified
// cancel chan is closed. Callers that have no interest in providing a cancel chan
// should specify nil, or else WithoutCancel() (for readability).
func (q *DelayFIFO) Pop(cancel <-chan struct{}) UniqueID {
	x := q.pop(cancel)
	if x == nil {
		return nil
	}
	return x.(UniqueID)
}

// variant of DelayQueue.Pop that implements optional cancellation
func (q *DelayFIFO) pop(cancel <-chan struct{}) interface{} {
	next := func() *qitem {
		q.lock()
		defer q.unlock()
		for {
			for q.queue().Len() == 0 {
				signal := make(chan struct{})
				go func() {
					defer close(signal)
					q.cond().Wait()
				}()
				select {
				case <-cancel:
					// we may not have the lock yet, so
					// broadcast to abort Wait, then
					// return after lock re-acquisition
					finishWaiting(q.cond(), signal)
					return nil
				case <-signal:
					// we have the lock, re-check
					// the queue for data...
				}
			}
			x := heap.Pop(q.queue())
			item := x.(*qitem)
			unique := item.value.(UniqueID)
			uid := unique.GetUID()
			if _, ok := q.items[uid]; !ok {
				// item was deleted, keep looking
				continue
			}
			delete(q.items, uid)
			return item
		}
	}
	return q.delegate.pop(next, cancel)
}

func NewDelayFIFO() *DelayFIFO {
	f := &DelayFIFO{
		delegate: NewDelayQueue(),
		items:    map[string]*qitem{},
	}
	return f
}
