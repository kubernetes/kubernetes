/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/util/sets"
)

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
	cancel := make(chan struct{})
	ch := make(chan interface{}, 1)
	go func() { ch <- q.pop(cancel) }()
	var x interface{}
	select {
	case <-time.After(timeout):
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

// Variant of DelayQueue.Pop() for UniqueDelayed items
func (q *DelayFIFO) Pop() UniqueID {
	return q.pop(nil).(UniqueID)
}

// variant of DelayQueue.Pop that implements optional cancellation
func (q *DelayFIFO) pop(cancel chan struct{}) interface{} {
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
					q.cond().Broadcast()
					<-signal
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
