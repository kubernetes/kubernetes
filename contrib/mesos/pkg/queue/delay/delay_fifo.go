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

package delay

import (
	"container/heap"
	"fmt"
	"reflect"
	"time"

	"k8s.io/kubernetes/contrib/mesos/pkg/queue"
	"k8s.io/kubernetes/contrib/mesos/pkg/queue/priority"
	"k8s.io/kubernetes/pkg/util/sets"
)

// If multiple adds/updates of a single item happen while an item is in the
// queue before it has been processed, it will only be processed once, and
// when it is processed, the most recent version will be processed. Items are
// popped in order of their priority, currently controlled by a delay or
// deadline assigned to each item in the queue.
type FIFOQueue struct {
	*Queue
	// We depend on the property that items in the set are in the queue and vice versa.
	items          map[string]priority.Item
	deadlinePolicy DeadlinePolicy
}

func NewFIFOQueue() *FIFOQueue {
	return &FIFOQueue{
		Queue: NewQueue(),
		items: map[string]priority.Item{},
	}
}

// Add inserts an item, and puts it in the queue. The item is only enqueued
// if it doesn't already exist in the set.
func (q *FIFOQueue) Add(d UniqueDelayed, rp ReplacementPolicy) {
	deadline := extractFromDelayed(d)
	id := d.GetUID()
	q.push(id, &fifoAddItem{
		Item: priority.NewItem(d, deadline),
		id:   id,
	}, rp)
}

// fifoAddItem adds locking and broadcasting to priority.Item.Push
type fifoAddItem struct {
	priority.Item
	id string
}

func (di *fifoAddItem) Push(queue heap.Interface) {
	dq := queue.(*FIFOQueue)
	dq.push(di.id, di, KeepExisting)
}

func (q *FIFOQueue) Offer(d UniqueDeadlined, rp ReplacementPolicy) bool {
	if deadline, ok := extractFromDeadlined(d); ok {
		id := d.GetUID()
		q.push(id, &fifoOfferItem{
			Item: priority.NewItem(d, deadline),
		}, rp)
		return true
	}
	return false
}

func (q *FIFOQueue) push(id string, newItem priority.Item, rp ReplacementPolicy) {
	q.lock.Lock()
	defer q.lock.Unlock()
	item, exists := q.items[id]
	if !exists {
		item = newItem
		heap.Push(q.Queue, item)
		q.items[id] = item
	} else {
		// replace existing item
		newValue := rp.replacementValue(item.Value(), newItem.Value())
		oldPriority := item.Priority().(DelayPriority)
		newPriority := newItem.Priority().(DelayPriority)
		newPriority = q.deadlinePolicy.nextDeadline(oldPriority, newPriority)
		item = priority.NewItem(newValue, newPriority)

		switch i := newItem.(type) {
		case *fifoAddItem:
			item = &fifoAddItem{Item: item, id: i.id}
		case *fifoOfferItem:
			item = &fifoOfferItem{Item: item}
		default:
			panic(fmt.Sprintf("unsupported newItem type: %v", reflect.TypeOf(i)))
		}

		heap.Fix(q.Queue, item.Index())
		q.items[id] = item
	}
	q.cond.Broadcast()
}

// fifoOfferItem offers the value priority.Item.Push
type fifoOfferItem struct {
	priority.Item
}

func (i *fifoOfferItem) Push(queue heap.Interface) {
	dq := queue.(*FIFOQueue)
	dq.Offer(i.Value().(UniqueDeadlined), KeepExisting)
}

// Delete removes an item. It doesn't add it to the queue, because
// this implementation assumes the consumer only cares about the objects,
// not their priority order.
func (q *FIFOQueue) Delete(id string) {
	q.lock.Lock()
	defer q.lock.Unlock()
	delete(q.items, id)
}

// List returns a list of all the items.
func (q *FIFOQueue) List() []queue.UniqueID {
	q.lock.RLock()
	defer q.lock.RUnlock()
	list := make([]queue.UniqueID, 0, len(q.items))
	for _, item := range q.items {
		list = append(list, item.Value().(UniqueDelayed))
	}
	return list
}

// ContainedIDs returns a stringset.StringSet containing all IDs of the stored items.
// This is a snapshot of a moment in time, and one should keep in mind that
// other go routines can add or remove items after you call this.
func (q *FIFOQueue) ContainedIDs() sets.String {
	q.lock.RLock()
	defer q.lock.RUnlock()
	set := sets.String{}
	for id := range q.items {
		set.Insert(id)
	}
	return set
}

// Get returns the requested item, or sets exists=false.
func (q *FIFOQueue) Get(id string) (queue.UniqueID, bool) {
	q.lock.RLock()
	defer q.lock.RUnlock()
	if item, exists := q.items[id]; exists {
		return item.Value().(queue.UniqueID), true
	}
	return nil, false
}

// Variant of DelayQueue.Pop() for UniqueDelayed items
func (q *FIFOQueue) Await(timeout time.Duration) queue.UniqueID {
	cancel := make(chan struct{})
	ch := make(chan interface{}, 1)
	go func() { ch <- q.pop(cancel) }()
	var x interface{}
	timer := time.NewTimer(timeout)
	select {
	case <-timer.C:
		close(cancel)
		x = <-ch
	case x = <-ch:
		timer.Stop()
	}
	if x != nil {
		return x.(queue.UniqueID)
	}
	return nil
}

// Variant of DelayQueue.Pop() for UniqueDelayed items
func (q *FIFOQueue) Pop() interface{} {
	return q.pop(nil).(queue.UniqueID)
}

// variant of DelayQueue.Pop that implements optional cancellation
func (q *FIFOQueue) pop(cancel chan struct{}) interface{} {
	next := func() pushableItem {
		q.lock.Lock()
		defer q.lock.Unlock()
		for {
			for q.Len() == 0 {
				signal := make(chan struct{})
				go func() {
					defer close(signal)
					q.cond.Wait()
				}()
				select {
				case <-cancel:
					// we may not have the lock yet, so
					// broadcast to abort Wait, then
					// return after lock re-acquisition
					q.cond.Broadcast()
					<-signal
					return nil
				case <-signal:
					// we have the lock, re-check
					// the queue for data...
				}
			}
			//TODO: should this just be q.Queue? If so, it deadlocks...
			x := heap.Pop(q.Queue.Queue)
			item := x.(pushableItem)
			unique := item.Value().(queue.UniqueID)
			uid := unique.GetUID()
			if _, ok := q.items[uid]; !ok {
				// item was deleted, keep looking
				continue
			}
			delete(q.items, uid)
			return item
		}
	}
	return q.Queue.pop(next, cancel)
}
