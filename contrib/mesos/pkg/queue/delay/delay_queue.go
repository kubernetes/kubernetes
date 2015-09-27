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
	"sync"

	"github.com/pivotal-golang/clock"

	"k8s.io/kubernetes/contrib/mesos/pkg/queue/priority"
)

// Queue is a thread-safe, time-based queue.
//
// Items can only be popped after their event time has been reached.
// Use Add to specify a delay duration and have the event time calculated.
// Use Offer to specify a specific event time.
type Queue struct {
	*priority.Queue
	lock  *sync.RWMutex
	cond  *sync.Cond
	clock clock.Clock
}

func NewDelayQueue(clock clock.Clock) *Queue {
	var lock sync.RWMutex
	return &Queue{
		Queue: priority.NewPriorityQueue(),
		lock:  &lock,
		cond:  sync.NewCond(&lock),
		clock: clock,
	}
}

func (q *Queue) Add(d Delayed) {
	p := NewDelayedPriority(d, q.clock)

	item := &addItem{
		Item: priority.NewItem(d, p),
		lock: q.lock,
		cond: q.cond,
	}
	item.Push(q.Queue)
}

// addItem adds locking and broadcasting to priority.Item.Push
type addItem struct {
	priority.Item
	lock *sync.RWMutex
	cond *sync.Cond
}

func (di *addItem) Push(queue heap.Interface) {
	di.lock.Lock()
	defer di.lock.Unlock()
	heap.Push(queue, di)
	di.cond.Broadcast()
}

// If there's a eventTime reported by d.EventTime() then `d` is added to the
// queue and this func returns true.
func (q *Queue) Offer(d Scheduled) bool {
	eventTime, ok := NewScheduledPriority(d)
	if ok {
		q.lock.Lock()
		defer q.lock.Unlock()
		heap.Push(q.Queue, &offerItem{
			Item: priority.NewItem(d, eventTime),
		})
		q.cond.Broadcast()
	}
	return ok
}

// offerItem offers the value priority.Item.Push
type offerItem struct {
	priority.Item
}

func (di *offerItem) Push(queue heap.Interface) {
	dq := queue.(*Queue)
	dq.Offer(di.Value().(Scheduled))
}

// wait for the delay of the next item in the queue to expire, blocking if
// there are no items in the queue. does not guarantee first-come-first-serve
// ordering with respect to clients.
func (q *Queue) Pop() interface{} {
	// doesn't implement cancellation, will always return a non-nil value
	return q.pop(func() pushableItem {
		q.lock.Lock()
		defer q.lock.Unlock()
		for q.Len() == 0 {
			q.cond.Wait()
		}
		x := heap.Pop(q.Queue)
		item := x.(pushableItem)
		return item
	}, nil)
}

// returns a non-nil value from the queue, or else nil if/when cancelled; if cancel
// is nil then cancellation is disabled and this func must return a non-nil value.
func (q *Queue) pop(next func() pushableItem, cancel <-chan struct{}) interface{} {
	var condCh chan struct{}
	var delayTimer clock.Timer
	for {
		item := next()
		if item == nil {
			// cancelled
			return nil
		}
		x := item.Value()
		delayPriority := item.Priority().(Priority)
		delayedTime := delayPriority.eventTime
		if delayedTime.After(q.clock.Now()) {
			// listen for calls to Add() while we're waiting for the eventTime
			if condCh == nil {
				condCh = make(chan struct{}, 1)
			}
			if delayTimer == nil {
				delayTimer = q.clock.NewTimer(delayedTime.Sub(q.clock.Now()))
				defer delayTimer.Stop()
				//TODO: delayTimer.Stop() sooner
			}
			go func() {
				q.lock.Lock()
				defer func() {
					q.lock.Unlock()
					condCh <- struct{}{}
				}()
				q.cond.Wait()
			}()
			select {
			case <-cancel:
				item.Push(q.Queue)
				return nil
			case <-condCh:
				// we may no longer have the earliest eventTime, re-try
				item.Push(q.Queue)
				continue
			case <-delayTimer.C():
				// noop
			case <-delayPriority.notify:
				// noop
			}
		}
		return x
	}
}

// pushableItem is an item that can push itself onto a heap.
type pushableItem interface {
	priority.Item
	// Push this Item into a heap.
	Push(heap.Interface)
}
