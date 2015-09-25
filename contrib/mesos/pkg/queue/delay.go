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
)

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

// returns a non-nil value from the queue, or else nil if/when cancelled; if cancel
// is nil then cancellation is disabled and this func must return a non-nil value.
func (q *DelayQueue) pop(next func() *qitem, cancel <-chan struct{}) interface{} {
	var waitCh chan struct{}
	var delayTimer *time.Timer
	for {
		item := next()
		if item == nil {
			// cancelled
			return nil
		}
		x := item.value
		delayedTime := item.priority.ts
		if delayedTime.After(time.Now()) {
			// listen for calls to Add() while we're waiting for the deadline
			if waitCh == nil {
				waitCh = make(chan struct{}, 1)
			}
			if delayTimer == nil {
				delayTimer = time.NewTimer(delayedTime.Sub(time.Now()))
			}
			go func() {
				q.lock.Lock()
				defer q.lock.Unlock()
				q.cond.Wait()
				waitCh <- struct{}{}
			}()
			select {
			case <-cancel:
				item.readd(item)
				return nil
			case <-waitCh:
				// we may no longer have the earliest deadline, re-try
				item.readd(item)
				continue
			case <-delayTimer.C:
				// noop
			case <-item.priority.notify:
				// noop
			}
		}
		return x
	}
}
