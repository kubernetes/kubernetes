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

package nodecontroller

import (
	"container/heap"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
)

// TimedValue is a value that should be processed at a designated time.
type TimedValue struct {
	Value     string
	AddedAt   time.Time
	ProcessAt time.Time
}

// now is used to test time
var now func() time.Time = time.Now

// TimedQueue is a priority heap where the lowest ProcessAt is at the front of the queue
type TimedQueue []*TimedValue

func (h TimedQueue) Len() int           { return len(h) }
func (h TimedQueue) Less(i, j int) bool { return h[i].ProcessAt.Before(h[j].ProcessAt) }
func (h TimedQueue) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *TimedQueue) Push(x interface{}) {
	*h = append(*h, x.(*TimedValue))
}

func (h *TimedQueue) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// A FIFO queue which additionally guarantees that any element can be added only once until
// it is removed.
type UniqueQueue struct {
	lock  sync.Mutex
	queue TimedQueue
	set   sets.String
}

// Adds a new value to the queue if it wasn't added before, or was explicitly removed by the
// Remove call. Returns true if new value was added.
func (q *UniqueQueue) Add(value TimedValue) bool {
	q.lock.Lock()
	defer q.lock.Unlock()

	if q.set.Has(value.Value) {
		return false
	}
	heap.Push(&q.queue, &value)
	q.set.Insert(value.Value)
	return true
}

// Replace replaces an existing value in the queue if it already exists, otherwise it does nothing.
// Returns true if the item was found.
func (q *UniqueQueue) Replace(value TimedValue) bool {
	q.lock.Lock()
	defer q.lock.Unlock()

	for i := range q.queue {
		if q.queue[i].Value != value.Value {
			continue
		}
		heap.Remove(&q.queue, i)
		heap.Push(&q.queue, &value)
		return true
	}
	return false
}

// Removes the value from the queue, so Get() call won't return it, and allow subsequent addition
// of the given value. If the value is not present does nothing and returns false.
func (q *UniqueQueue) Remove(value string) bool {
	q.lock.Lock()
	defer q.lock.Unlock()

	q.set.Delete(value)
	for i, val := range q.queue {
		if val.Value == value {
			heap.Remove(&q.queue, i)
			return true
		}
	}
	return false
}

// Returns the oldest added value that wasn't returned yet.
func (q *UniqueQueue) Get() (TimedValue, bool) {
	q.lock.Lock()
	defer q.lock.Unlock()
	if len(q.queue) == 0 {
		return TimedValue{}, false
	}
	result := heap.Pop(&q.queue).(*TimedValue)
	q.set.Delete(result.Value)
	return *result, true
}

// Head returns the oldest added value that wasn't returned yet without removing it.
func (q *UniqueQueue) Head() (TimedValue, bool) {
	q.lock.Lock()
	defer q.lock.Unlock()
	if len(q.queue) == 0 {
		return TimedValue{}, false
	}
	result := q.queue[0]
	return *result, true
}

// RateLimitedTimedQueue is a unique item priority queue ordered by the expected next time
// of execution. It is also rate limited.
type RateLimitedTimedQueue struct {
	queue   UniqueQueue
	limiter util.RateLimiter
}

// Creates new queue which will use given RateLimiter to oversee execution.
func NewRateLimitedTimedQueue(limiter util.RateLimiter) *RateLimitedTimedQueue {
	return &RateLimitedTimedQueue{
		queue: UniqueQueue{
			queue: TimedQueue{},
			set:   sets.NewString(),
		},
		limiter: limiter,
	}
}

// ActionFunc takes a timed value and returns false if the item must be retried, with an optional
// time.Duration if some minimum wait interval should be used.
type ActionFunc func(TimedValue) (bool, time.Duration)

// Try processes the queue. Ends prematurely if RateLimiter forbids an action and leak is true.
// Otherwise, requeues the item to be processed. Each value is processed once if fn returns true,
// otherwise it is added back to the queue. The returned remaining is used to identify the minimum
// time to execute the next item in the queue.
func (q *RateLimitedTimedQueue) Try(fn ActionFunc) {
	val, ok := q.queue.Head()
	for ok {
		// rate limit the queue checking
		if !q.limiter.CanAccept() {
			glog.V(10).Info("Try rate limitted...")
			// Try again later
			break
		}

		now := now()
		if now.Before(val.ProcessAt) {
			break
		}

		if ok, wait := fn(val); !ok {
			val.ProcessAt = now.Add(wait + 1)
			q.queue.Replace(val)
		} else {
			q.queue.Remove(val.Value)
		}
		val, ok = q.queue.Head()
	}
}

// Adds value to the queue to be processed. Won't add the same value a second time if it was already
// added and not removed.
func (q *RateLimitedTimedQueue) Add(value string) bool {
	now := now()
	return q.queue.Add(TimedValue{
		Value:     value,
		AddedAt:   now,
		ProcessAt: now,
	})
}

// Removes Node from the Evictor. The Node won't be processed until added again.
func (q *RateLimitedTimedQueue) Remove(value string) bool {
	return q.queue.Remove(value)
}
