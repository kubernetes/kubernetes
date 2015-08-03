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
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/golang/glog"
)

// A FIFO queue which additionally guarantees that any element can be added only once until
// it is removed.
type UniqueQueue struct {
	lock  sync.Mutex
	queue []string
	set   util.StringSet
}

// Entity responsible for evicting Pods from inserted Nodes. It uses RateLimiter to avoid
// evicting everything at once. Note that we rate limit eviction of Nodes not individual Pods.
type PodEvictor struct {
	queue                   UniqueQueue
	deletingPodsRateLimiter util.RateLimiter
}

// Adds a new value to the queue if it wasn't added before, or was explicitly removed by the
// Remove call. Returns true if new value was added.
func (q *UniqueQueue) Add(value string) bool {
	q.lock.Lock()
	defer q.lock.Unlock()

	if !q.set.Has(value) {
		q.queue = append(q.queue, value)
		q.set.Insert(value)
		return true
	} else {
		return false
	}
}

// Removes the value from the queue, so Get() call won't return it, and allow subsequent addition
// of the given value. If the value is not present does nothing and returns false.
func (q *UniqueQueue) Remove(value string) bool {
	q.lock.Lock()
	defer q.lock.Unlock()

	q.set.Delete(value)
	for i, val := range q.queue {
		if val == value {
			if i > 0 && i < len(q.queue)-1 {
				q.queue = append(q.queue[0:i], q.queue[i+1:len(q.queue)]...)
			} else if i > 0 {
				q.queue = q.queue[0 : len(q.queue)-1]
			} else {
				q.queue = q.queue[1:len(q.queue)]
			}
			return true
		}
	}
	return false
}

// Returns the oldest added value that wasn't returned yet.
func (q *UniqueQueue) Get() (string, bool) {
	q.lock.Lock()
	defer q.lock.Unlock()
	if len(q.queue) == 0 {
		return "", false
	}

	result := q.queue[0]
	q.queue = q.queue[1:len(q.queue)]
	return result, true
}

// Creates new PodEvictor which will use given RateLimiter to oversee eviction.
func NewPodEvictor(deletingPodsRateLimiter util.RateLimiter) *PodEvictor {
	return &PodEvictor{
		queue: UniqueQueue{
			queue: make([]string, 0),
			set:   util.NewStringSet(),
		},
		deletingPodsRateLimiter: deletingPodsRateLimiter,
	}
}

// Tries to evict all Pods from previously inserted Nodes. Ends prematurely if RateLimiter forbids any eviction.
// Each Node is processed only once, as long as it's not Removed, i.e. calling multiple AddNodeToEvict does not result
// with multiple evictions as long as RemoveNodeToEvict is not called.
func (pe *PodEvictor) TryEvict(delFunc func(string)) {
	val, ok := pe.queue.Get()
	for ok {
		if pe.deletingPodsRateLimiter.CanAccept() {
			glog.Infof("PodEvictor is evicting Pods on Node: %v", val)
			delFunc(val)
		} else {
			glog.V(1).Info("PodEvictor is rate limitted.")
			break
		}
		val, ok = pe.queue.Get()
	}
}

// Adds Node to the Evictor to be processed later. Won't add the same Node second time if it was already
// added and not removed.
func (pe *PodEvictor) AddNodeToEvict(nodeName string) bool {
	return pe.queue.Add(nodeName)
}

// Removes Node from the Evictor. The Node won't be processed until added again.
func (pe *PodEvictor) RemoveNodeToEvict(nodeName string) bool {
	return pe.queue.Remove(nodeName)
}
