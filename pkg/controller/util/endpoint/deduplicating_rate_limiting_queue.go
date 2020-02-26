/*
Copyright 2020 The Kubernetes Authors.

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

package endpoint

import (
	"sync"

	"k8s.io/client-go/util/workqueue"
)

// DeduplicatingRateLimitingQueue implements a wrapper on workqueue.RateLimitingInterface that also rate limits initial key additions
// to the queue (unlike workqueue.RateLimitingInterface which rate limits only error retries).
// TODO(mborsz): Promote to workqueue package once we validate API in endpoint controllers.
type DeduplicatingRateLimitingQueue struct {
	queue       workqueue.RateLimitingInterface
	rateLimiter workqueue.RateLimiter

	// notStarted keeps a set of keys that has been added to queue and hasn't been released to the reader yet.
	notStarted     map[interface{}]bool
	notStartedLock sync.Mutex
}

// NewDeduplicatingRateLimitingQueue creates a new DeduplicatingRateLimitingQueue instance.
func NewDeduplicatingRateLimitingQueue(queue workqueue.RateLimitingInterface, rateLimiter workqueue.RateLimiter) *DeduplicatingRateLimitingQueue {
	return &DeduplicatingRateLimitingQueue{
		queue:       queue,
		rateLimiter: rateLimiter,
		notStarted:  make(map[interface{}]bool),
	}
}

// Enqueue adds key to the queue if it's not there yet.
func (r *DeduplicatingRateLimitingQueue) Enqueue(key interface{}, rateLimit bool) {
	r.notStartedLock.Lock()
	defer r.notStartedLock.Unlock()
	if _, ok := r.notStarted[key]; ok {
		// The key is waiting in queue.
		// There is no point in readding the key, so let's save some rate limiter tokens.
		return
	}
	r.notStarted[key] = true
	if rateLimit {
		// We consume rate limiter's token only if we really add element to the queue.
		r.queue.AddAfter(key, r.rateLimiter.When(key))
	} else {
		r.queue.Add(key)
	}
}

// EnqueueRetry adds key back to the queue after error occurred while handling.
// Please note that adding back is not subject of rate limiting at RateLimitingQueue level -- it will
// be rate limited by the queue itself (potentially using different rate limiter).
func (r *DeduplicatingRateLimitingQueue) EnqueueRetry(key interface{}) {
	r.queue.AddRateLimited(key)
}

// Get blocks until the next element is available or queue is shutdown.
// We cannot take a lock for r.queue.Get() as it can block for a long time, but that's fine:
func (r *DeduplicatingRateLimitingQueue) Get() (interface{}, bool) {
	key, quit := r.queue.Get()

	// If Enqueue takes the lock at this point, it will skip adding element to queue which is correct
	// as the caller of Get() hasn't started processing of the key yet.
	r.notStartedLock.Lock()
	defer r.notStartedLock.Unlock()
	delete(r.notStarted, key)
	return key, quit
}

// DeleteKey marks that given key has been deleted from the system completely and we should free resources associated with it.
func (r *DeduplicatingRateLimitingQueue) DeleteKey(key interface{}) {
	r.rateLimiter.Forget(key)
	r.queue.Forget(key)

	r.notStartedLock.Lock()
	defer r.notStartedLock.Unlock()
	delete(r.notStarted, key)
}

// Expose methods from r.queue. We cannot embed workqueue.RateLimitingInterface directly in RateLimitingQueue to expose them automatically,
// as this would expose AddRateLimited method that would have confising name in context of this struct and would be errorprone
// ("rate limited" in AddRateLimited refers to rate limiting of error handling).

// Done marks that given attempt to process key has finished (potentially with error) and the next attempt can start.
func (r *DeduplicatingRateLimitingQueue) Done(key interface{}) {
	r.queue.Done(key)
}

// Forget marks that all attempts to retry given key (either because of success or exceeding number of retries) has finished.
func (r *DeduplicatingRateLimitingQueue) Forget(key interface{}) {
	r.queue.Forget(key)
}

// ShutDown frees resources used by the queue.
func (r *DeduplicatingRateLimitingQueue) ShutDown() {
	r.queue.ShutDown()
}

// NumRequeues returns a number of past attempts to retry processing failing key.
func (r *DeduplicatingRateLimitingQueue) NumRequeues(key interface{}) int {
	return r.queue.NumRequeues(key)
}

// Len returns a queue depth.
func (r *DeduplicatingRateLimitingQueue) Len() int {
	return r.queue.Len()
}
