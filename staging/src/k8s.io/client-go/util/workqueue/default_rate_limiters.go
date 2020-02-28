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
	"math"
	"sync"
	"time"

	"golang.org/x/time/rate"
)

type RateLimiter interface {
	// When gets an item and gets to decide how long that item should wait
	When(item interface{}) time.Duration
	// Forget indicates that an item is finished being retried.  Doesn't matter whether its for perm failing
	// or for success, we'll stop tracking it
	Forget(item interface{})
	// NumRequeues returns back how many failures the item has had
	NumRequeues(item interface{}) int
}

// DefaultControllerRateLimiter is a no-arg constructor for a default rate limiter for a workqueue.  It has
// both overall and per-item rate limiting.  The overall is a token bucket and the per-item is exponential
func DefaultControllerRateLimiter() RateLimiter {
	return NewMaxOfRateLimiter(
		NewItemExponentialFailureRateLimiter(5*time.Millisecond, 1000*time.Second),
		// 10 qps, 100 bucket size.  This is only for retry speed and its only the overall factor (not per item)
		&BucketRateLimiter{Limiter: rate.NewLimiter(rate.Limit(10), 100)},
	)
}

// BucketRateLimiter adapts a standard bucket to the workqueue ratelimiter API
type BucketRateLimiter struct {
	*rate.Limiter
}

var _ RateLimiter = &BucketRateLimiter{}

func (r *BucketRateLimiter) When(item interface{}) time.Duration {
	return r.Limiter.Reserve().Delay()
}

func (r *BucketRateLimiter) NumRequeues(item interface{}) int {
	return 0
}

func (r *BucketRateLimiter) Forget(item interface{}) {
}

// itemState holds state for a single item in ItemBucketRateLimiter.
type itemState struct {
	// nextBatchStart is a time when the next batch starts.
	// All requests before this batch should join this batch.
	nextBatchStart time.Time
	// limiter is used to rate limit starting new batches.
	limiter *rate.Limiter
}

// ItemBucketRateLimiter implements a workqueue ratelimiter API using standard rate.Limiter.
// The rate limiter connects multiple items with the same key into batches.
// It always keeps up to one future batch. Items can join that batch as long as it's not started yet.
// Creating a new batch is rate limited according to qps and burst.
//
// Unlike other RateLimiters in this file, this RateLimiter isn't intended to be used to rate
// limit error retries, rather the successful path.
//
// Intended usage:
// // With qps = 1.0, burst = 1, the first batch can start instantanously, the next every 1s.
// r := NewItemBucketRateLimiter(1.0, 1)
//
// t := time.Now()
// queue.AddAfter("key1", r.When("key1")) // = AddAfter("key1", 0) => burst = 1 so it can go now.
// time.Sleep(1 * time.Millisecond)
//
// queue.AddAfter("key1", r.When("key1")) // = AddAfter("key1", ~1s) => scheduled to be run at t + 1s
// queue.AddAfter("key1", r.When("key1")) // = AddAfter("key1", ~1s) => scheduled to be run at t + 1s
//
// time.Sleep(1 * time.Second) // The t + 1s batch has started already.
// queue.AddAfter("key1", r.When("key1")) // = AddAfter("key1", 1s)  // scheduled to be run at the next batch
// queue.AddAfter("key2", r.When("key2"))
//
// // When key1 is deleted from the system.
// queue.Forget("key1")
//
type ItemBucketRateLimiter struct {
	qps   float64
	burst int

	limitersLock sync.Mutex
	limiters     map[interface{}]*itemState
}

var _ RateLimiter = &ItemBucketRateLimiter{}

// NewItemBucketRateLimiter creates new ItemBucketRateLimiter instance.
func NewItemBucketRateLimiter(qps float64, burst int) *ItemBucketRateLimiter {
	return &ItemBucketRateLimiter{
		qps:      qps,
		burst:    burst,
		limiters: make(map[interface{}]*itemState),
	}
}

// When returns a time.Duration which we need to wait before item is processed.
func (r *ItemBucketRateLimiter) When(item interface{}) time.Duration {
	r.limitersLock.Lock()
	defer r.limitersLock.Unlock()

	entry, ok := r.limiters[item]
	if !ok {
		entry = &itemState{
			// Set nextBucketStart far in past: January 1, year 1, 00:00:00 UTC.
			nextBatchStart: time.Time{},
			limiter:        rate.NewLimiter(rate.Limit(r.qps), r.burst),
		}
		r.limiters[item] = entry
	}

	// Check when the next bucket starts, if it hasn't started yet, try to join that bucket.
	if entry.nextBatchStart.After(time.Now()) {
		return entry.nextBatchStart.Sub(time.Now())
	}

	delay := entry.limiter.Reserve().Delay()
	entry.nextBatchStart = time.Now().Add(delay)
	return delay
}

// NumRequeues returns always 0 (doesn't apply to ItemBucketRateLimiter).
func (r *ItemBucketRateLimiter) NumRequeues(item interface{}) int {
	return 0
}

// Forget removes item from the internal state.
func (r *ItemBucketRateLimiter) Forget(item interface{}) {
	r.limitersLock.Lock()
	defer r.limitersLock.Unlock()

	delete(r.limiters, item)
}

// ItemExponentialFailureRateLimiter does a simple baseDelay*2^<num-failures> limit
// dealing with max failures and expiration are up to the caller
type ItemExponentialFailureRateLimiter struct {
	failuresLock sync.Mutex
	failures     map[interface{}]int

	baseDelay time.Duration
	maxDelay  time.Duration
}

var _ RateLimiter = &ItemExponentialFailureRateLimiter{}

func NewItemExponentialFailureRateLimiter(baseDelay time.Duration, maxDelay time.Duration) RateLimiter {
	return &ItemExponentialFailureRateLimiter{
		failures:  map[interface{}]int{},
		baseDelay: baseDelay,
		maxDelay:  maxDelay,
	}
}

func DefaultItemBasedRateLimiter() RateLimiter {
	return NewItemExponentialFailureRateLimiter(time.Millisecond, 1000*time.Second)
}

func (r *ItemExponentialFailureRateLimiter) When(item interface{}) time.Duration {
	r.failuresLock.Lock()
	defer r.failuresLock.Unlock()

	exp := r.failures[item]
	r.failures[item] = r.failures[item] + 1

	// The backoff is capped such that 'calculated' value never overflows.
	backoff := float64(r.baseDelay.Nanoseconds()) * math.Pow(2, float64(exp))
	if backoff > math.MaxInt64 {
		return r.maxDelay
	}

	calculated := time.Duration(backoff)
	if calculated > r.maxDelay {
		return r.maxDelay
	}

	return calculated
}

func (r *ItemExponentialFailureRateLimiter) NumRequeues(item interface{}) int {
	r.failuresLock.Lock()
	defer r.failuresLock.Unlock()

	return r.failures[item]
}

func (r *ItemExponentialFailureRateLimiter) Forget(item interface{}) {
	r.failuresLock.Lock()
	defer r.failuresLock.Unlock()

	delete(r.failures, item)
}

// ItemFastSlowRateLimiter does a quick retry for a certain number of attempts, then a slow retry after that
type ItemFastSlowRateLimiter struct {
	failuresLock sync.Mutex
	failures     map[interface{}]int

	maxFastAttempts int
	fastDelay       time.Duration
	slowDelay       time.Duration
}

var _ RateLimiter = &ItemFastSlowRateLimiter{}

func NewItemFastSlowRateLimiter(fastDelay, slowDelay time.Duration, maxFastAttempts int) RateLimiter {
	return &ItemFastSlowRateLimiter{
		failures:        map[interface{}]int{},
		fastDelay:       fastDelay,
		slowDelay:       slowDelay,
		maxFastAttempts: maxFastAttempts,
	}
}

func (r *ItemFastSlowRateLimiter) When(item interface{}) time.Duration {
	r.failuresLock.Lock()
	defer r.failuresLock.Unlock()

	r.failures[item] = r.failures[item] + 1

	if r.failures[item] <= r.maxFastAttempts {
		return r.fastDelay
	}

	return r.slowDelay
}

func (r *ItemFastSlowRateLimiter) NumRequeues(item interface{}) int {
	r.failuresLock.Lock()
	defer r.failuresLock.Unlock()

	return r.failures[item]
}

func (r *ItemFastSlowRateLimiter) Forget(item interface{}) {
	r.failuresLock.Lock()
	defer r.failuresLock.Unlock()

	delete(r.failures, item)
}

// MaxOfRateLimiter calls every RateLimiter and returns the worst case response
// When used with a token bucket limiter, the burst could be apparently exceeded in cases where particular items
// were separately delayed a longer time.
type MaxOfRateLimiter struct {
	limiters []RateLimiter
}

func (r *MaxOfRateLimiter) When(item interface{}) time.Duration {
	ret := time.Duration(0)
	for _, limiter := range r.limiters {
		curr := limiter.When(item)
		if curr > ret {
			ret = curr
		}
	}

	return ret
}

func NewMaxOfRateLimiter(limiters ...RateLimiter) RateLimiter {
	return &MaxOfRateLimiter{limiters: limiters}
}

func (r *MaxOfRateLimiter) NumRequeues(item interface{}) int {
	ret := 0
	for _, limiter := range r.limiters {
		curr := limiter.NumRequeues(item)
		if curr > ret {
			ret = curr
		}
	}

	return ret
}

func (r *MaxOfRateLimiter) Forget(item interface{}) {
	for _, limiter := range r.limiters {
		limiter.Forget(item)
	}
}
