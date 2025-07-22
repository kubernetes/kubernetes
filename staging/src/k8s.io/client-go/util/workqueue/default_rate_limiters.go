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

// Deprecated: RateLimiter is deprecated, use TypedRateLimiter instead.
type RateLimiter TypedRateLimiter[any]

type TypedRateLimiter[T comparable] interface {
	// When gets an item and gets to decide how long that item should wait
	When(item T) time.Duration
	// Forget indicates that an item is finished being retried.  Doesn't matter whether it's for failing
	// or for success, we'll stop tracking it
	Forget(item T)
	// NumRequeues returns back how many failures the item has had
	NumRequeues(item T) int
}

// DefaultControllerRateLimiter is a no-arg constructor for a default rate limiter for a workqueue.  It has
// both overall and per-item rate limiting.  The overall is a token bucket and the per-item is exponential
//
// Deprecated: Use DefaultTypedControllerRateLimiter instead.
func DefaultControllerRateLimiter() RateLimiter {
	return DefaultTypedControllerRateLimiter[any]()
}

// DefaultTypedControllerRateLimiter is a no-arg constructor for a default rate limiter for a workqueue.  It has
// both overall and per-item rate limiting.  The overall is a token bucket and the per-item is exponential
func DefaultTypedControllerRateLimiter[T comparable]() TypedRateLimiter[T] {
	return NewTypedMaxOfRateLimiter(
		NewTypedItemExponentialFailureRateLimiter[T](5*time.Millisecond, 1000*time.Second),
		// 10 qps, 100 bucket size.  This is only for retry speed and its only the overall factor (not per item)
		&TypedBucketRateLimiter[T]{Limiter: rate.NewLimiter(rate.Limit(10), 100)},
	)
}

// Deprecated: BucketRateLimiter is deprecated, use TypedBucketRateLimiter instead.
type BucketRateLimiter = TypedBucketRateLimiter[any]

// TypedBucketRateLimiter adapts a standard bucket to the workqueue ratelimiter API
type TypedBucketRateLimiter[T comparable] struct {
	*rate.Limiter
}

var _ RateLimiter = &BucketRateLimiter{}

func (r *TypedBucketRateLimiter[T]) When(item T) time.Duration {
	return r.Limiter.Reserve().Delay()
}

func (r *TypedBucketRateLimiter[T]) NumRequeues(item T) int {
	return 0
}

func (r *TypedBucketRateLimiter[T]) Forget(item T) {
}

// Deprecated: ItemExponentialFailureRateLimiter is deprecated, use TypedItemExponentialFailureRateLimiter instead.
type ItemExponentialFailureRateLimiter = TypedItemExponentialFailureRateLimiter[any]

// TypedItemExponentialFailureRateLimiter does a simple baseDelay*2^<num-failures> limit
// dealing with max failures and expiration are up to the caller
type TypedItemExponentialFailureRateLimiter[T comparable] struct {
	failuresLock sync.Mutex
	failures     map[T]int

	baseDelay time.Duration
	maxDelay  time.Duration
}

var _ RateLimiter = &ItemExponentialFailureRateLimiter{}

// Deprecated: NewItemExponentialFailureRateLimiter is deprecated, use NewTypedItemExponentialFailureRateLimiter instead.
func NewItemExponentialFailureRateLimiter(baseDelay time.Duration, maxDelay time.Duration) RateLimiter {
	return NewTypedItemExponentialFailureRateLimiter[any](baseDelay, maxDelay)
}

func NewTypedItemExponentialFailureRateLimiter[T comparable](baseDelay time.Duration, maxDelay time.Duration) TypedRateLimiter[T] {
	return &TypedItemExponentialFailureRateLimiter[T]{
		failures:  map[T]int{},
		baseDelay: baseDelay,
		maxDelay:  maxDelay,
	}
}

// Deprecated: DefaultItemBasedRateLimiter is deprecated, use DefaultTypedItemBasedRateLimiter instead.
func DefaultItemBasedRateLimiter() RateLimiter {
	return DefaultTypedItemBasedRateLimiter[any]()
}

func DefaultTypedItemBasedRateLimiter[T comparable]() TypedRateLimiter[T] {
	return NewTypedItemExponentialFailureRateLimiter[T](time.Millisecond, 1000*time.Second)
}

func (r *TypedItemExponentialFailureRateLimiter[T]) When(item T) time.Duration {
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

func (r *TypedItemExponentialFailureRateLimiter[T]) NumRequeues(item T) int {
	r.failuresLock.Lock()
	defer r.failuresLock.Unlock()

	return r.failures[item]
}

func (r *TypedItemExponentialFailureRateLimiter[T]) Forget(item T) {
	r.failuresLock.Lock()
	defer r.failuresLock.Unlock()

	delete(r.failures, item)
}

// ItemFastSlowRateLimiter does a quick retry for a certain number of attempts, then a slow retry after that
// Deprecated: Use TypedItemFastSlowRateLimiter instead.
type ItemFastSlowRateLimiter = TypedItemFastSlowRateLimiter[any]

// TypedItemFastSlowRateLimiter does a quick retry for a certain number of attempts, then a slow retry after that
type TypedItemFastSlowRateLimiter[T comparable] struct {
	failuresLock sync.Mutex
	failures     map[T]int

	maxFastAttempts int
	fastDelay       time.Duration
	slowDelay       time.Duration
}

var _ RateLimiter = &ItemFastSlowRateLimiter{}

// Deprecated: NewItemFastSlowRateLimiter is deprecated, use NewTypedItemFastSlowRateLimiter instead.
func NewItemFastSlowRateLimiter(fastDelay, slowDelay time.Duration, maxFastAttempts int) RateLimiter {
	return NewTypedItemFastSlowRateLimiter[any](fastDelay, slowDelay, maxFastAttempts)
}

func NewTypedItemFastSlowRateLimiter[T comparable](fastDelay, slowDelay time.Duration, maxFastAttempts int) TypedRateLimiter[T] {
	return &TypedItemFastSlowRateLimiter[T]{
		failures:        map[T]int{},
		fastDelay:       fastDelay,
		slowDelay:       slowDelay,
		maxFastAttempts: maxFastAttempts,
	}
}

func (r *TypedItemFastSlowRateLimiter[T]) When(item T) time.Duration {
	r.failuresLock.Lock()
	defer r.failuresLock.Unlock()

	r.failures[item] = r.failures[item] + 1

	if r.failures[item] <= r.maxFastAttempts {
		return r.fastDelay
	}

	return r.slowDelay
}

func (r *TypedItemFastSlowRateLimiter[T]) NumRequeues(item T) int {
	r.failuresLock.Lock()
	defer r.failuresLock.Unlock()

	return r.failures[item]
}

func (r *TypedItemFastSlowRateLimiter[T]) Forget(item T) {
	r.failuresLock.Lock()
	defer r.failuresLock.Unlock()

	delete(r.failures, item)
}

// MaxOfRateLimiter calls every RateLimiter and returns the worst case response
// When used with a token bucket limiter, the burst could be apparently exceeded in cases where particular items
// were separately delayed a longer time.
//
// Deprecated: Use TypedMaxOfRateLimiter instead.
type MaxOfRateLimiter = TypedMaxOfRateLimiter[any]

// TypedMaxOfRateLimiter calls every RateLimiter and returns the worst case response
// When used with a token bucket limiter, the burst could be apparently exceeded in cases where particular items
// were separately delayed a longer time.
type TypedMaxOfRateLimiter[T comparable] struct {
	limiters []TypedRateLimiter[T]
}

func (r *TypedMaxOfRateLimiter[T]) When(item T) time.Duration {
	ret := time.Duration(0)
	for _, limiter := range r.limiters {
		curr := limiter.When(item)
		if curr > ret {
			ret = curr
		}
	}

	return ret
}

// Deprecated: NewMaxOfRateLimiter is deprecated, use NewTypedMaxOfRateLimiter instead.
func NewMaxOfRateLimiter(limiters ...TypedRateLimiter[any]) RateLimiter {
	return NewTypedMaxOfRateLimiter(limiters...)
}

func NewTypedMaxOfRateLimiter[T comparable](limiters ...TypedRateLimiter[T]) TypedRateLimiter[T] {
	return &TypedMaxOfRateLimiter[T]{limiters: limiters}
}

func (r *TypedMaxOfRateLimiter[T]) NumRequeues(item T) int {
	ret := 0
	for _, limiter := range r.limiters {
		curr := limiter.NumRequeues(item)
		if curr > ret {
			ret = curr
		}
	}

	return ret
}

func (r *TypedMaxOfRateLimiter[T]) Forget(item T) {
	for _, limiter := range r.limiters {
		limiter.Forget(item)
	}
}

// WithMaxWaitRateLimiter have maxDelay which avoids waiting too long
// Deprecated: Use TypedWithMaxWaitRateLimiter instead.
type WithMaxWaitRateLimiter = TypedWithMaxWaitRateLimiter[any]

// TypedWithMaxWaitRateLimiter have maxDelay which avoids waiting too long
type TypedWithMaxWaitRateLimiter[T comparable] struct {
	limiter  TypedRateLimiter[T]
	maxDelay time.Duration
}

// Deprecated: NewWithMaxWaitRateLimiter is deprecated, use NewTypedWithMaxWaitRateLimiter instead.
func NewWithMaxWaitRateLimiter(limiter RateLimiter, maxDelay time.Duration) RateLimiter {
	return NewTypedWithMaxWaitRateLimiter[any](limiter, maxDelay)
}

func NewTypedWithMaxWaitRateLimiter[T comparable](limiter TypedRateLimiter[T], maxDelay time.Duration) TypedRateLimiter[T] {
	return &TypedWithMaxWaitRateLimiter[T]{limiter: limiter, maxDelay: maxDelay}
}

func (w TypedWithMaxWaitRateLimiter[T]) When(item T) time.Duration {
	delay := w.limiter.When(item)
	if delay > w.maxDelay {
		return w.maxDelay
	}

	return delay
}

func (w TypedWithMaxWaitRateLimiter[T]) Forget(item T) {
	w.limiter.Forget(item)
}

func (w TypedWithMaxWaitRateLimiter[T]) NumRequeues(item T) int {
	return w.limiter.NumRequeues(item)
}
