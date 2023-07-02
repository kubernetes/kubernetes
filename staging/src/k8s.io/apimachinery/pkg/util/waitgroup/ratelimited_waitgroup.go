/*
Copyright 2023 The Kubernetes Authors.

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

package waitgroup

import (
	"context"
	"fmt"
	"sync"
)

// RateLimiter abstracts the rate limiter used by RateLimitedSafeWaitGroup.
// The implementation must be thread-safe.
type RateLimiter interface {
	Wait(ctx context.Context) error
}

// RateLimiterFactoryFunc is used by the RateLimitedSafeWaitGroup to create a new
// instance of a RateLimiter that will be used to rate limit the return rate
// of the active number of request(s). 'count' is the number of requests in
// flight that are expected to invoke 'Done' on this wait group.
type RateLimiterFactoryFunc func(count int) (RateLimiter, context.Context, context.CancelFunc)

// RateLimitedSafeWaitGroup must not be copied after first use.
type RateLimitedSafeWaitGroup struct {
	wg sync.WaitGroup
	// Once Wait is initiated, all consecutive Done invocation will be
	// rate limited using this rate limiter.
	limiter RateLimiter
	stopCtx context.Context

	mu sync.Mutex
	// wait indicate whether Wait is called, if true,
	// then any Add with positive delta will return error.
	wait bool
	// number of request(s) currently using the wait group
	count int
}

// Add adds delta, which may be negative, similar to sync.WaitGroup.
// If Add with a positive delta happens after Wait, it will return error,
// which prevent unsafe Add.
func (wg *RateLimitedSafeWaitGroup) Add(delta int) error {
	wg.mu.Lock()
	defer wg.mu.Unlock()

	if wg.wait && delta > 0 {
		return fmt.Errorf("add with positive delta after Wait is forbidden")
	}
	wg.wg.Add(delta)
	wg.count += delta
	return nil
}

// Done decrements the WaitGroup counter, rate limiting is applied only
// when the wait group is in waiting mode.
func (wg *RateLimitedSafeWaitGroup) Done() {
	var limiter RateLimiter
	func() {
		wg.mu.Lock()
		defer wg.mu.Unlock()

		wg.count -= 1
		if wg.wait {
			// we are using the limiter outside the scope of the lock
			limiter = wg.limiter
		}
	}()

	defer wg.wg.Done()
	if limiter != nil {
		limiter.Wait(wg.stopCtx)
	}
}

// Wait blocks until the WaitGroup counter is zero or a hard limit has elapsed.
// It returns the number of active request(s) accounted for at the time Wait
// has been invoked, number of request(s) that have drianed (done using the
// wait group immediately before Wait returns).
// Ideally, the both numbers returned should be equal, to indicate that all
// request(s) using the wait group have released their lock.
func (wg *RateLimitedSafeWaitGroup) Wait(limiterFactory RateLimiterFactoryFunc) (int, int, error) {
	if limiterFactory == nil {
		return 0, 0, fmt.Errorf("rate limiter factory must be specified")
	}

	var cancel context.CancelFunc
	var countNow, countAfter int
	func() {
		wg.mu.Lock()
		defer wg.mu.Unlock()

		wg.limiter, wg.stopCtx, cancel = limiterFactory(wg.count)
		countNow = wg.count
		wg.wait = true
	}()

	defer cancel()
	// there should be a hard stop, in case request(s) are not responsive
	// enough to invoke Done before the grace period is over.
	waitDoneCh := make(chan struct{})
	go func() {
		defer close(waitDoneCh)
		wg.wg.Wait()
	}()

	var err error
	select {
	case <-wg.stopCtx.Done():
		err = wg.stopCtx.Err()
	case <-waitDoneCh:
	}

	func() {
		wg.mu.Lock()
		defer wg.mu.Unlock()

		countAfter = wg.count
	}()
	return countNow, countAfter, err
}
