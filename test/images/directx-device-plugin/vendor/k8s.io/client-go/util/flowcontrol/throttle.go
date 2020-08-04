/*
Copyright 2014 The Kubernetes Authors.

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

package flowcontrol

import (
	"sync"
	"time"

	"golang.org/x/time/rate"
)

type RateLimiter interface {
	// TryAccept returns true if a token is taken immediately. Otherwise,
	// it returns false.
	TryAccept() bool
	// Accept returns once a token becomes available.
	Accept()
	// Stop stops the rate limiter, subsequent calls to CanAccept will return false
	Stop()
	// QPS returns QPS of this rate limiter
	QPS() float32
}

type tokenBucketRateLimiter struct {
	limiter *rate.Limiter
	clock   Clock
	qps     float32
}

// NewTokenBucketRateLimiter creates a rate limiter which implements a token bucket approach.
// The rate limiter allows bursts of up to 'burst' to exceed the QPS, while still maintaining a
// smoothed qps rate of 'qps'.
// The bucket is initially filled with 'burst' tokens, and refills at a rate of 'qps'.
// The maximum number of tokens in the bucket is capped at 'burst'.
func NewTokenBucketRateLimiter(qps float32, burst int) RateLimiter {
	limiter := rate.NewLimiter(rate.Limit(qps), burst)
	return newTokenBucketRateLimiter(limiter, realClock{}, qps)
}

// An injectable, mockable clock interface.
type Clock interface {
	Now() time.Time
	Sleep(time.Duration)
}

type realClock struct{}

func (realClock) Now() time.Time {
	return time.Now()
}
func (realClock) Sleep(d time.Duration) {
	time.Sleep(d)
}

// NewTokenBucketRateLimiterWithClock is identical to NewTokenBucketRateLimiter
// but allows an injectable clock, for testing.
func NewTokenBucketRateLimiterWithClock(qps float32, burst int, c Clock) RateLimiter {
	limiter := rate.NewLimiter(rate.Limit(qps), burst)
	return newTokenBucketRateLimiter(limiter, c, qps)
}

func newTokenBucketRateLimiter(limiter *rate.Limiter, c Clock, qps float32) RateLimiter {
	return &tokenBucketRateLimiter{
		limiter: limiter,
		clock:   c,
		qps:     qps,
	}
}

func (t *tokenBucketRateLimiter) TryAccept() bool {
	return t.limiter.AllowN(t.clock.Now(), 1)
}

// Accept will block until a token becomes available
func (t *tokenBucketRateLimiter) Accept() {
	now := t.clock.Now()
	t.clock.Sleep(t.limiter.ReserveN(now, 1).DelayFrom(now))
}

func (t *tokenBucketRateLimiter) Stop() {
}

func (t *tokenBucketRateLimiter) QPS() float32 {
	return t.qps
}

type fakeAlwaysRateLimiter struct{}

func NewFakeAlwaysRateLimiter() RateLimiter {
	return &fakeAlwaysRateLimiter{}
}

func (t *fakeAlwaysRateLimiter) TryAccept() bool {
	return true
}

func (t *fakeAlwaysRateLimiter) Stop() {}

func (t *fakeAlwaysRateLimiter) Accept() {}

func (t *fakeAlwaysRateLimiter) QPS() float32 {
	return 1
}

type fakeNeverRateLimiter struct {
	wg sync.WaitGroup
}

func NewFakeNeverRateLimiter() RateLimiter {
	rl := fakeNeverRateLimiter{}
	rl.wg.Add(1)
	return &rl
}

func (t *fakeNeverRateLimiter) TryAccept() bool {
	return false
}

func (t *fakeNeverRateLimiter) Stop() {
	t.wg.Done()
}

func (t *fakeNeverRateLimiter) Accept() {
	t.wg.Wait()
}

func (t *fakeNeverRateLimiter) QPS() float32 {
	return 1
}
