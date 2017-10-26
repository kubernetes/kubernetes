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
	"context"
	"sync"

	"k8s.io/apimachinery/pkg/util/clock"
)

type RateLimiter interface {
	// TryAccept returns true if a token is taken immediately. Otherwise,
	// it returns false.
	TryAccept() bool
	// Accept returns once a token becomes available.
	Accept()
	// AcceptContext returns once a token becomes available or the context is done.  If the latter,
	// context.Err() is returned.
	AcceptContext(context.Context) error
	// Stop stops the rate limiter, subsequent calls to CanAccept will return false
	Stop()
	// Saturation returns a percentage number which describes how saturated
	// this rate limiter is.
	// Usually we use token bucket rate limiter. In that case,
	// 1.0 means no tokens are available; 0.0 means we have a full bucket of tokens to use.
	Saturation() float64
	// QPS returns QPS of this rate limiter
	QPS() float32
}

type tokenBucketRateLimiter struct {
	limiter Limiter
	qps     float32
}

// NewTokenBucketRateLimiter creates a rate limiter which implements a token bucket approach.
// The rate limiter allows bursts of up to 'burst' to exceed the QPS, while still maintaining a
// smoothed qps rate of 'qps'.
// The bucket is initially filled with 'burst' tokens, and refills at a rate of 'qps'.
// The maximum number of tokens in the bucket is capped at 'burst'.
func NewTokenBucketRateLimiter(qps float32, burst int) (RateLimiter, error) {
	limiter, err := NewBucketLimiter(float64(qps), int64(burst))
	if err != nil {
		return nil, err
	}
	return newTokenBucketRateLimiter(limiter, qps), nil
}

func MustNewTokenBucketRateLimiter(qps float32, burst int) RateLimiter {
	rlimiter, err := NewTokenBucketRateLimiter(qps, burst)
	if err != nil {
		panic(err)
	}
	return rlimiter
}

// NewTokenBucketRateLimiterWithClock is identical to NewTokenBucketRateLimiter
// but allows an injectable clock, for testing.
func NewTokenBucketRateLimiterWithClock(qps float32, burst int, clock clock.Clock) (RateLimiter, error) {
	limiter, err := NewBucketLimiterWithClock(float64(qps), int64(burst), clock)
	if err != nil {
		return nil, err
	}
	return newTokenBucketRateLimiter(limiter, qps), nil
}

func MustNewTokenBucketRateLimiterWithClock(qps float32, burst int, clock clock.Clock) RateLimiter {
	rlimiter, err := NewTokenBucketRateLimiterWithClock(qps, burst, clock)
	if err != nil {
		panic(err)
	}
	return rlimiter
}

func newTokenBucketRateLimiter(limiter Limiter, qps float32) RateLimiter {
	return &tokenBucketRateLimiter{
		limiter: limiter,
		qps:     qps,
	}
}

func (t *tokenBucketRateLimiter) TryAccept() bool {
	return t.limiter.TakeAvailable(1) == 1
}

func (t *tokenBucketRateLimiter) Saturation() float64 {
	capacity := t.limiter.Capacity()
	avail := t.limiter.Available()
	return float64(capacity-avail) / float64(capacity)
}

// Accept will block until a token becomes available
func (t *tokenBucketRateLimiter) Accept() {
	t.limiter.Wait(1)
}

func (t *tokenBucketRateLimiter) AcceptContext(ctx context.Context) error {
	return t.limiter.WaitContext(ctx, 1)
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

func (t *fakeAlwaysRateLimiter) Saturation() float64 {
	return 0
}

func (t *fakeAlwaysRateLimiter) Stop() {}

func (t *fakeAlwaysRateLimiter) Accept() {}

func (t *fakeAlwaysRateLimiter) AcceptContext(_ context.Context) error {
	return nil
}

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

func (t *fakeNeverRateLimiter) Saturation() float64 {
	return 1
}

func (t *fakeNeverRateLimiter) Stop() {
	t.wg.Done()
}

func (t *fakeNeverRateLimiter) Accept() {
	t.wg.Wait()
}

func (t *fakeNeverRateLimiter) AcceptContext(ctx context.Context) error {
	<-ctx.Done()
	return ctx.Err()
}

func (t *fakeNeverRateLimiter) QPS() float32 {
	return 1
}
