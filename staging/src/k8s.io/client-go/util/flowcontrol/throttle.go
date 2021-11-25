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
	"errors"
	"sync"
	"time"

	"golang.org/x/time/rate"
	"k8s.io/utils/clock"
)

type PassiveRateLimiter interface {
	// TryAccept returns true if a token is taken immediately. Otherwise,
	// it returns false.
	TryAccept() bool
	// Stop stops the rate limiter, subsequent calls to CanAccept will return false
	Stop()
	// QPS returns QPS of this rate limiter
	QPS() float32
}

type RateLimiter interface {
	PassiveRateLimiter
	// Accept returns once a token becomes available.
	Accept()
	// Wait returns nil if a token is taken before the Context is done.
	Wait(ctx context.Context) error
}

type tokenBucketPassiveRateLimiter struct {
	limiter *rate.Limiter
	qps     float32
	clock   clock.PassiveClock
}

type tokenBucketRateLimiter struct {
	tokenBucketPassiveRateLimiter
	clock Clock
}

// NewTokenBucketRateLimiter creates a rate limiter which implements a token bucket approach.
// The rate limiter allows bursts of up to 'burst' to exceed the QPS, while still maintaining a
// smoothed qps rate of 'qps'.
// The bucket is initially filled with 'burst' tokens, and refills at a rate of 'qps'.
// The maximum number of tokens in the bucket is capped at 'burst'.
func NewTokenBucketRateLimiter(qps float32, burst int) RateLimiter {
	limiter := rate.NewLimiter(rate.Limit(qps), burst)
	return newTokenBucketRateLimiterWithClock(limiter, clock.RealClock{}, qps)
}

// NewTokenBucketPassiveRateLimiter is similar to NewTokenBucketRateLimiter except that it returns
// a PassiveRateLimiter which does not have Accept() and Wait() methods.
func NewTokenBucketPassiveRateLimiter(qps float32, burst int) PassiveRateLimiter {
	limiter := rate.NewLimiter(rate.Limit(qps), burst)
	return newTokenBucketRateLimiterWithPassiveClock(limiter, clock.RealClock{}, qps)
}

// An injectable, mockable clock interface.
type Clock interface {
	clock.PassiveClock
	Sleep(time.Duration)
}

var _ Clock = (*clock.RealClock)(nil)

// NewTokenBucketRateLimiterWithClock is identical to NewTokenBucketRateLimiter
// but allows an injectable clock, for testing.
func NewTokenBucketRateLimiterWithClock(qps float32, burst int, c Clock) RateLimiter {
	limiter := rate.NewLimiter(rate.Limit(qps), burst)
	return newTokenBucketRateLimiterWithClock(limiter, c, qps)
}

// NewTokenBucketPassiveRateLimiterWithClock is similar to NewTokenBucketRateLimiterWithClock
// except that it returns a PassiveRateLimiter which does not have Accept() and Wait() methods
// and uses a PassiveClock.
func NewTokenBucketPassiveRateLimiterWithClock(qps float32, burst int, c clock.PassiveClock) PassiveRateLimiter {
	limiter := rate.NewLimiter(rate.Limit(qps), burst)
	return newTokenBucketRateLimiterWithPassiveClock(limiter, c, qps)
}

func newTokenBucketRateLimiterWithClock(limiter *rate.Limiter, c Clock, qps float32) *tokenBucketRateLimiter {
	return &tokenBucketRateLimiter{
		tokenBucketPassiveRateLimiter: *newTokenBucketRateLimiterWithPassiveClock(limiter, c, qps),
		clock:                         c,
	}
}

func newTokenBucketRateLimiterWithPassiveClock(limiter *rate.Limiter, c clock.PassiveClock, qps float32) *tokenBucketPassiveRateLimiter {
	return &tokenBucketPassiveRateLimiter{
		limiter: limiter,
		qps:     qps,
		clock:   c,
	}
}

func (tbprl *tokenBucketPassiveRateLimiter) Stop() {
}

func (tbprl *tokenBucketPassiveRateLimiter) QPS() float32 {
	return tbprl.qps
}

func (tbprl *tokenBucketPassiveRateLimiter) TryAccept() bool {
	return tbprl.limiter.AllowN(tbprl.clock.Now(), 1)
}

// Accept will block until a token becomes available
func (tbrl *tokenBucketRateLimiter) Accept() {
	now := tbrl.clock.Now()
	tbrl.clock.Sleep(tbrl.limiter.ReserveN(now, 1).DelayFrom(now))
}

func (tbrl *tokenBucketRateLimiter) Wait(ctx context.Context) error {
	return tbrl.limiter.Wait(ctx)
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

func (t *fakeAlwaysRateLimiter) Wait(ctx context.Context) error {
	return nil
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

func (t *fakeNeverRateLimiter) Wait(ctx context.Context) error {
	return errors.New("can not be accept")
}

var (
	_ RateLimiter = (*tokenBucketRateLimiter)(nil)
	_ RateLimiter = (*fakeAlwaysRateLimiter)(nil)
	_ RateLimiter = (*fakeNeverRateLimiter)(nil)
)

var _ PassiveRateLimiter = (*tokenBucketPassiveRateLimiter)(nil)
