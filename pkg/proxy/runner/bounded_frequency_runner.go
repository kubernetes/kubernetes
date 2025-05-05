/*
Copyright 2017 The Kubernetes Authors.

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

package runner

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/client-go/util/flowcontrol"

	"k8s.io/klog/v2"
)

// BoundedFrequencyRunner manages runs of a user-provided work function.
type BoundedFrequencyRunner struct {
	name        string        // the name of this instance
	minInterval time.Duration // the min time between runs, modulo bursts
	maxInterval time.Duration // the max time between runs

	run chan struct{} // try an async run

	mu      sync.Mutex  // guards runs of fn and all mutations
	fn      func()      // the work function
	lastRun time.Time   // time of last run
	timer   timer       // timer for deferred runs
	limiter rateLimiter // rate limiter for on-demand runs

	retry     chan struct{} // schedule a retry
	retryMu   sync.Mutex    // guards retryTime
	retryTime time.Time     // when to retry
}

// designed so that flowcontrol.RateLimiter satisfies
type rateLimiter interface {
	TryAccept() bool
	Stop()
}

type nullLimiter struct{}

func (nullLimiter) TryAccept() bool {
	return true
}

func (nullLimiter) Stop() {}

var _ rateLimiter = nullLimiter{}

// for testing
type timer interface {
	// C returns the timer's selectable channel.
	C() <-chan time.Time

	// See time.Timer.Reset.
	Reset(d time.Duration) bool

	// See time.Timer.Stop.
	Stop() bool

	// See time.Now.
	Now() time.Time

	// Remaining returns the time until the timer will go off (if it is running).
	Remaining() time.Duration

	// See time.Since.
	Since(t time.Time) time.Duration

	// See time.Sleep.
	Sleep(d time.Duration)
}

// implement our timer in terms of std time.Timer.
type realTimer struct {
	timer *time.Timer
	next  time.Time
}

func (rt *realTimer) C() <-chan time.Time {
	return rt.timer.C
}

func (rt *realTimer) Reset(d time.Duration) bool {
	rt.next = time.Now().Add(d)
	return rt.timer.Reset(d)
}

func (rt *realTimer) Stop() bool {
	return rt.timer.Stop()
}

func (rt *realTimer) Now() time.Time {
	return time.Now()
}

func (rt *realTimer) Remaining() time.Duration {
	return rt.next.Sub(time.Now())
}

func (rt *realTimer) Since(t time.Time) time.Duration {
	return time.Since(t)
}

func (rt *realTimer) Sleep(d time.Duration) {
	time.Sleep(d)
}

var _ timer = &realTimer{}

// NewBoundedFrequencyRunner creates and returns a new BoundedFrequencyRunner.
// This runner manages the execution frequency of the provided work function `fn`.
//
// All runs will be async to the caller of BoundedFrequencyRunner.Run, but
// multiple runs are serialized. If the function needs to hold locks, it must
// take them internally.
//
// The runner guarantees two properties:
//  1. Minimum Interval (`minInterval`): At least `minInterval` must pass between
//     the *completion* of one execution and the *start* of the next. Calls to
//     `Run()` during this cooldown period are coalesced and deferred until the
//     interval expires. This prevents burst executions.
//  2. Maximum Interval (`maxInterval`): The function `fn` is guaranteed to run
//     at least once per `maxInterval`, ensuring periodic execution even without
//     explicit `Run()` calls (e.g., for refreshing state).
//
// `maxInterval` must be greater than or equal to `minInterval`; otherwise,
// this function will panic.
func NewBoundedFrequencyRunner(name string, fn func(), minInterval, maxInterval time.Duration, burstRuns int) *BoundedFrequencyRunner {
	timer := &realTimer{timer: time.NewTimer(0)} // will tick immediately
	<-timer.C()                                  // consume the first tick
	return construct(name, fn, minInterval, maxInterval, burstRuns, timer)
}

// Make an instance with dependencies injected.
func construct(name string, fn func(), minInterval, maxInterval time.Duration, burstRuns int, timer timer) *BoundedFrequencyRunner {
	if maxInterval < minInterval {
		panic(fmt.Sprintf("%s: maxInterval (%v) must be >= minInterval (%v)", name, maxInterval, minInterval))
	}

	bfr := &BoundedFrequencyRunner{
		name:        name,
		fn:          fn,
		minInterval: minInterval,
		maxInterval: maxInterval,
		run:         make(chan struct{}, 1),
		retry:       make(chan struct{}, 1),
		timer:       timer,
	}
	if minInterval == 0 {
		bfr.limiter = nullLimiter{}
	} else {
		// allow burst updates in short succession
		qps := float32(time.Second) / float32(minInterval)
		bfr.limiter = flowcontrol.NewTokenBucketRateLimiterWithClock(qps, burstRuns, timer)
	}
	return bfr
}

// Loop handles the periodic timer and run requests.  This is expected to be
// called as a goroutine.
func (bfr *BoundedFrequencyRunner) Loop(stop <-chan struct{}) {
	klog.V(3).InfoS("Loop running", "runner", bfr.name)
	bfr.timer.Reset(bfr.maxInterval)
	for {
		select {
		case <-stop:
			bfr.stop()
			klog.V(3).InfoS("Loop stopping", "runner", bfr.name)
			return
		case <-bfr.timer.C():
			bfr.tryRun()
		case <-bfr.run:
			bfr.tryRun()
		case <-bfr.retry:
			bfr.doRetry()
		}
	}
}

// Run the work function as soon as possible.  If this is called while Loop is not
// running, the call may be deferred indefinitely.
// Once there is a queued request to call the work function, further calls to
// Run() will have no effect until after it runs.
func (bfr *BoundedFrequencyRunner) Run() {
	// If bfr.run is empty, push an element onto it. Otherwise, do nothing.
	select {
	case bfr.run <- struct{}{}:
	default:
	}
}

// RetryAfter ensures that the function will run again after no later than interval. This
// can be called from inside a run of the BoundedFrequencyRunner's function, or
// asynchronously.
func (bfr *BoundedFrequencyRunner) RetryAfter(interval time.Duration) {
	// This could be called either with or without bfr.mu held, so we can't grab that
	// lock, and therefore we can't update the timer directly.

	// If the Loop thread is currently running fn then it may be a while before it
	// processes our retry request. But we want to retry at interval from now, not at
	// interval from "whenever doRetry eventually gets called". So we convert to
	// absolute time.
	retryTime := bfr.timer.Now().Add(interval)

	// We can't just write retryTime to a channel because there could be multiple
	// RetryAfter calls before Loop gets a chance to read from the channel. So we
	// record the soonest requested retry time in bfr.retryTime and then only signal
	// the Loop thread once, just like Run does.
	bfr.retryMu.Lock()
	defer bfr.retryMu.Unlock()
	if !bfr.retryTime.IsZero() && bfr.retryTime.Before(retryTime) {
		return
	}
	bfr.retryTime = retryTime

	select {
	case bfr.retry <- struct{}{}:
	default:
	}
}

// assumes the lock is not held
func (bfr *BoundedFrequencyRunner) stop() {
	bfr.mu.Lock()
	defer bfr.mu.Unlock()
	bfr.limiter.Stop()
	bfr.timer.Stop()
}

// assumes the lock is not held
func (bfr *BoundedFrequencyRunner) doRetry() {
	bfr.mu.Lock()
	defer bfr.mu.Unlock()
	bfr.retryMu.Lock()
	defer bfr.retryMu.Unlock()

	if bfr.retryTime.IsZero() {
		return
	}

	// Timer wants an interval not an absolute time, so convert retryTime back now
	retryInterval := bfr.retryTime.Sub(bfr.timer.Now())
	bfr.retryTime = time.Time{}
	if retryInterval < bfr.timer.Remaining() {
		klog.V(3).InfoS("retrying", "runner", bfr.name, "interval", retryInterval)
		bfr.timer.Stop()
		bfr.timer.Reset(retryInterval)
	}
}

// assumes the lock is not held
func (bfr *BoundedFrequencyRunner) tryRun() {
	bfr.mu.Lock()
	defer bfr.mu.Unlock()

	if bfr.limiter.TryAccept() {
		// We're allowed to run the function right now.
		bfr.fn()
		bfr.lastRun = bfr.timer.Now()
		bfr.timer.Stop()
		bfr.timer.Reset(bfr.maxInterval)
		klog.V(3).InfoS("ran", "runner", bfr.name, "minInterval", bfr.minInterval, "maxInternval", bfr.maxInterval)
		return
	}

	// It can't run right now, figure out when it can run next.
	elapsed := bfr.timer.Since(bfr.lastRun)   // how long since last run
	nextPossible := bfr.minInterval - elapsed // time to next possible run
	nextScheduled := bfr.timer.Remaining()    // time to next scheduled run
	klog.V(4).InfoS("can't run", "runner", bfr.name, "elapsed", elapsed, "nextPossible", nextPossible, "nextScheduled", nextScheduled)

	// It's hard to avoid race conditions in the unit tests unless we always reset
	// the timer here, even when it's unchanged
	if nextPossible < nextScheduled {
		nextScheduled = nextPossible
	}
	bfr.timer.Stop()
	bfr.timer.Reset(nextScheduled)
}
