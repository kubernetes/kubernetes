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
	name string // the name of this instance

	minInterval   time.Duration // the min time between runs
	retryInterval time.Duration // the time between a run and a retry
	maxInterval   time.Duration // the max time between runs

	run chan struct{} // try an async run

	mu      sync.Mutex   // guards runs of fn and all mutations
	fn      func() error // the work function
	lastRun time.Time    // time of last run
	timer   timer        // timer for deferred runs
	limiter rateLimiter  // rate limiter for on-demand runs
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
//
// If `fn` returns an error, then it will be run again no later than `retryInterval`
// (unless another trigger, like `Run()` or `maxInterval`, causes it to run sooner). Any
// successful run will abort the retry attempt.
func NewBoundedFrequencyRunner(name string, fn func() error, minInterval, retryInterval, maxInterval time.Duration) *BoundedFrequencyRunner {
	timer := &realTimer{timer: time.NewTimer(0)} // will tick immediately
	<-timer.C()                                  // consume the first tick
	return construct(name, fn, minInterval, retryInterval, maxInterval, timer)
}

// Make an instance with dependencies injected.
func construct(name string, fn func() error, minInterval, retryInterval, maxInterval time.Duration, timer timer) *BoundedFrequencyRunner {
	if maxInterval < minInterval {
		panic(fmt.Sprintf("%s: maxInterval (%v) must be >= minInterval (%v)", name, maxInterval, minInterval))
	}

	bfr := &BoundedFrequencyRunner{
		name: name,
		fn:   fn,

		minInterval:   minInterval,
		retryInterval: retryInterval,
		maxInterval:   maxInterval,

		run:   make(chan struct{}, 1),
		timer: timer,
	}
	if minInterval == 0 {
		bfr.limiter = nullLimiter{}
	} else {
		qps := float32(time.Second) / float32(minInterval)
		bfr.limiter = flowcontrol.NewTokenBucketRateLimiterWithClock(qps, 1, timer)
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

// assumes the lock is not held
func (bfr *BoundedFrequencyRunner) stop() {
	bfr.mu.Lock()
	defer bfr.mu.Unlock()
	bfr.limiter.Stop()
	bfr.timer.Stop()
}

// assumes the lock is not held
func (bfr *BoundedFrequencyRunner) tryRun() {
	bfr.mu.Lock()
	defer bfr.mu.Unlock()

	if bfr.limiter.TryAccept() {
		// We're allowed to run the function right now.
		err := bfr.fn()

		bfr.lastRun = bfr.timer.Now()
		bfr.timer.Stop()

		nextInterval := bfr.maxInterval
		if err != nil {
			// an error will schedule a retry after the retryInterval,
			// any successful run before that will stop the retry attempt.
			nextInterval = bfr.retryInterval
			klog.V(3).InfoS("scheduling retry", "runner", bfr.name, "interval", nextInterval, "error", err)
		}
		bfr.timer.Reset(nextInterval)
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
