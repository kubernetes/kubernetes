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
	"time"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

// BoundedFrequencyRunner manages runs of a user-provided function.
// See NewBoundedFrequencyRunner for examples.
type BoundedFrequencyRunner struct {
	name          string        // the name of this instance
	minInterval   time.Duration // the min time between runs, modulo bursts
	retryInterval time.Duration
	maxInterval   time.Duration // the max time between runs

	run chan struct{} // try an async run

	fn               func() error // function to run
	minIntervalTimer clock.Timer
	nextRunTimer     clock.Timer // Combined timer for maxInterval and retryInterval logic
	clock            clock.Clock

	ready chan struct{} // signals the loop is ready
}

// NewBoundedFrequencyRunner creates and returns a new BoundedFrequencyRunner.
// This runner manages the execution frequency of the provided function `fn`.
//
// The runner guarantees three properties:
//  1. Minimum Interval (`minInterval`): At least `minInterval` must pass between
//     the *completion* of one execution and the *start* of the next. Calls to
//     `Run()` during this cooldown period are coalesced and deferred until the
//     interval expires. This prevents burst executions.
//  2. Maximum Interval (`maxInterval`): The function `fn` is guaranteed to run
//     at least once per `maxInterval`, ensuring periodic execution even without
//     explicit `Run()` calls (e.g., for refreshing state).
//  3. Retry Interval (`retryInterval`): Any error returned by `fn`
//     will run again no later than `retryInterval`, unless another trigger (like
//     `Run()` or `maxInterval`) causes it to run sooner. Any successful run will
//     abort the retry attempt.
//
// After `fn` completes, the loop waits for the `minInterval` timer to expire
// before it becomes eligible to execute `fn` again. This ensures the minimum
// time between executions.
//
// `maxInterval` must be greater than or equal to `minInterval`; otherwise,
// this function will panic.
func NewBoundedFrequencyRunner(name string, fn func() error, minInterval, retryInterval, maxInterval time.Duration) *BoundedFrequencyRunner {
	return construct(name, fn, minInterval, retryInterval, maxInterval, clock.RealClock{})
}

// Make an instance with dependencies injected.
func construct(name string, fn func() error, minInterval, retryInterval, maxInterval time.Duration, clock clock.Clock) *BoundedFrequencyRunner {
	if maxInterval < minInterval {
		panic(fmt.Sprintf("%s: maxInterval (%v) must be >= minInterval (%v)", name, maxInterval, minInterval))
	}
	if clock == nil {
		panic(fmt.Sprintf("%s: clock must be non-nil", name))
	}

	bfr := &BoundedFrequencyRunner{
		name:          name,
		fn:            fn,
		minInterval:   minInterval,
		retryInterval: retryInterval,
		maxInterval:   maxInterval,
		run:           make(chan struct{}, 1),
		ready:         make(chan struct{}), // used to synchronize tests
		clock:         clock,
	}

	return bfr
}

// Loop handles the periodic timer and run requests.  This is expected to be
// called as a goroutine.
func (bfr *BoundedFrequencyRunner) Loop(stop <-chan struct{}) {
	klog.V(3).InfoS("Loop running", "name", bfr.name)
	defer close(bfr.run)

	bfr.minIntervalTimer = bfr.clock.NewTimer(bfr.minInterval)
	defer bfr.minIntervalTimer.Stop()

	// Initialize nextRunTimer with maxInterval
	bfr.nextRunTimer = bfr.clock.NewTimer(bfr.maxInterval)
	defer bfr.nextRunTimer.Stop()

	// Signal the loop is ready
	close(bfr.ready)

	for {
		select {
		case <-stop:
			klog.V(3).InfoS("Loop stopping", "name", bfr.name)
			return
		case <-bfr.nextRunTimer.C(): // Wait on the single timer
		case <-bfr.run:
		}

		func() {
			// stop the timers here to allow the tests using the fake clock to synchronize
			// with the fakeClock.HasWaiters() method. The timers are reset after the function
			// is executed.
			bfr.minIntervalTimer.Stop()
			bfr.nextRunTimer.Stop() // Stop the single timer before running fn

			// avoid crashing if the function executed crashes
			defer utilruntime.HandleCrash()
			err := bfr.fn()

			// Determine the next interval based on the result
			nextInterval := bfr.maxInterval
			if err != nil {
				// If error, ensure next run is within retryInterval and maxInterval
				if bfr.retryInterval < nextInterval {
					nextInterval = bfr.retryInterval
				}
				klog.V(3).InfoS("scheduling retry", "name", bfr.name, "interval", nextInterval, "error", err)
			}
			// Reset the timers
			bfr.minIntervalTimer.Reset(bfr.minInterval)
			bfr.nextRunTimer.Reset(nextInterval) // Reset with the calculated interval
		}()

		select {
		case <-stop:
			klog.V(3).InfoS("Loop stopping", "name", bfr.name)
			return
		case <-bfr.minIntervalTimer.C():
		}
	}
}

// Run the function as soon as possible.  If this is called while Loop is not
// running, the call may be deferred indefinitely.
// If there is already a queued request to call the underlying function, it
// may be dropped - it is just guaranteed that we will try calling the
// underlying function as soon as possible starting from now.
func (bfr *BoundedFrequencyRunner) Run() {
	// Wait until the loop is ready
	select {
	case <-bfr.ready:
	default:
		return
	}
	// It attempts to send a signal to the internal queue. If the queue is already
	// full (meaning an execution is already pending or waiting for the minInterval),
	// this call is effectively coalesced with the previous one, and no new signal
	// is added. This ensures that rapid calls to Run() don't queue up indefinitely.
	select {
	case bfr.run <- struct{}{}:
	default:
	}
}
