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
	"sync/atomic"
	"testing"
	"time"

	clock "k8s.io/utils/clock/testing"
)

// Track calls to the managed function.
type receiver struct {
	counter atomic.Int32
	// counterCh signals completion of F() and sends the new count.
	// It's unbuffered to make the send in F() blocking.
	counterCh chan int
	resultMu  sync.RWMutex
	result    error
}

func (r *receiver) F() error {
	newCount := r.counter.Add(1)
	// Blocking send: F() will wait here until the test reads from counterCh.
	r.counterCh <- int(newCount)
	r.resultMu.RLock()
	defer r.resultMu.RUnlock()
	return r.result
}

func newReceiver() *receiver {
	return &receiver{
		counterCh: make(chan int),
	}
}

func (r *receiver) calls() <-chan int {
	return r.counterCh
}

func (r *receiver) setReturnValue(err error) {
	r.resultMu.Lock()
	defer r.resultMu.Unlock()
	r.result = err
}

// assertCalls waits for the receiver's function to be called and asserts that
// the total call count matches expectedCalls. It fails the test if the timeout is reached
// or if the call count doesn't match.
func assertCalls(t *testing.T, r *receiver, expectedCalls int) {
	t.Helper()
	select {
	case calls := <-r.calls():
		if calls != expectedCalls {
			t.Fatalf("expected %d calls, but got %d", expectedCalls, calls)
		}
	case <-time.After(1 * time.Second):
		t.Fatalf("timed out waiting for function execution (expected %d calls, got %d)", expectedCalls, r.counter.Load())
	}
}

// assertNoCalls waits for 100 millisecond and asserts that the receiver's
// function was *not* called during that time. It fails the test if a call is detected.
func assertNoCalls(t *testing.T, r *receiver) {
	t.Helper()
	select {
	case calls := <-r.calls():
		t.Fatalf("unexpected function execution detected (call count: %d)", calls)
	case <-time.After(100 * time.Millisecond):
	}
}

func Test_BoundedFrequencyRunner(t *testing.T) {
	var minInterval = 1 * time.Second
	var retryInterval = 5 * time.Second
	var maxInterval = 10 * time.Second
	obj := newReceiver()
	fakeClock := clock.NewFakeClock(time.Now())
	runner := construct("test-runner", obj.F, minInterval, retryInterval, maxInterval, fakeClock)
	stop := make(chan struct{})
	defer close(stop)

	go runner.Loop(stop)

	// Run once, immediately.
	// rel=0ms
	runner.Run()
	assertCalls(t, obj, 1)
	// wait for the timers to be reset
	for fakeClock.Waiters() != 2 {
		time.Sleep(1 * time.Millisecond)
	}
	// Run again, before minInterval expires. No execution expected.
	fakeClock.Step(500 * time.Millisecond) // rel=500ms
	runner.Run()
	assertNoCalls(t, obj)

	// Run again, before minInterval expires. No execution expected.
	fakeClock.Step(499 * time.Millisecond) // rel=999ms
	runner.Run()
	assertNoCalls(t, obj)

	// Do the deferred run
	fakeClock.Step(1 * time.Millisecond) // rel=1000ms
	assertCalls(t, obj, 2)

	runner.Run()
	assertNoCalls(t, obj)

	// Run again, before minInterval expires. No execution expected.
	fakeClock.Step(1 * time.Millisecond) // rel=1ms
	runner.Run()
	assertNoCalls(t, obj)

	// Ensure that we don't run again early. No execution expected.
	fakeClock.Step(998 * time.Millisecond) // rel=999ms
	assertNoCalls(t, obj)

	// Do the deferred run
	fakeClock.Step(1 * time.Millisecond) // rel=1000ms
	assertCalls(t, obj, 3)
	// wait for the timers to be reset
	for fakeClock.Waiters() != 2 {
		time.Sleep(1 * time.Millisecond)
	}
	// Let minInterval pass, but there are no runs queued. No execution expected.
	fakeClock.Step(1 * time.Second) // rel=1000ms
	assertNoCalls(t, obj)

	// Let maxInterval pass
	fakeClock.Step(maxInterval) // rel=10000ms
	assertCalls(t, obj, 4)
	// wait for the timers to be reset
	for fakeClock.Waiters() != 2 {
		time.Sleep(1 * time.Millisecond)
	}
	// Run again, before minInterval expires. No execution expected.
	fakeClock.Step(1 * time.Millisecond) // rel=1ms
	runner.Run()
	assertNoCalls(t, obj)

	// Let minInterval pass
	fakeClock.Step(999 * time.Millisecond) // rel=1000ms
	assertCalls(t, obj, 5)
}

func Test_BoundedFrequencyRunnerRetry(t *testing.T) {
	var minInterval = 1 * time.Second
	var retryInterval = 5 * time.Second
	var maxInterval = 10 * time.Second
	obj := newReceiver()
	fakeClock := clock.NewFakeClock(time.Now())
	runner := construct("test-runner", obj.F, minInterval, retryInterval, maxInterval, fakeClock)
	stop := make(chan struct{})
	defer close(stop)

	go runner.Loop(stop)

	// Run once, immediately, and queue a retry
	// rel=0ms
	obj.setReturnValue(fmt.Errorf("sync error"))
	runner.Run()
	assertCalls(t, obj, 1)
	// wait for the timers to be reset
	for fakeClock.Waiters() != 2 {
		time.Sleep(1 * time.Millisecond)
	}

	// next run will succeed
	obj.setReturnValue(nil)
	assertNoCalls(t, obj)

	// Nothing happens...
	fakeClock.Step(minInterval) // rel=1000ms
	assertNoCalls(t, obj)

	// After retryInterval, function is called
	fakeClock.Step(4 * time.Second) // rel=5000ms
	assertCalls(t, obj, 2)
	// wait for the timers to be reset
	for fakeClock.Waiters() != 2 {
		time.Sleep(1 * time.Millisecond)
	}

	// Run again, before minInterval expires and trigger a retry
	fakeClock.Step(499 * time.Millisecond) // rel=499ms
	obj.setReturnValue(fmt.Errorf("sync error"))
	runner.Run()
	assertNoCalls(t, obj)

	// Do the deferred run, queue another retry after it returns
	fakeClock.Step(501 * time.Millisecond) // rel=1000ms
	assertCalls(t, obj, 3)

	// next run will succeed
	obj.setReturnValue(nil)
	assertNoCalls(t, obj)

	// Wait for minInterval to pass
	fakeClock.Step(time.Second) // rel=1000ms
	assertNoCalls(t, obj)

	// Now do another successful that abort the retry
	runner.Run()
	assertCalls(t, obj, 4)
	// wait for the timers to be reset
	for fakeClock.Waiters() != 2 {
		time.Sleep(1 * time.Millisecond)
	}

	// Retry was cancelled because we already ran
	fakeClock.Step(4 * time.Second)
	assertNoCalls(t, obj)

	// New run will trigger a retry.
	obj.setReturnValue(fmt.Errorf("sync error"))
	runner.Run()
	assertCalls(t, obj, 5)
	for fakeClock.Waiters() != 2 { // wait for retryIntervalTimer
		time.Sleep(1 * time.Millisecond)
	}

	// next run will succeed
	obj.setReturnValue(nil)
	assertNoCalls(t, obj)

	// Call Run again before minInterval passes
	fakeClock.Step(100 * time.Millisecond) // rel=100ms
	runner.Run()
	assertNoCalls(t, obj)

	// Deferred run will run after minInterval passes
	fakeClock.Step(900 * time.Millisecond) // rel=1000ms
	assertCalls(t, obj, 6)
	// wait for the timers to be reset
	for fakeClock.Waiters() != 2 {
		time.Sleep(1 * time.Millisecond)
	}

	// Retry was cancelled because we already ran
	fakeClock.Step(4 * time.Second) // rel=4s since run, 5s since RetryAfter
	assertNoCalls(t, obj)

	// Rerun happens after maxInterval
	fakeClock.Step(5 * time.Second) // rel=9s since run, 10s since RetryAfter
	assertNoCalls(t, obj)

	fakeClock.Step(time.Second) // rel=10s since run
	assertCalls(t, obj, 7)
}

func Test_BoundedFrequencyRunnerRetryShorterThanMinInterval(t *testing.T) {
	var minInterval = 5 * time.Second
	var retryInterval = 1 * time.Second // Shorter than minInterval
	var maxInterval = 10 * time.Second
	obj := newReceiver()
	fakeClock := clock.NewFakeClock(time.Now())
	runner := construct("test-runner-short-retry", obj.F, minInterval, retryInterval, maxInterval, fakeClock)
	stop := make(chan struct{})
	defer close(stop)

	go runner.Loop(stop)

	// Run once immediately and trigger a retry.
	// rel=0s
	obj.setReturnValue(fmt.Errorf("sync error"))
	runner.Run()
	assertCalls(t, obj, 1)
	for fakeClock.Waiters() != 2 {
		time.Sleep(1 * time.Millisecond)
	}

	// next run will succeed
	obj.setReturnValue(nil)
	assertNoCalls(t, obj)

	// Advance clock past retryInterval, but still within minInterval.
	// rel=1s
	fakeClock.Step(retryInterval)
	assertNoCalls(t, obj) // Still shouldn't run because minInterval hasn't passed since run 1 finished.

	// Advance clock just before minInterval expires.
	// rel=4.999s
	fakeClock.Step(minInterval - retryInterval - 1*time.Millisecond)
	assertNoCalls(t, obj)

	// Advance clock past minInterval. The retry should now trigger the run.
	// rel=5s
	fakeClock.Step(1 * time.Millisecond)
	assertCalls(t, obj, 2) // Run happens now, triggered by the earlier retry, respecting minInterval.
	// wait for the timers to be reset
	for fakeClock.Waiters() != 2 {
		time.Sleep(1 * time.Millisecond)
	}
	// Let maxInterval pass without any Run() or Retry() calls.
	fakeClock.Step(maxInterval) // rel=10s since run 2
	assertCalls(t, obj, 3)
}

func TestBoundedFrequencyRunner_Run_RunsAgainAfterMinInterval_RealClock(t *testing.T) {
	// Use relatively short intervals for real clock testing
	var minInterval = 500 * time.Millisecond
	var retryInterval = 800 * time.Millisecond
	var maxInterval = 1500 * time.Millisecond
	obj := newReceiver()
	runner := NewBoundedFrequencyRunner("test-runner", obj.F, minInterval, retryInterval, maxInterval)

	stopCh := make(chan struct{})
	defer close(stopCh)
	go runner.Loop(stopCh)

	runner.Run() // First run
	assertCalls(t, obj, 1)

	time.Sleep(2 * minInterval)
	assertNoCalls(t, obj)

	runner.Run() // Second run
	assertCalls(t, obj, 2)
}

func TestBoundedFrequencyRunner_Run_DoesNotRunBeforeMinInterval_RealClock(t *testing.T) {
	// Use relatively short intervals for real clock testing
	var minInterval = 500 * time.Millisecond
	var retryInterval = 800 * time.Millisecond
	var maxInterval = 1500 * time.Millisecond
	obj := newReceiver()
	runner := NewBoundedFrequencyRunner("test-runner", obj.F, minInterval, retryInterval, maxInterval)

	stopCh := make(chan struct{})
	defer close(stopCh)
	go runner.Loop(stopCh)

	runner.Run() // First run
	assertCalls(t, obj, 1)

	time.Sleep(minInterval / 4)
	runner.Run()
	assertNoCalls(t, obj)
}

func TestBoundedFrequencyRunner_RunAfterMaxInterval_RealClock(t *testing.T) {
	// Use relatively short intervals for real clock testing
	var minInterval = 100 * time.Millisecond
	var retryInterval = 200 * time.Millisecond
	var maxInterval = 500 * time.Millisecond
	obj := newReceiver()
	runner := NewBoundedFrequencyRunner("test-runner", obj.F, minInterval, retryInterval, maxInterval)

	stopCh := make(chan struct{})
	defer close(stopCh)
	go runner.Loop(stopCh)

	assertNoCalls(t, obj)

	time.Sleep(maxInterval)
	assertCalls(t, obj, 1)
}

func Test_BoundedFrequencyRunnerRetry_RealClock(t *testing.T) {
	// Use relatively short intervals for real clock testing
	var minInterval = 100 * time.Millisecond
	var retryInterval = 500 * time.Millisecond
	var maxInterval = 10 * time.Second

	obj := newReceiver()
	// Use the real clock constructor
	runner := NewBoundedFrequencyRunner("test-runner-real-clock", obj.F, minInterval, retryInterval, maxInterval)

	stopCh := make(chan struct{})
	defer close(stopCh)
	go runner.Loop(stopCh)

	t.Log("Triggering first retry")
	// Run once immediately and trigger a retry.
	// rel=0s
	obj.setReturnValue(fmt.Errorf("sync error"))
	runner.Run()
	assertCalls(t, obj, 1)

	// Check before retryInterval
	time.Sleep(retryInterval / 4)
	assertNoCalls(t, obj)

	// Check after retryInterval
	time.Sleep(retryInterval) // Wait past retryInterval
	assertCalls(t, obj, 2)

	// Check after retryInterval (relative to the *first* Retry call in this batch)
	time.Sleep(retryInterval)
	assertCalls(t, obj, 3)

	time.Sleep(retryInterval / 8)
	assertNoCalls(t, obj)

	time.Sleep(retryInterval) // Wait past the new retryInterval
	assertCalls(t, obj, 4)
}
