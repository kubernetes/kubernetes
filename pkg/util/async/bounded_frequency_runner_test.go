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

package async

import (
	"sync"
	"testing"
	"time"
)

// Track calls to the managed function.
type receiver struct {
	lock sync.Mutex
	run  bool
}

func (r *receiver) F() {
	r.lock.Lock()
	defer r.lock.Unlock()
	r.run = true
}

func (r *receiver) reset() bool {
	r.lock.Lock()
	defer r.lock.Unlock()
	was := r.run
	r.run = false
	return was
}

// A single change event in the fake timer.
type timerUpdate struct {
	active bool
	next   time.Duration // iff active == true
}

// Fake time.
type fakeTimer struct {
	c chan time.Time

	lock   sync.Mutex
	now    time.Time
	active bool

	updated chan timerUpdate
}

func newFakeTimer() *fakeTimer {
	ft := &fakeTimer{
		c:       make(chan time.Time),
		updated: make(chan timerUpdate),
	}
	return ft
}

func (ft *fakeTimer) C() <-chan time.Time {
	return ft.c
}

func (ft *fakeTimer) Reset(in time.Duration) bool {
	ft.lock.Lock()
	defer ft.lock.Unlock()

	was := ft.active
	ft.active = true
	ft.updated <- timerUpdate{
		active: true,
		next:   in,
	}
	return was
}

func (ft *fakeTimer) Stop() bool {
	ft.lock.Lock()
	defer ft.lock.Unlock()

	was := ft.active
	ft.active = false
	ft.updated <- timerUpdate{
		active: false,
	}
	return was
}

func (ft *fakeTimer) Now() time.Time {
	ft.lock.Lock()
	defer ft.lock.Unlock()

	return ft.now
}

func (ft *fakeTimer) Since(t time.Time) time.Duration {
	ft.lock.Lock()
	defer ft.lock.Unlock()

	return ft.now.Sub(t)
}

func (ft *fakeTimer) Sleep(d time.Duration) {
	ft.lock.Lock()
	defer ft.lock.Unlock()

	ft.advance(d)
}

// advance the current time.
func (ft *fakeTimer) advance(d time.Duration) {
	ft.lock.Lock()
	defer ft.lock.Unlock()

	ft.now = ft.now.Add(d)
}

// send a timer tick.
func (ft *fakeTimer) tick() {
	ft.lock.Lock()
	defer ft.lock.Unlock()

	ft.active = false
	ft.c <- ft.now
}

// return the calling line number (for printing)
// test the timer's state
func checkTimer(name string, t *testing.T, upd timerUpdate, active bool, next time.Duration) {
	if upd.active != active {
		t.Fatalf("%s: expected timer active=%v", name, active)
	}
	if active && upd.next != next {
		t.Fatalf("%s: expected timer to be %v, got %v", name, next, upd.next)
	}
}

// test and reset the receiver's state
func checkReceiver(name string, t *testing.T, receiver *receiver, expected bool) {
	triggered := receiver.reset()
	if expected && !triggered {
		t.Fatalf("%s: function should have been called", name)
	} else if !expected && triggered {
		t.Fatalf("%s: function should not have been called", name)
	}
}

// Durations embedded in test cases depend on these.
var minInterval = 1 * time.Second
var maxInterval = 10 * time.Second

func waitForReset(name string, t *testing.T, timer *fakeTimer, obj *receiver, expectCall bool, expectNext time.Duration) {
	upd := <-timer.updated // wait for stop
	checkReceiver(name, t, obj, expectCall)
	checkReceiver(name, t, obj, false) // prove post-condition
	checkTimer(name, t, upd, false, 0)
	upd = <-timer.updated // wait for reset
	checkTimer(name, t, upd, true, expectNext)
}

func waitForRun(name string, t *testing.T, timer *fakeTimer, obj *receiver) {
	waitForReset(name, t, timer, obj, true, maxInterval)
}

func waitForDefer(name string, t *testing.T, timer *fakeTimer, obj *receiver, expectNext time.Duration) {
	waitForReset(name, t, timer, obj, false, expectNext)
}

func Test_BoundedFrequencyRunnerNoBurst(t *testing.T) {
	obj := &receiver{}
	timer := newFakeTimer()
	runner := construct("test-runner", obj.F, minInterval, maxInterval, 1, timer)
	stop := make(chan struct{})

	var upd timerUpdate

	// Start.
	go runner.Loop(stop)
	upd = <-timer.updated // wait for initial time to be set to max
	checkTimer("init", t, upd, true, maxInterval)
	checkReceiver("init", t, obj, false)

	// Run once, immediately.
	// rel=0ms
	runner.Run()
	waitForRun("first run", t, timer, obj)

	// Run again, before minInterval expires.
	timer.advance(500 * time.Millisecond) // rel=500ms
	runner.Run()
	waitForDefer("too soon after first", t, timer, obj, 500*time.Millisecond)

	// Run again, before minInterval expires.
	timer.advance(499 * time.Millisecond) // rel=999ms
	runner.Run()
	waitForDefer("still too soon after first", t, timer, obj, 1*time.Millisecond)

	// Run again, once minInterval has passed (race with timer).
	timer.advance(1 * time.Millisecond) // rel=1000ms
	runner.Run()
	waitForRun("second run", t, timer, obj)

	// Run again, before minInterval expires.
	// rel=0ms
	runner.Run()
	waitForDefer("too soon after second", t, timer, obj, 1*time.Second)

	// Run again, before minInterval expires.
	timer.advance(1 * time.Millisecond) // rel=1ms
	runner.Run()
	waitForDefer("still too soon after second", t, timer, obj, 999*time.Millisecond)

	// Let the timer tick prematurely.
	timer.advance(998 * time.Millisecond) // rel=999ms
	timer.tick()
	waitForDefer("premature tick", t, timer, obj, 1*time.Millisecond)

	// Let the timer tick.
	timer.advance(1 * time.Millisecond) // rel=1000ms
	timer.tick()
	waitForRun("first tick", t, timer, obj)

	// Let the timer tick.
	timer.advance(10 * time.Second) // rel=10000ms
	timer.tick()
	waitForRun("second tick", t, timer, obj)

	// Run again, before minInterval expires.
	timer.advance(1 * time.Millisecond) // rel=1ms
	runner.Run()
	waitForDefer("too soon after tick", t, timer, obj, 999*time.Millisecond)

	// Let the timer tick.
	timer.advance(999 * time.Millisecond) // rel=1000ms
	timer.tick()
	waitForRun("third tick", t, timer, obj)

	// Clean up.
	stop <- struct{}{}
}

func Test_BoundedFrequencyRunnerBurst(t *testing.T) {
	obj := &receiver{}
	timer := newFakeTimer()
	runner := construct("test-runner", obj.F, minInterval, maxInterval, 2, timer)
	stop := make(chan struct{})

	var upd timerUpdate

	// Start.
	go runner.Loop(stop)
	upd = <-timer.updated // wait for initial time to be set to max
	checkTimer("init", t, upd, true, maxInterval)
	checkReceiver("init", t, obj, false)

	// Run once, immediately.
	// abs=0ms, rel=0ms
	runner.Run()
	waitForRun("first run", t, timer, obj)

	// Run again, before minInterval expires, with burst.
	timer.advance(1 * time.Millisecond) // abs=1ms, rel=1ms
	runner.Run()
	waitForRun("second run", t, timer, obj)

	// Run again, before minInterval expires.
	timer.advance(498 * time.Millisecond) // abs=499ms, rel=498ms
	runner.Run()
	waitForDefer("too soon after second", t, timer, obj, 502*time.Millisecond)

	// Run again, before minInterval expires.
	timer.advance(1 * time.Millisecond) // abs=500ms, rel=499ms
	runner.Run()
	waitForDefer("too soon after second 2", t, timer, obj, 501*time.Millisecond)

	// Run again, before minInterval expires.
	timer.advance(1 * time.Millisecond) // abs=501ms, rel=500ms
	runner.Run()
	waitForDefer("too soon after second 3", t, timer, obj, 500*time.Millisecond)

	// Run again, once burst has replenished.
	timer.advance(499 * time.Millisecond) // abs=1000ms, rel=999ms
	runner.Run()
	waitForRun("third run", t, timer, obj)

	// Run again, before minInterval expires.
	timer.advance(1 * time.Millisecond) // abs=1001ms, rel=1ms
	runner.Run()
	waitForDefer("too soon after third", t, timer, obj, 999*time.Millisecond)

	// Run again, before minInterval expires.
	timer.advance(998 * time.Millisecond) // abs=1999ms, rel=999ms
	runner.Run()
	waitForDefer("too soon after third 2", t, timer, obj, 1*time.Millisecond)

	// Run again, once burst has replenished.
	timer.advance(1 * time.Millisecond) // abs=2000ms, rel=1000ms
	runner.Run()
	waitForRun("fourth run", t, timer, obj)

	// Run again, once burst has fully replenished.
	timer.advance(2 * time.Second) // abs=4000ms, rel=2000ms
	runner.Run()
	waitForRun("fifth run", t, timer, obj)
	runner.Run()
	waitForRun("sixth run", t, timer, obj)
	runner.Run()
	waitForDefer("too soon after sixth", t, timer, obj, 1*time.Second)

	// Let the timer tick.
	timer.advance(1 * time.Second) // abs=5000ms, rel=1000ms
	timer.tick()
	waitForRun("first tick", t, timer, obj)

	// Let the timer tick.
	timer.advance(10 * time.Second) // abs=15000ms, rel=10000ms
	timer.tick()
	waitForRun("second tick", t, timer, obj)

	// Clean up.
	stop <- struct{}{}
}
