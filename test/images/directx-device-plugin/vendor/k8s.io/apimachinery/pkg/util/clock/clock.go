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

package clock

import (
	"sync"
	"time"
)

// Clock allows for injecting fake or real clocks into code that
// needs to do arbitrary things based on time.
type Clock interface {
	Now() time.Time
	Since(time.Time) time.Duration
	After(time.Duration) <-chan time.Time
	NewTimer(time.Duration) Timer
	Sleep(time.Duration)
	NewTicker(time.Duration) Ticker
}

// RealClock really calls time.Now()
type RealClock struct{}

// Now returns the current time.
func (RealClock) Now() time.Time {
	return time.Now()
}

// Since returns time since the specified timestamp.
func (RealClock) Since(ts time.Time) time.Duration {
	return time.Since(ts)
}

// Same as time.After(d).
func (RealClock) After(d time.Duration) <-chan time.Time {
	return time.After(d)
}

func (RealClock) NewTimer(d time.Duration) Timer {
	return &realTimer{
		timer: time.NewTimer(d),
	}
}

func (RealClock) NewTicker(d time.Duration) Ticker {
	return &realTicker{
		ticker: time.NewTicker(d),
	}
}

func (RealClock) Sleep(d time.Duration) {
	time.Sleep(d)
}

// FakeClock implements Clock, but returns an arbitrary time.
type FakeClock struct {
	lock sync.RWMutex
	time time.Time

	// waiters are waiting for the fake time to pass their specified time
	waiters []fakeClockWaiter
}

type fakeClockWaiter struct {
	targetTime    time.Time
	stepInterval  time.Duration
	skipIfBlocked bool
	destChan      chan time.Time
	fired         bool
}

func NewFakeClock(t time.Time) *FakeClock {
	return &FakeClock{
		time: t,
	}
}

// Now returns f's time.
func (f *FakeClock) Now() time.Time {
	f.lock.RLock()
	defer f.lock.RUnlock()
	return f.time
}

// Since returns time since the time in f.
func (f *FakeClock) Since(ts time.Time) time.Duration {
	f.lock.RLock()
	defer f.lock.RUnlock()
	return f.time.Sub(ts)
}

// Fake version of time.After(d).
func (f *FakeClock) After(d time.Duration) <-chan time.Time {
	f.lock.Lock()
	defer f.lock.Unlock()
	stopTime := f.time.Add(d)
	ch := make(chan time.Time, 1) // Don't block!
	f.waiters = append(f.waiters, fakeClockWaiter{
		targetTime: stopTime,
		destChan:   ch,
	})
	return ch
}

// Fake version of time.NewTimer(d).
func (f *FakeClock) NewTimer(d time.Duration) Timer {
	f.lock.Lock()
	defer f.lock.Unlock()
	stopTime := f.time.Add(d)
	ch := make(chan time.Time, 1) // Don't block!
	timer := &fakeTimer{
		fakeClock: f,
		waiter: fakeClockWaiter{
			targetTime: stopTime,
			destChan:   ch,
		},
	}
	f.waiters = append(f.waiters, timer.waiter)
	return timer
}

func (f *FakeClock) NewTicker(d time.Duration) Ticker {
	f.lock.Lock()
	defer f.lock.Unlock()
	tickTime := f.time.Add(d)
	ch := make(chan time.Time, 1) // hold one tick
	f.waiters = append(f.waiters, fakeClockWaiter{
		targetTime:    tickTime,
		stepInterval:  d,
		skipIfBlocked: true,
		destChan:      ch,
	})

	return &fakeTicker{
		c: ch,
	}
}

// Move clock by Duration, notify anyone that's called After, Tick, or NewTimer
func (f *FakeClock) Step(d time.Duration) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.setTimeLocked(f.time.Add(d))
}

// Sets the time.
func (f *FakeClock) SetTime(t time.Time) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.setTimeLocked(t)
}

// Actually changes the time and checks any waiters. f must be write-locked.
func (f *FakeClock) setTimeLocked(t time.Time) {
	f.time = t
	newWaiters := make([]fakeClockWaiter, 0, len(f.waiters))
	for i := range f.waiters {
		w := &f.waiters[i]
		if !w.targetTime.After(t) {

			if w.skipIfBlocked {
				select {
				case w.destChan <- t:
					w.fired = true
				default:
				}
			} else {
				w.destChan <- t
				w.fired = true
			}

			if w.stepInterval > 0 {
				for !w.targetTime.After(t) {
					w.targetTime = w.targetTime.Add(w.stepInterval)
				}
				newWaiters = append(newWaiters, *w)
			}

		} else {
			newWaiters = append(newWaiters, f.waiters[i])
		}
	}
	f.waiters = newWaiters
}

// Returns true if After has been called on f but not yet satisfied (so you can
// write race-free tests).
func (f *FakeClock) HasWaiters() bool {
	f.lock.RLock()
	defer f.lock.RUnlock()
	return len(f.waiters) > 0
}

func (f *FakeClock) Sleep(d time.Duration) {
	f.Step(d)
}

// IntervalClock implements Clock, but each invocation of Now steps the clock forward the specified duration
type IntervalClock struct {
	Time     time.Time
	Duration time.Duration
}

// Now returns i's time.
func (i *IntervalClock) Now() time.Time {
	i.Time = i.Time.Add(i.Duration)
	return i.Time
}

// Since returns time since the time in i.
func (i *IntervalClock) Since(ts time.Time) time.Duration {
	return i.Time.Sub(ts)
}

// Unimplemented, will panic.
// TODO: make interval clock use FakeClock so this can be implemented.
func (*IntervalClock) After(d time.Duration) <-chan time.Time {
	panic("IntervalClock doesn't implement After")
}

// Unimplemented, will panic.
// TODO: make interval clock use FakeClock so this can be implemented.
func (*IntervalClock) NewTimer(d time.Duration) Timer {
	panic("IntervalClock doesn't implement NewTimer")
}

// Unimplemented, will panic.
// TODO: make interval clock use FakeClock so this can be implemented.
func (*IntervalClock) NewTicker(d time.Duration) Ticker {
	panic("IntervalClock doesn't implement NewTicker")
}

func (*IntervalClock) Sleep(d time.Duration) {
	panic("IntervalClock doesn't implement Sleep")
}

// Timer allows for injecting fake or real timers into code that
// needs to do arbitrary things based on time.
type Timer interface {
	C() <-chan time.Time
	Stop() bool
	Reset(d time.Duration) bool
}

// realTimer is backed by an actual time.Timer.
type realTimer struct {
	timer *time.Timer
}

// C returns the underlying timer's channel.
func (r *realTimer) C() <-chan time.Time {
	return r.timer.C
}

// Stop calls Stop() on the underlying timer.
func (r *realTimer) Stop() bool {
	return r.timer.Stop()
}

// Reset calls Reset() on the underlying timer.
func (r *realTimer) Reset(d time.Duration) bool {
	return r.timer.Reset(d)
}

// fakeTimer implements Timer based on a FakeClock.
type fakeTimer struct {
	fakeClock *FakeClock
	waiter    fakeClockWaiter
}

// C returns the channel that notifies when this timer has fired.
func (f *fakeTimer) C() <-chan time.Time {
	return f.waiter.destChan
}

// Stop stops the timer and returns true if the timer has not yet fired, or false otherwise.
func (f *fakeTimer) Stop() bool {
	f.fakeClock.lock.Lock()
	defer f.fakeClock.lock.Unlock()

	newWaiters := make([]fakeClockWaiter, 0, len(f.fakeClock.waiters))
	for i := range f.fakeClock.waiters {
		w := &f.fakeClock.waiters[i]
		if w != &f.waiter {
			newWaiters = append(newWaiters, *w)
		}
	}

	f.fakeClock.waiters = newWaiters

	return !f.waiter.fired
}

// Reset resets the timer to the fake clock's "now" + d. It returns true if the timer has not yet
// fired, or false otherwise.
func (f *fakeTimer) Reset(d time.Duration) bool {
	f.fakeClock.lock.Lock()
	defer f.fakeClock.lock.Unlock()

	active := !f.waiter.fired

	f.waiter.fired = false
	f.waiter.targetTime = f.fakeClock.time.Add(d)

	return active
}

type Ticker interface {
	C() <-chan time.Time
	Stop()
}

type realTicker struct {
	ticker *time.Ticker
}

func (t *realTicker) C() <-chan time.Time {
	return t.ticker.C
}

func (t *realTicker) Stop() {
	t.ticker.Stop()
}

type fakeTicker struct {
	c <-chan time.Time
}

func (t *fakeTicker) C() <-chan time.Time {
	return t.c
}

func (t *fakeTicker) Stop() {
}
