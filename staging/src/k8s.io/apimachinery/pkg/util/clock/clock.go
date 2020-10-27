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

// PassiveClock allows for injecting fake or real clocks into code
// that needs to read the current time but does not support scheduling
// activity in the future.
type PassiveClock interface {
	Now() time.Time
	Since(time.Time) time.Duration
}

// Clock allows for injecting fake or real clocks into code that
// needs to do arbitrary things based on time.
type Clock interface {
	PassiveClock
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

// After is the same as time.After(d).
func (RealClock) After(d time.Duration) <-chan time.Time {
	return time.After(d)
}

// NewTimer returns a new Timer.
func (RealClock) NewTimer(d time.Duration) Timer {
	return &realTimer{
		timer: time.NewTimer(d),
	}
}

// NewTicker returns a new Ticker.
func (RealClock) NewTicker(d time.Duration) Ticker {
	return &realTicker{
		ticker: time.NewTicker(d),
	}
}

// Sleep pauses the RealClock for duration d.
func (RealClock) Sleep(d time.Duration) {
	time.Sleep(d)
}

// FakePassiveClock implements PassiveClock, but returns an arbitrary time.
type FakePassiveClock struct {
	lock sync.RWMutex
	time time.Time
}

// FakeClock implements Clock, but returns an arbitrary time.
type FakeClock struct {
	FakePassiveClock

	// waiters are waiting for the fake time to pass their specified time
	waiters []fakeClockWaiter
}

type fakeClockWaiter struct {
	targetTime    time.Time
	stepInterval  time.Duration
	skipIfBlocked bool
	destChan      chan time.Time
}

// NewFakePassiveClock returns a new FakePassiveClock.
func NewFakePassiveClock(t time.Time) *FakePassiveClock {
	return &FakePassiveClock{
		time: t,
	}
}

// NewFakeClock returns a new FakeClock
func NewFakeClock(t time.Time) *FakeClock {
	return &FakeClock{
		FakePassiveClock: *NewFakePassiveClock(t),
	}
}

// Now returns f's time.
func (f *FakePassiveClock) Now() time.Time {
	f.lock.RLock()
	defer f.lock.RUnlock()
	return f.time
}

// Since returns time since the time in f.
func (f *FakePassiveClock) Since(ts time.Time) time.Duration {
	f.lock.RLock()
	defer f.lock.RUnlock()
	return f.time.Sub(ts)
}

// SetTime sets the time on the FakePassiveClock.
func (f *FakePassiveClock) SetTime(t time.Time) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.time = t
}

// After is the Fake version of time.After(d).
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

// NewTimer is the Fake version of time.NewTimer(d).
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

// NewTicker returns a new Ticker.
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

// Step moves clock by Duration, notifies anyone that's called After, Tick, or NewTimer
func (f *FakeClock) Step(d time.Duration) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.setTimeLocked(f.time.Add(d))
}

// SetTime sets the time on a FakeClock.
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
				default:
				}
			} else {
				w.destChan <- t
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

// HasWaiters returns true if After has been called on f but not yet satisfied (so you can
// write race-free tests).
func (f *FakeClock) HasWaiters() bool {
	f.lock.RLock()
	defer f.lock.RUnlock()
	return len(f.waiters) > 0
}

// Sleep pauses the FakeClock for duration d.
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

// After is currently unimplemented, will panic.
// TODO: make interval clock use FakeClock so this can be implemented.
func (*IntervalClock) After(d time.Duration) <-chan time.Time {
	panic("IntervalClock doesn't implement After")
}

// NewTimer is currently unimplemented, will panic.
// TODO: make interval clock use FakeClock so this can be implemented.
func (*IntervalClock) NewTimer(d time.Duration) Timer {
	panic("IntervalClock doesn't implement NewTimer")
}

// NewTicker is currently unimplemented, will panic.
// TODO: make interval clock use FakeClock so this can be implemented.
func (*IntervalClock) NewTicker(d time.Duration) Ticker {
	panic("IntervalClock doesn't implement NewTicker")
}

// Sleep is currently unimplemented; will panic.
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

// Stop conditionally stops the timer.  If the timer has neither fired
// nor been stopped then this call stops the timer and returns true,
// otherwise this call returns false.  This is like time.Timer::Stop.
func (f *fakeTimer) Stop() bool {
	f.fakeClock.lock.Lock()
	defer f.fakeClock.lock.Unlock()
	// The timer has already fired or been stopped, unless it is found
	// among the clock's waiters.
	stopped := false
	oldWaiters := f.fakeClock.waiters
	newWaiters := make([]fakeClockWaiter, 0, len(oldWaiters))
	seekChan := f.waiter.destChan
	for i := range oldWaiters {
		// Identify the timer's fakeClockWaiter by the identity of the
		// destination channel, nothing else is necessarily unique and
		// constant since the timer's creation.
		if oldWaiters[i].destChan == seekChan {
			stopped = true
		} else {
			newWaiters = append(newWaiters, oldWaiters[i])
		}
	}

	f.fakeClock.waiters = newWaiters

	return stopped
}

// Reset conditionally updates the firing time of the timer.  If the
// timer has neither fired nor been stopped then this call resets the
// timer to the fake clock's "now" + d and returns true, otherwise
// it creates a new waiter, adds it to the clock, and returns true.
//
// It is not possible to return false, because a fake timer can be reset
// from any state (waiting to fire, already fired, and stopped).
//
// See the GoDoc for time.Timer::Reset for more context on why
// the return value of Reset() is not useful.
func (f *fakeTimer) Reset(d time.Duration) bool {
	f.fakeClock.lock.Lock()
	defer f.fakeClock.lock.Unlock()
	waiters := f.fakeClock.waiters
	seekChan := f.waiter.destChan
	for i := range waiters {
		if waiters[i].destChan == seekChan {
			waiters[i].targetTime = f.fakeClock.time.Add(d)
			return true
		}
	}
	// No existing waiter, timer has already fired or been reset.
	// We should still enable Reset() to succeed by creating a
	// new waiter and adding it to the clock's waiters.
	newWaiter := fakeClockWaiter{
		targetTime: f.fakeClock.time.Add(d),
		destChan:   seekChan,
	}
	f.fakeClock.waiters = append(f.fakeClock.waiters, newWaiter)
	return true
}

// Ticker defines the Ticker interface
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
