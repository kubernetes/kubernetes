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

package util

import (
	"sync"
	"time"
)

// Clock allows for injecting fake or real clocks into code that
// needs to do arbitrary things based on time.
type Clock interface {
	Now() time.Time
	Since(time.Time) time.Duration
	After(d time.Duration) <-chan time.Time
	Sleep(d time.Duration)
	Tick(d time.Duration) <-chan time.Time
}

var (
	_ = Clock(RealClock{})
	_ = Clock(&FakeClock{})
	_ = Clock(&IntervalClock{})
)

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

func (RealClock) Tick(d time.Duration) <-chan time.Time {
	return time.Tick(d)
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
	destChan      chan<- time.Time
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

func (f *FakeClock) Tick(d time.Duration) <-chan time.Time {
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

	return ch
}

// Move clock by Duration, notify anyone that's called After or Tick
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
func (*IntervalClock) Tick(d time.Duration) <-chan time.Time {
	panic("IntervalClock doesn't implement Tick")
}

func (*IntervalClock) Sleep(d time.Duration) {
	panic("IntervalClock doesn't implement Sleep")
}
