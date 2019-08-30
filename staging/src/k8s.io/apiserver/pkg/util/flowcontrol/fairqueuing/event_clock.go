/*
Copyright 2019 The Kubernetes Authors.

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

package fairqueuing

import (
	"container/heap"
	"math/rand"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
)

// EventFunc does some work that needs to be done at or after the
// given time.  All of the work that needs to be done is finished
// before this function returns and, if there is an associated
// `sync.WaitGroup` (see NewFakeEventClock), its counter goes to zero;
// no other work is left to be completed in goroutines.
type EventFunc func(time.Time)

// SettablePassiveClock allows setting current time of a passive clock
type SettablePassiveClock interface {
	clock.PassiveClock
	SetTime(time.Time)
}

// EventClock fires event on time
type EventClock interface {
	clock.PassiveClock
	EventAfterDuration(f EventFunc, d time.Duration)
	EventAfterTime(f EventFunc, t time.Time)
}

// RealEventClock fires event on real world time
type RealEventClock struct {
	clock.RealClock
}

// EventAfterDuration schedules an EventFunc
func (RealEventClock) EventAfterDuration(f EventFunc, d time.Duration) {
	ch := time.After(d)
	go func() {
		select {
		case t := <-ch:
			f(t)
		}
	}()
}

// EventAfterTime schedules an EventFunc
func (r RealEventClock) EventAfterTime(f EventFunc, t time.Time) {
	now := time.Now()
	d := t.Sub(now)
	if d <= 0 {
		go f(now)
	} else {
		r.EventAfterDuration(f, d)
	}
}

// FakeEventClock is one whose time does not pass implicitly but
// rather is explicitly set by invocations of its SetTime method
type FakeEventClock struct {
	clock.FakePassiveClock

	// waiters is a heap of waiting work, sorted by time
	waiters     eventWaiterHeap
	waitersLock sync.RWMutex

	// clientWG may be nil and if not supplies constraints on time
	// passing in Run.  The Run method will not pick a new time until
	// this is nil or its counter is zero.
	clientWG *sync.WaitGroup

	// fuzz is the amount of noise to add to scheduling.  An event
	// requested to run at time T will run at some time chosen
	// uniformly at random from the interval [T, T+fuzz]; the upper
	// bound is exclusive iff fuzz is non-zero.
	fuzz time.Duration

	// rand is the random number generator to use in fuzzing
	rand *rand.Rand
}

type eventWaiterHeap []eventWaiter

var _ heap.Interface = (*eventWaiterHeap)(nil)

type eventWaiter struct {
	targetTime time.Time
	f          EventFunc
}

// NewFakeEventClock constructor.  The given `r *rand.Rand` must
// henceforth not be used for any other purpose.  If `r` is nil then a
// fresh one will be constructed, seeded with the current real time.
// The clientWG can be `nil` and if not is used to let Run know about
// additional work that has to complete before time can advance.
func NewFakeEventClock(t time.Time, clientWG *sync.WaitGroup, fuzz time.Duration, r *rand.Rand) *FakeEventClock {
	if r == nil {
		r = rand.New(rand.NewSource(time.Now().UnixNano()))
		r.Uint64()
		r.Uint64()
		r.Uint64()
	}
	return &FakeEventClock{
		FakePassiveClock: *clock.NewFakePassiveClock(t),
		clientWG:         clientWG,
		fuzz:             fuzz,
		rand:             r,
	}
}

// GetNextTime returns the next time at which there is work scheduled,
// and a bool indicating whether there is any such time
func (fec *FakeEventClock) GetNextTime() (time.Time, bool) {
	fec.waitersLock.RLock()
	defer fec.waitersLock.RUnlock()
	if len(fec.waiters) > 0 {
		return fec.waiters[0].targetTime, true
	}
	return time.Time{}, false
}

// Run runs all the events scheduled, and all the events they
// schedule, and so on, until there are none scheduled or the next
// time would exceed the limit.  The clientWG given in the constructor
// gates each advance of time.
func (fec *FakeEventClock) Run(limit *time.Time) {
	for {
		if fec.clientWG != nil {
			fec.clientWG.Wait()
		}
		t, ok := fec.GetNextTime()
		if !ok || limit != nil && t.After(*limit) {
			break
		}
		fec.SetTime(t)
	}
}

// SetTime sets the time and runs to completion all events that should
// be started by the given time --- including any further events they
// schedule
func (fec *FakeEventClock) SetTime(t time.Time) {
	fec.FakePassiveClock.SetTime(t)
	for {
		foundSome := false
		func() {
			fec.waitersLock.Lock()
			defer fec.waitersLock.Unlock()
			// This loop is because events run at a given time may schedule more
			// events to run at that or an earlier time.
			// Events should not advance the clock.  But just in case they do...
			now := fec.Now()
			var wg sync.WaitGroup
			for len(fec.waiters) > 0 && !now.Before(fec.waiters[0].targetTime) {
				ew := heap.Pop(&fec.waiters).(eventWaiter)
				wg.Add(1)
				go func(f EventFunc) { f(now); wg.Done() }(ew.f)
				foundSome = true
			}
			wg.Wait()
		}()
		if !foundSome {
			break
		}
	}
}

// EventAfterDuration schedules the given function to be invoked once
// the given duration has passed.
func (fec *FakeEventClock) EventAfterDuration(f EventFunc, d time.Duration) {
	fec.waitersLock.Lock()
	defer fec.waitersLock.Unlock()
	now := fec.Now()
	fd := time.Duration(float32(fec.fuzz) * fec.rand.Float32())
	heap.Push(&fec.waiters, eventWaiter{targetTime: now.Add(d + fd), f: f})
}

// EventAfterTime schedules the given function to be invoked once
// the given time has arrived.
func (fec *FakeEventClock) EventAfterTime(f EventFunc, t time.Time) {
	fec.waitersLock.Lock()
	defer fec.waitersLock.Unlock()
	fd := time.Duration(float32(fec.fuzz) * fec.rand.Float32())
	heap.Push(&fec.waiters, eventWaiter{targetTime: t.Add(fd), f: f})
}

func (ewh eventWaiterHeap) Len() int { return len(ewh) }

func (ewh eventWaiterHeap) Less(i, j int) bool { return ewh[i].targetTime.Before(ewh[j].targetTime) }

func (ewh eventWaiterHeap) Swap(i, j int) { ewh[i], ewh[j] = ewh[j], ewh[i] }

func (ewh *eventWaiterHeap) Push(x interface{}) {
	*ewh = append(*ewh, x.(eventWaiter))
}

func (ewh *eventWaiterHeap) Pop() interface{} {
	old := *ewh
	n := len(old)
	x := old[n-1]
	*ewh = old[:n-1]
	return x
}
