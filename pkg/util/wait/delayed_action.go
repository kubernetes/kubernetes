/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package wait

import (
	"time"
)

// nowFn is a function for testing
var nowFn = time.Now

// Interval defines an interval over which a certain number of events are expected.
type Interval struct {
	Count    int
	Interval time.Duration
}

// NewChannelDelayedAction provides efficient aggregation and rate limiting of actions that must be executed,
// but can tolerate delay. It accepts a series of steps that allow increasingly larger delays at higher
// rates. Intended to be used in a for{ select{...} } loop on single threaded actions:
//
//     delay := NewChannelDelayedAction(Interval{Count: 10, Interval: time.Second})
//     for {
//       select {
//       case <-delay.After():
//         delay.Done()
//         // perform a rate limited action
//       ...
//       }
//       if delay.Run() {
//         // perform a rate limited action
//       }
//     }
//
// If the for loop runs faster than 10 events per second, Run() will return false and the delay channel will have
// an item when the wait is over. Done() is required to clear the state in the loop. If the rate drops
// below 10/sec, then no delays will be used. The rate is calculated over the time necessary to accumulate 10
// events (not an instaneous rate).
func NewChannelDelayedAction(steps ...Interval) *ChannelDelayedAction {
	return &ChannelDelayedAction{steps: steps}
}

// ChannelDelayedAction allows callers to delay expensive actions inside of for-select loops.
type ChannelDelayedAction struct {
	active <-chan time.Time
	t      *time.Timer

	steps    []Interval
	count    int
	interval time.Duration
	last     time.Time
}

// After is the channel to listen for delayed actions in a for select loop.
func (q *ChannelDelayedAction) After() <-chan time.Time {
	return q.active
}

// Done indicates that the active channel has been read and that subsequent actions should be on
// a different interval.
func (q *ChannelDelayedAction) Done() {
	q.active = nil
	q.last = time.Time{}
}

// Run returns true if the action should be performed, and false if it should be run only when After()
// returns.
func (q *ChannelDelayedAction) Run() bool {
	q.count++
	if q.active != nil {
		return false
	}
	if q.shouldRun() {
		return true
	}

	if q.t == nil {
		q.t = time.NewTimer(q.interval)
	} else if q.t.Reset(q.interval) {
		<-q.t.C
	}
	q.active = q.t.C
	return false
}

// shouldRun returns true if the action should run right away, resetting q.interval and q.count
// as necessary.
func (q *ChannelDelayedAction) shouldRun() bool {
	count := q.count

	// have not accumulated to first threshold, so record start of interval
	if count < q.steps[0].Count {
		if q.last.IsZero() {
			q.last = nowFn()
		}
		return true
	}

	// we're over the first threshold, but we need to have a sufficient rate to be delayed
	if !q.last.IsZero() {
		now := nowFn()
		interval := now.Sub(q.last)
		// check our rate is above the first threshold
		rate := float32(count) / float32(interval)
		stepRate := float32(q.steps[0].Count) / float32(q.steps[0].Interval)
		q.count = 0
		if rate < stepRate {
			// no rate high enough, continue to invoke directly
			q.last = now
			return true
		}
		// move to longer polling
		q.interval = q.steps[0].Interval
		return false
	}

	// find the highest threshold we match
	for i := len(q.steps) - 1; i >= 0; i-- {
		if count >= q.steps[i].Count {
			q.count = 0
			q.interval = q.steps[i].Interval
			return false
		}
	}
	return true
}
