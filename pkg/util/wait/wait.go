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
	"errors"
	"math/rand"
	"time"
)

// Jitter returns a time.Duration between duration and duration + maxFactor * duration,
// to allow clients to avoid converging on periodic behavior.  If maxFactor is 0.0, a
// suggested default value will be chosen.
func Jitter(duration time.Duration, maxFactor float64) time.Duration {
	if maxFactor <= 0.0 {
		maxFactor = 1.0
	}
	wait := duration + time.Duration(rand.Float64()*maxFactor*float64(duration))
	return wait
}

// ErrWaitTimeout is returned when the condition exited without success
var ErrWaitTimeout = errors.New("timed out waiting for the condition")

// ConditionFunc returns true if the condition is satisfied, or an error
// if the loop should be aborted.
type ConditionFunc func() (done bool, err error)

// Poll tries a condition func until it returns true, an error, or the timeout
// is reached. condition will always be invoked at least once but some intervals
// may be missed if the condition takes too long or the time window is too short.
// If you want to Poll something forever, see PollInfinite.
// Poll always waits the interval before the first check of the condition.
func Poll(interval, timeout time.Duration, condition ConditionFunc) error {
	return pollInternal(poller(interval, timeout), condition)
}

func pollInternal(wait WaitFunc, condition ConditionFunc) error {
	done := make(chan struct{})
	defer close(done)
	return WaitFor(wait, condition, done)
}

func PollImmediate(interval, timeout time.Duration, condition ConditionFunc) error {
	return pollImmediateInternal(poller(interval, timeout), condition)
}

func pollImmediateInternal(wait WaitFunc, condition ConditionFunc) error {
	done, err := condition()
	if err != nil {
		return err
	}
	if done {
		return nil
	}
	return pollInternal(wait, condition)
}

// PollInfinite polls forever.
func PollInfinite(interval time.Duration, condition ConditionFunc) error {
	done := make(chan struct{})
	defer close(done)
	return WaitFor(poller(interval, 0), condition, done)
}

// WaitFunc creates a channel that receives an item every time a test
// should be executed and is closed when the last test should be invoked.
type WaitFunc func(done <-chan struct{}) <-chan struct{}

// WaitFor gets a channel from wait(), and then invokes fn once for every value
// placed on the channel and once more when the channel is closed.  If fn
// returns an error the loop ends and that error is returned, and if fn returns
// true the loop ends and nil is returned. ErrWaitTimeout will be returned if
// the channel is closed without fn ever returning true.
func WaitFor(wait WaitFunc, fn ConditionFunc, done <-chan struct{}) error {
	c := wait(done)
	for {
		_, open := <-c
		ok, err := fn()
		if err != nil {
			return err
		}
		if ok {
			return nil
		}
		if !open {
			break
		}
	}
	return ErrWaitTimeout
}

// poller returns a WaitFunc that will send to the channel every
// interval until timeout has elapsed and then close the channel.
// Over very short intervals you may receive no ticks before
// the channel is closed.  If timeout is 0, the channel
// will never be closed.
func poller(interval, timeout time.Duration) WaitFunc {
	return WaitFunc(func(done <-chan struct{}) <-chan struct{} {
		ch := make(chan struct{})

		go func() {
			defer close(ch)

			tick := time.NewTicker(interval)
			defer tick.Stop()

			var after <-chan time.Time
			if timeout != 0 {
				// time.After is more convenient, but it
				// potentially leaves timers around much longer
				// than necessary if we exit early.
				timer := time.NewTimer(timeout)
				after = timer.C
				defer timer.Stop()
			}

			for {
				select {
				case <-tick.C:
					ch <- struct{}{}
				case <-after:
					return
				case <-done:
					return
				}
			}
		}()

		return ch
	})
}
