/*
Copyright 2014 Google Inc. All rights reserved.

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
	"time"
)

// ErrWaitTimeout is returned when the condition exited without success
var ErrWaitTimeout = errors.New("timed out waiting for the condition")

// ConditionFunc returns true if the condition is satisfied, or an error
// if the loop should be aborted.
type ConditionFunc func() (done bool, err error)

// Poll tries a condition func until it returns true, an error, or the
// wait channel is closed.  Will always poll at least once.
func Poll(interval time.Duration, cycles int, condition ConditionFunc) error {
	return WaitFor(poller(interval, cycles), condition)
}

// WaitFunc creates a channel that receives an item every time a test
// should be executed and is closed when the last test should be invoked.
type WaitFunc func() <-chan struct{}

// WaitFor implements the looping for a wait.
func WaitFor(wait WaitFunc, c ConditionFunc) error {
	w := wait()
	for {
		_, open := <-w
		ok, err := c()
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
// interval until at most cycles * interval has elapsed and then
// close the channel. Over very short intervals you may receive
// no ticks before being closed.
func poller(interval time.Duration, cycles int) WaitFunc {
	return WaitFunc(func() <-chan struct{} {
		ch := make(chan struct{})
		go func() {
			tick := time.NewTicker(interval)
			defer tick.Stop()
			after := time.After(interval * time.Duration(cycles))
			for {
				select {
				case <-tick.C:
					ch <- struct{}{}
				case <-after:
					close(ch)
					return
				}
			}
		}()
		return ch
	})
}
