/*
Copyright 2023 The Kubernetes Authors.

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

package helper

import (
	"sync"
	"sync/atomic"
	"time"
)

// SleepControl sends a controller to sleep after a certain amount of time without a
// tick event, or wakes it up on tick when it is sleeping.
type SleepControl struct {
	// WakeupFn is called when the controller wakes up. SleepFn is called when the
	// controller falls asleep.
	WakeupFn, SleepFn func()

	// FallAsleepAfter is the duration after which the controller will fall asleep.
	FallAsleepAfter time.Duration

	// sleepTime is the time at which the controller will fall asleep. It is updated
	// by the Tick method to now + FallAsleepAfter.
	sleepTime atomic.Value

	lock  sync.RWMutex
	awake bool
}

// Tick keeps the controller awake, and wakes it up if it is sleeping.
func (c *SleepControl) Tick() {
	c.lock.RLock()
	awake := c.awake
	c.lock.RUnlock()

	if !awake {
		c.lock.Lock()
		defer c.lock.Unlock()
		if !c.awake {
			c.WakeupFn()
			c.awake = true
		}
	}
	c.sleepTime.Store(time.Now().Add(c.FallAsleepAfter))
}

// Run runs the sleep control loop. It will call SleepFn when the controller has not
// seen a call of Tick for more than FallAsleepAfter. Period is the duration after which
// the controller will check if it has to fall asleep.
//
// When stopCh is closed, Run will call SleepFn if the controller is not sleeping, and
// return.
func (c *SleepControl) Run(period time.Duration, stopCh <-chan struct{}) {
	for {
		select {
		case <-stopCh:
			c.lock.Lock()
			defer c.lock.Unlock()

			if c.awake {
				c.SleepFn()
			}

			return
		case <-time.After(period):
			func() {
				c.lock.Lock()
				defer c.lock.Unlock()

				if c.awake && time.Now().After(c.sleepTime.Load().(time.Time)) {
					c.SleepFn()
					c.awake = false
				}
			}()
		}
	}
}
