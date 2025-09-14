/*
Copyright 2025 The Kubernetes Authors.

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

package apidispatcher

import (
	"sync"
)

// goroutinesLimiter limits the number of goroutines that can be dispatched.
type goroutinesLimiter struct {
	lock   sync.RWMutex
	cond   *sync.Cond
	closed bool

	// maxGoroutines is the maximum number of goroutines that can be dispatched concurrently by the APIDispatcher.
	maxGoroutines int
	// goroutines is the current number of goroutines allowed to run.
	// Must be used under lock.
	goroutines int
}

// newGoroutinesLimiter returns a new goroutinesLimiter object.
func newGoroutinesLimiter(maxGoroutines int) *goroutinesLimiter {
	p := goroutinesLimiter{
		maxGoroutines: maxGoroutines,
	}
	p.cond = sync.NewCond(&p.lock)

	return &p
}

// acquire attempts to reserve a goroutine to run.
// It blocks until a goroutine is available.
// It returns true if the goroutine was successfully acquired.
// If the limiter is closed, it returns false.
// Goroutine should be released after use.
func (gl *goroutinesLimiter) acquire() bool {
	gl.lock.Lock()
	defer gl.lock.Unlock()

	for gl.goroutines >= gl.maxGoroutines {
		if gl.closed {
			return false
		}
		// Wait for a goroutine to become available.
		gl.cond.Wait()
	}
	gl.goroutines++
	return true
}

// release decrements the active goroutine count.
// Goroutine should be previously acquired.
func (gl *goroutinesLimiter) release() {
	gl.lock.Lock()
	defer gl.lock.Unlock()

	gl.goroutines--
	gl.cond.Broadcast()
}

// close shuts down the goroutinesLimiter.
func (gl *goroutinesLimiter) close() {
	gl.lock.Lock()
	defer gl.lock.Unlock()

	gl.closed = true
	gl.cond.Broadcast()
}
