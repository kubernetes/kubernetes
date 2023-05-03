/*
Copyright 2020 The Kubernetes Authors.

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

package internal

import (
	"sync"
	"time"
)

// AtMostEvery will never run the method more than once every specified
// duration.
type AtMostEvery struct {
	delay    time.Duration
	lastCall time.Time
	mutex    sync.Mutex
}

// NewAtMostEvery creates a new AtMostEvery, that will run the method at
// most every given duration.
func NewAtMostEvery(delay time.Duration) *AtMostEvery {
	return &AtMostEvery{
		delay: delay,
	}
}

// updateLastCall returns true if the lastCall time has been updated,
// false if it was too early.
func (s *AtMostEvery) updateLastCall() bool {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	if time.Since(s.lastCall) < s.delay {
		return false
	}
	s.lastCall = time.Now()
	return true
}

// Do will run the method if enough time has passed, and return true.
// Otherwise, it does nothing and returns false.
func (s *AtMostEvery) Do(fn func()) bool {
	if !s.updateLastCall() {
		return false
	}
	fn()
	return true
}
