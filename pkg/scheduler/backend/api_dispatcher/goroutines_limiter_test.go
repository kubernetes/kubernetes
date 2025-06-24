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
	"testing"
	"time"
)

func TestGoroutinesLimiter(t *testing.T) {
	t.Run("Two goroutines are correctly acquired", func(t *testing.T) {
		limiter := newGoroutinesLimiter(2)

		runner := limiter.acquire()
		if runner == nil {
			t.Errorf("Expected the first goroutine to be acquired, but it's not")
		}
		runner = limiter.acquire()
		if runner == nil {
			t.Errorf("Expected the second goroutine to be acquired, but it's not")
		}
	})

	t.Run("Goroutine is acquired after releasing", func(t *testing.T) {
		limiter := newGoroutinesLimiter(1)

		runner := limiter.acquire()
		if runner == nil {
			t.Errorf("Expected the goroutine to be acquired, but it's not")
		}

		runnerCh := make(chan goroutineRunner)
		go func() {
			runnerCh <- limiter.acquire()
		}()

		select {
		case <-runnerCh:
			t.Fatal("acquire() should have blocked when the limit was reached, but it returned immediately")
		case <-time.After(100 * time.Millisecond):
		}

		runner.run(func() {
			// Executing the goroutine.
		})
		select {
		case runner := <-runnerCh:
			if runner == nil {
				t.Errorf("Expected the goroutine to be acquired after releasing one")
			}
		case <-time.After(100 * time.Millisecond):
			t.Fatal("acquire() should have been unblocked after release(), but it remained blocked")
		}
	})

	t.Run("Acquiring terminates when limiter is closed", func(t *testing.T) {
		limiter := newGoroutinesLimiter(1)

		runner := limiter.acquire()
		if runner == nil {
			t.Errorf("Expected the goroutine to be acquired, but it's not")
		}

		runnerCh := make(chan goroutineRunner)
		go func() {
			runnerCh <- limiter.acquire()
		}()

		select {
		case <-runnerCh:
			t.Fatal("acquire() should have blocked when the limit was reached, but it returned immediately")
		case <-time.After(100 * time.Millisecond):
		}

		limiter.close()
		select {
		case runner := <-runnerCh:
			if runner != nil {
				t.Errorf("Expected the goroutine not to be acquired after closing the limiter")
			}
		case <-time.After(100 * time.Millisecond):
			t.Fatal("acquire() should have been unblocked after close(), but it remained blocked")
		}
	})
}
