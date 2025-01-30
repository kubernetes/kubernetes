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

package wait

import (
	"context"
	"sync"
	"time"

	"k8s.io/utils/clock"
)

// DelayFunc returns the next time interval to wait.
type DelayFunc func() time.Duration

// Timer takes an arbitrary delay function and returns a timer that can handle arbitrary interval changes.
// Use Backoff{...}.Timer() for simple delays and more efficient timers.
func (fn DelayFunc) Timer(c clock.Clock) Timer {
	return &variableTimer{fn: fn, new: c.NewTimer}
}

// Until takes an arbitrary delay function and runs until cancelled or the condition indicates exit. This
// offers all of the functionality of the methods in this package.
func (fn DelayFunc) Until(ctx context.Context, immediate, sliding bool, condition ConditionWithContextFunc) error {
	return loopConditionUntilContext(ctx, &variableTimer{fn: fn, new: internalClock.NewTimer}, immediate, sliding, condition)
}

// Concurrent returns a version of this DelayFunc that is safe for use by multiple goroutines that
// wish to share a single delay timer.
func (fn DelayFunc) Concurrent() DelayFunc {
	var lock sync.Mutex
	return func() time.Duration {
		lock.Lock()
		defer lock.Unlock()
		return fn()
	}
}
