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

package ktesting

import (
	"context"
	"fmt"
	"time"
)

// cleanupErr creates a cause when canceling a context because the test has completed.
// It is a context.Canceled error.
func cleanupErr(testName string) error {
	return canceledError(fmt.Sprintf("test %s is cleaning up", testName))
}

type canceledError string

func (c canceledError) Error() string { return string(c) }

func (c canceledError) Is(target error) bool {
	return target == context.Canceled
}

// withTimeout corresponds to [context.WithTimeout]. In contrast to
// [context.WithTimeout], it automatically cancels during test cleanup, provides
// the given cause when the deadline is reached, and its cancel function
// requires a cause.
func withTimeout(ctx context.Context, tb TB, timeout time.Duration, timeoutCause string) (context.Context, func(cause string)) {
	tb.Helper()

	now := time.Now()

	cancelCtx, cancel := context.WithCancelCause(ctx)
	after := time.NewTimer(timeout)
	stopCtx, stop := context.WithCancel(ctx) // Only used internally, doesn't need a cause.
	tb.Cleanup(func() {
		cancel(cleanupErr(tb.Name()))
		stop()
	})
	go func() {
		select {
		case <-stopCtx.Done():
			after.Stop()
			// No need to set a cause here. The cause or error of
			// the parent context will be used.
		case <-after.C:
			// Code using this tCtx may or may not log the
			// information above when it runs into the
			// cancellation. It's better if we do it, just to be on
			// the safe side.
			//
			// Would be nice to log this with the source code location
			// of our caller, but testing.Logf does not support that.
			tb.Logf("\nWARNING: %s\n", timeoutCause)
			cancel(canceledError(timeoutCause))
		}
	}()

	// Determine which deadline is sooner: ours or that of our parent.
	deadline := now.Add(timeout)
	if parentDeadline, ok := ctx.Deadline(); ok {
		if deadline.After(parentDeadline) {
			deadline = parentDeadline
		}
	}

	// We always have a deadline.
	return deadlineContext{Context: cancelCtx, deadline: deadline}, func(cause string) {
		var cancelCause error
		if cause != "" {
			cancelCause = canceledError(cause)
		}
		cancel(cancelCause)
	}
}

type deadlineContext struct {
	context.Context
	deadline time.Time
}

func (d deadlineContext) Deadline() (time.Time, bool) {
	return d.deadline, true
}
