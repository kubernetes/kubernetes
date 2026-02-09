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
	"time"

	"k8s.io/klog/v2"
)

// WithCancel sets up cancellation in a [TContext.Cleanup] callback and
// constructs a new TContext where [TContext.Cancel] cancels only the new
// context.
func (tc *TC) WithCancel() TContext {
	ctx, cancel := context.WithCancelCause(tc)

	tc = tc.clone()
	tc.Context = ctx
	tc.cancel = func(cause string) {
		var cancelCause error
		if cause != "" {
			cancelCause = canceledError(cause)
		}
		cancel(cancelCause)
	}
	return tc
}

// WithoutCancel causes the returned context to ignore cancellation of its parent.
// Calling Cancel will not cancel the parent either.
// This matches [context.WithoutCancel].
func (tc *TC) WithoutCancel() TContext {
	ctx := context.WithoutCancel(tc)

	tc = tc.clone()
	tc.Context = ctx
	tc.cancel = nil
	return tc
}

// WithTimeout sets up new context with a timeout. Canceling the timeout gets
// registered in a cleanup callback. [TContext.Cancel] cancels only
// the new context. The cause is used as reason why the context is canceled
// once the timeout is reached. It may be empty, in which case the usual
// "context canceled" error is used.
func (tc *TC) WithTimeout(timeout time.Duration, timeoutCause string) TContext {
	ctx, cancel := withTimeout(tc, tc.TB(), timeout, timeoutCause)

	tc = tc.clone()
	tc.Context = ctx
	tc.cancel = cancel
	return tc
}

// WithLogger constructs a new context with a different logger.
func (tc *TC) WithLogger(logger klog.Logger) TContext {
	ctx := klog.NewContext(tc, logger)

	tc = tc.clone()
	tc.Context = ctx
	return tc
}
