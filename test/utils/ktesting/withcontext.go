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

	"github.com/onsi/gomega"
	"k8s.io/klog/v2"
)

// WithCancel sets up cancellation in a [TContext.Cleanup] callback and
// constructs a new TContext where [TContext.Cancel] cancels only the new
// context.
func WithCancel(tCtx TContext) TContext {
	ctx, cancel := context.WithCancelCause(tCtx)
	tCtx.Cleanup(func() {
		cancel(cleanupErr(tCtx.Name()))
	})

	return withContext{
		TContext: tCtx,
		Context:  ctx,
		cancel: func(cause string) {
			var cancelCause error
			if cause != "" {
				cancelCause = canceledError(cause)
			}
			cancel(cancelCause)
		},
	}
}

// WithTimeout sets up new context with a timeout. Canceling the timeout gets
// registered in a [TContext.Cleanup] callback. [TContext.Cancel] cancels only
// the new context. The cause is used as reason why the context is canceled
// once the timeout is reached. It may be empty, in which case the usual
// "context canceled" error is used.
func WithTimeout(tCtx TContext, timeout time.Duration, timeoutCause string) TContext {
	tCtx.Helper()
	ctx, cancel := withTimeout(tCtx, tCtx.TB(), timeout, timeoutCause)

	return withContext{
		TContext: tCtx,
		Context:  ctx,
		cancel:   cancel,
	}
}

// WithLogger constructs a new context with a different logger.
func WithLogger(tCtx TContext, logger klog.Logger) TContext {
	ctx := klog.NewContext(tCtx, logger)

	return withContext{
		TContext: tCtx,
		Context:  ctx,
		cancel:   tCtx.Cancel,
	}
}

// withContext combines some TContext with a new [context.Context] derived
// from it. Because both provide the [context.Context] interface, methods must
// be defined which pick the newer one.
type withContext struct {
	TContext
	context.Context

	cancel func(cause string)
}

func (wCtx withContext) Cancel(cause string) {
	wCtx.cancel(cause)
}

func (wCtx withContext) CleanupCtx(cb func(TContext)) {
	wCtx.Helper()
	cleanupCtx(wCtx, cb)
}

func (wCtx withContext) Expect(actual interface{}, extra ...interface{}) gomega.Assertion {
	wCtx.Helper()
	return expect(wCtx, actual, extra...)
}

func (wCtx withContext) ExpectNoError(err error, explain ...interface{}) {
	wCtx.Helper()
	expectNoError(wCtx, err, explain...)
}

func (cCtx withContext) Run(name string, cb func(tCtx TContext)) bool {
	return run(cCtx, name, cb)
}

func (wCtx withContext) Logger() klog.Logger {
	return klog.FromContext(wCtx)
}

func (wCtx withContext) Deadline() (time.Time, bool) {
	return wCtx.Context.Deadline()
}

func (wCtx withContext) Done() <-chan struct{} {
	return wCtx.Context.Done()
}

func (wCtx withContext) Err() error {
	return wCtx.Context.Err()
}

func (wCtx withContext) Value(key any) any {
	return wCtx.Context.Value(key)
}
