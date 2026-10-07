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
	"errors"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/onsi/gomega"
)

// cleanupDrivenContextTB embeds *testing.T for all the boilerplate TB
// methods, but reimplements Cleanup and Context to mimic Ginkgo's testing
// proxy: Context() creates a context and registers its own cancel func via
// Cleanup instead of canceling it independently when the test is done.
// Cleanup callbacks run in LIFO order, exactly like both "testing" and
// Ginkgo do it.
type cleanupDrivenContextTB struct {
	*testing.T

	mu       sync.Mutex
	cleanups []func()
}

func (tb *cleanupDrivenContextTB) Cleanup(f func()) {
	tb.mu.Lock()
	defer tb.mu.Unlock()
	tb.cleanups = append(tb.cleanups, f)
}

// runCleanups simulates the end of the test: all callbacks registered via
// Cleanup run now, in LIFO order.
func (tb *cleanupDrivenContextTB) runCleanups() {
	for {
		tb.mu.Lock()
		if len(tb.cleanups) == 0 {
			tb.mu.Unlock()
			return
		}
		f := tb.cleanups[len(tb.cleanups)-1]
		tb.cleanups = tb.cleanups[:len(tb.cleanups)-1]
		tb.mu.Unlock()
		f()
	}
}

func (tb *cleanupDrivenContextTB) Context() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	tb.Cleanup(cancel)
	return ctx
}

// TestRunWhenDoneCleanupDrivenContext ensures that there's no deadlock
// with TB implementations (like Ginkgo's) which cancel their Context() from
// their own Cleanup instead of independently, which runs after (LIFO order)
// runWhenDone's own Cleanup and thus could have blocked it forever.
func TestRunWhenDoneCleanupDrivenContext(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		fake := &cleanupDrivenContextTB{T: t}

		var called bool
		runWhenDone(fake, func() {
			called = true
		})

		// If this deadlocks, synctest reports it as a test failure.
		fake.runCleanups()

		if !called {
			t.Error("runWhenDone's callback was not invoked")
		}
	})
}

func TestCleanupErr(t *testing.T) {
	actual := cleanupErr(t.Name())
	if !errors.Is(actual, context.Canceled) {
		t.Errorf("cleanupErr %T should be a %T", actual, context.Canceled)
	}
}

func TestCause(t *testing.T) {
	timeoutCause := canceledError("I timed out")
	cancelText := "I got canceled"
	var cancelCause error = canceledError(cancelText)
	parentCause := errors.New("parent canceled")

	contextBackground := func(t *testing.T) context.Context {
		return context.Background()
	}

	for name, tt := range map[string]struct {
		parentCtx              func(t *testing.T) context.Context
		timeout                time.Duration
		sleep                  time.Duration
		cancelCause            *error
		cancelText             *string
		expectErr, expectCause error
		expectDeadline         time.Duration
	}{
		"nothing": {
			parentCtx: contextBackground,
			timeout:   5 * time.Millisecond,
			sleep:     time.Millisecond,
		},
		"timeout": {
			parentCtx:   contextBackground,
			timeout:     time.Millisecond,
			sleep:       5 * time.Millisecond,
			expectErr:   context.Canceled,
			expectCause: canceledError(timeoutCause),
		},
		"parent-canceled": {
			parentCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return ctx
			},
			timeout:     time.Millisecond,
			sleep:       5 * time.Millisecond,
			expectErr:   context.Canceled,
			expectCause: context.Canceled,
		},
		"parent-cause": {
			parentCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithCancelCause(context.Background())
				cancel(parentCause)
				return ctx
			},
			timeout:     time.Millisecond,
			sleep:       5 * time.Millisecond,
			expectErr:   context.Canceled,
			expectCause: parentCause,
		},
		"deadline-no-parent": {
			parentCtx:      contextBackground,
			timeout:        time.Minute,
			expectDeadline: time.Minute,
		},
		"deadline-parent": {
			parentCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
				t.Cleanup(cancel)
				return ctx
			},
			timeout:        2 * time.Minute,
			expectDeadline: time.Minute,
		},
		"deadline-child": {
			parentCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
				t.Cleanup(cancel)
				return ctx
			},
			timeout:        time.Minute,
			expectDeadline: time.Minute,
		},
		"cancelCause": {
			parentCtx:   contextBackground,
			cancelCause: &cancelCause,
			expectErr:   context.Canceled,
			expectCause: cancelCause,
		},
		"cancelText": {
			parentCtx:   contextBackground,
			cancelText:  &cancelText,
			expectErr:   context.Canceled,
			expectCause: cancelCause,
		},
		"cancelEmptyText": {
			// Canceling with an empty string must preserve the standard
			// context.Canceled cause, just like context.CancelFunc does.
			parentCtx:   contextBackground,
			cancelText:  new(string),
			expectErr:   context.Canceled,
			expectCause: context.Canceled,
		},
		"cancelBecauseNil": {
			// CancelBecause is a transparent pass-through to
			// context.CancelCauseFunc, so nil must also result in the
			// standard context.Canceled cause.
			parentCtx:   contextBackground,
			cancelCause: new(error),
			expectErr:   context.Canceled,
			expectCause: context.Canceled,
		},
	} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				tCtx := Init(t)
				tCtx = tCtx.WithContext(tt.parentCtx(t)).WithTimeout(tt.timeout, timeoutCause.Error())
				if tt.cancelCause != nil {
					tCtx.CancelBecause(*tt.cancelCause)
				}
				if tt.cancelText != nil {
					tCtx.Cancel(*tt.cancelText)
				}
				if tt.expectDeadline != 0 {
					actualDeadline, ok := tCtx.Deadline()
					if !ok {
						tCtx.Error("should have a deadline and hasn't")
					} else {
						tCtx.Assert(time.Until(actualDeadline)).To(gomega.Equal(tt.expectDeadline), "remaining time till Deadline()")
					}
				}
				// Unblock background goroutines.
				time.Sleep(tt.sleep)
				// Wait for them to do their work.
				synctest.Wait()
				// Now check.
				actualErr := tCtx.Err()
				actualCause := context.Cause(tCtx)
				if tt.expectErr == nil {
					tCtx.Assert(actualErr).To(gomega.Succeed(), "ctx.Err()")
				} else {
					tCtx.Assert(actualErr).To(gomega.MatchError(tt.expectErr), "ctx.Err()")
				}
				if tt.expectCause == nil {
					tCtx.Assert(actualCause).To(gomega.Succeed(), "context.Cause()")
				} else {
					tCtx.Assert(actualCause).To(gomega.MatchError(tt.expectCause), "context.Cause()")
				}
			})
		})
	}
}

// TestCancel checks how cancellation propagates or doesn't propagate
// when setting up child contexts through WithCancel or WithoutCancel.
func TestCancel(t *testing.T) {
	tCtx := Init(t)
	tCtx2 := tCtx.WithoutCancel()
	tCtx3 := tCtx.WithoutCancel()
	tCtx4 := tCtx.WithCancel()
	tCtx5 := tCtx.WithCancel()

	tCtx.AssertNoError(tCtx.Err())
	tCtx.AssertNoError(tCtx2.Err())
	tCtx.AssertNoError(tCtx3.Err())
	tCtx.AssertNoError(tCtx4.Err())
	tCtx.AssertNoError(tCtx5.Err())

	tCtx2.Cancel("cancel 2")
	tCtx.AssertNoError(tCtx.Err())
	tCtx.Assert(context.Cause(tCtx2)).To(gomega.MatchError(gomega.ContainSubstring("cancel 2")))
	tCtx.AssertNoError(tCtx3.Err())
	tCtx.AssertNoError(tCtx4.Err())
	tCtx.AssertNoError(tCtx5.Err())

	tCtx4.Cancel("cancel 4")
	tCtx.AssertNoError(tCtx.Err())
	tCtx.Assert(context.Cause(tCtx2)).To(gomega.MatchError(gomega.ContainSubstring("cancel 2")))
	tCtx.AssertNoError(tCtx3.Err())
	tCtx.Assert(context.Cause(tCtx4)).To(gomega.MatchError(gomega.ContainSubstring("cancel 4")))
	tCtx.AssertNoError(tCtx5.Err())

	tCtx.Cancel("cancel root")
	tCtx.Assert(context.Cause(tCtx)).To(gomega.MatchError(gomega.ContainSubstring("cancel root")))
	tCtx.Assert(context.Cause(tCtx2)).To(gomega.MatchError(gomega.ContainSubstring("cancel 2")))
	tCtx.AssertNoError(tCtx3.Err())
	tCtx.Assert(context.Cause(tCtx4)).To(gomega.MatchError(gomega.ContainSubstring("cancel 4")))
	tCtx.Assert(context.Cause(tCtx5)).To(gomega.MatchError(gomega.ContainSubstring("cancel root")))
}
