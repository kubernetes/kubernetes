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
	"testing"
	"testing/synctest"
	"time"

	"github.com/onsi/gomega"
)

func TestCleanupErr(t *testing.T) {
	actual := cleanupErr(t.Name())
	if !errors.Is(actual, context.Canceled) {
		t.Errorf("cleanupErr %T should be a %T", actual, context.Canceled)
	}
}

func TestCause(t *testing.T) {
	timeoutCause := canceledError("I timed out")
	parentCause := errors.New("parent canceled")

	contextBackground := func(t *testing.T) context.Context {
		return context.Background()
	}

	for name, tt := range map[string]struct {
		parentCtx              func(t *testing.T) context.Context
		timeout                time.Duration
		sleep                  time.Duration
		cancelCause            string
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
	} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				tCtx := Init(t)
				ctx, cancel := withTimeout(tt.parentCtx(t), t, tt.timeout, timeoutCause.Error())
				if tt.cancelCause != "" {
					cancel(tt.cancelCause)
				}
				if tt.expectDeadline != 0 {
					actualDeadline, ok := ctx.Deadline()
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
				actualErr := ctx.Err()
				actualCause := context.Cause(ctx)
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
