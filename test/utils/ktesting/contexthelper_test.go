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
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
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

	t.Parallel()
	for name, tt := range map[string]struct {
		parentCtx              context.Context
		timeout                time.Duration
		sleep                  time.Duration
		cancelCause            string
		expectErr, expectCause error
		expectDeadline         time.Duration
	}{
		"nothing": {
			parentCtx: context.Background(),
			timeout:   5 * time.Millisecond,
			sleep:     time.Millisecond,
		},
		"timeout": {
			parentCtx:   context.Background(),
			timeout:     time.Millisecond,
			sleep:       5 * time.Millisecond,
			expectErr:   context.Canceled,
			expectCause: canceledError(timeoutCause),
		},
		"parent-canceled": {
			parentCtx: func() context.Context {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return ctx
			}(),
			timeout:     time.Millisecond,
			sleep:       5 * time.Millisecond,
			expectErr:   context.Canceled,
			expectCause: context.Canceled,
		},
		"parent-cause": {
			parentCtx: func() context.Context {
				ctx, cancel := context.WithCancelCause(context.Background())
				cancel(parentCause)
				return ctx
			}(),
			timeout:     time.Millisecond,
			sleep:       5 * time.Millisecond,
			expectErr:   context.Canceled,
			expectCause: parentCause,
		},
		"deadline-no-parent": {
			parentCtx:      context.Background(),
			timeout:        time.Minute,
			expectDeadline: time.Minute,
		},
		"deadline-parent": {
			parentCtx: func() context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
				t.Cleanup(cancel)
				return ctx
			}(),
			timeout:        2 * time.Minute,
			expectDeadline: time.Minute,
		},
		"deadline-child": {
			parentCtx: func() context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
				t.Cleanup(cancel)
				return ctx
			}(),
			timeout:        time.Minute,
			expectDeadline: time.Minute,
		},
	} {
		tt := tt
		t.Run(name, func(t *testing.T) {
			ctx, cancel := withTimeout(tt.parentCtx, t, tt.timeout, timeoutCause.Error())
			if tt.cancelCause != "" {
				cancel(tt.cancelCause)
			}
			if tt.expectDeadline != 0 {
				actualDeadline, ok := ctx.Deadline()
				if assert.True(t, ok, "should have had a deadline") {
					// Testing timing behavior is unreliable in Prow because
					// the test runs in parallel with several others.
					// Therefore this check is skipped if a CI environment is
					// detected.
					ci, _ := os.LookupEnv("CI")
					switch strings.ToLower(ci) {
					case "yes", "true", "1":
						// Skip.
					default:
						assert.InDelta(t, time.Until(actualDeadline), tt.expectDeadline, float64(time.Second), "remaining time till Deadline()")
					}
				}
			}
			time.Sleep(tt.sleep)
			actualErr := ctx.Err()
			actualCause := context.Cause(ctx)
			ci, _ := os.LookupEnv("CI")
			switch strings.ToLower(ci) {
			case "yes", "true", "1":
				// Skip.
			default:
				assert.Equal(t, tt.expectErr, actualErr, "ctx.Err()")
				assert.Equal(t, tt.expectCause, actualCause, "context.Cause()")
			}
		})
	}
}
