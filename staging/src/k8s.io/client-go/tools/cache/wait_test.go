/*
Copyright The Kubernetes Authors.

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

package cache

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/textlogger"
)

func init() {
	// The test below is sensitive to the time zone, log output uses time.Local.
	time.Local = time.UTC
}

func TestWaitFor(t *testing.T) {
	for name, tc := range map[string]struct {
		what          string
		checkers      []DoneChecker
		timeout       time.Duration
		timeoutReason string

		expectDone bool

		// Time is predictable and starts at the synctest epoch.
		// %[1]d is the pid, %[2]d the line number of the WaitFor call.
		expectOutput string
	}{
		"empty": {
			expectDone: true,
		},
		"no-caches": {
			what:       "my-caches",
			expectDone: true,
			expectOutput: `I0101 00:00:00.000000 %7[1]d wait_test.go:%[2]d] "Waiting" for="my-caches"
`,
		},
		"no-logging": {
			checkers:   []DoneChecker{newMockChecker("first", 10*time.Second), newMockChecker("second", 5*time.Second), newMockChecker("last", 0*time.Second)},
			expectDone: true,
		},
		"with-logging": {
			what:       "my-caches",
			checkers:   []DoneChecker{newMockChecker("first", 10*time.Second), newMockChecker("second", 5*time.Second), newMockChecker("last", 0*time.Second)},
			expectDone: true,
			expectOutput: `I0101 00:00:00.000000 %7[1]d wait_test.go:%[2]d] "Waiting" for="my-caches"
I0101 00:00:00.000000 %7[1]d wait_test.go:%[2]d] "Done waiting" for="my-caches" instance="last"
I0101 00:00:05.000000 %7[1]d wait_test.go:%[2]d] "Done waiting" for="my-caches" instance="second"
I0101 00:00:10.000000 %7[1]d wait_test.go:%[2]d] "Done waiting" for="my-caches" instance="first"
`,
		},
		"some-timeout": {
			timeout:    time.Minute,
			what:       "my-caches",
			checkers:   []DoneChecker{newMockChecker("first", 10*time.Second), newMockChecker("second", -1), newMockChecker("last", 0*time.Second)},
			expectDone: false,
			expectOutput: `I0101 00:00:00.000000 %7[1]d wait_test.go:%[2]d] "Waiting" for="my-caches"
I0101 00:00:00.000000 %7[1]d wait_test.go:%[2]d] "Done waiting" for="my-caches" instance="last"
I0101 00:00:10.000000 %7[1]d wait_test.go:%[2]d] "Done waiting" for="my-caches" instance="first"
I0101 00:01:00.000000 %7[1]d wait_test.go:%[2]d] "Timed out waiting" for="my-caches" cause="context deadline exceeded" instances=["second"]
`,
		},
		"some-canceled": {
			timeout:    -1,
			what:       "my-caches",
			checkers:   []DoneChecker{newMockChecker("first", 10*time.Second), newMockChecker("second", -1), newMockChecker("last", 0*time.Second)},
			expectDone: false,
			expectOutput: `I0101 00:00:00.000000 %7[1]d wait_test.go:%[2]d] "Waiting" for="my-caches"
I0101 00:00:00.000000 %7[1]d wait_test.go:%[2]d] "Done waiting" for="my-caches" instance="last"
I0101 00:00:00.000000 %7[1]d wait_test.go:%[2]d] "Timed out waiting" for="my-caches" cause="context canceled" instances=["first","second"]
`,
		},
		"more": {
			timeoutReason: "go fish",
			timeout:       5 * time.Second,
			what:          "my-caches",
			checkers:      []DoneChecker{newMockChecker("first", 10*time.Second), newMockChecker("second", -1), newMockChecker("last", 0*time.Second)},
			expectDone:    false,
			expectOutput: `I0101 00:00:00.000000 %7[1]d wait_test.go:%[2]d] "Waiting" for="my-caches"
I0101 00:00:00.000000 %7[1]d wait_test.go:%[2]d] "Done waiting" for="my-caches" instance="last"
I0101 00:00:05.000000 %7[1]d wait_test.go:%[2]d] "Timed out waiting" for="my-caches" cause="go fish" instances=["first","second"]
`,
		},
		"all": {
			timeout:    time.Minute,
			what:       "my-caches",
			checkers:   []DoneChecker{newMockChecker("first", -1), newMockChecker("second", -1), newMockChecker("last", -1)},
			expectDone: false,
			expectOutput: `I0101 00:00:00.000000 %7[1]d wait_test.go:%[2]d] "Waiting" for="my-caches"
I0101 00:01:00.000000 %7[1]d wait_test.go:%[2]d] "Timed out waiting" for="my-caches" cause="context deadline exceeded" instances=["first","last","second"]
`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				var buffer bytes.Buffer
				logger := textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(&buffer)))
				ctx := klog.NewContext(context.Background(), logger)
				var wg sync.WaitGroup
				defer wg.Wait()
				if tc.timeout != 0 {
					switch tc.timeoutReason {
					case "":
						if tc.timeout > 0 {
							c, cancel := context.WithTimeout(ctx, tc.timeout)
							defer cancel()
							ctx = c
						} else {
							c, cancel := context.WithCancel(ctx)
							cancel()
							ctx = c
						}
					default:
						c, cancel := context.WithCancelCause(ctx)
						wg.Go(func() {
							time.Sleep(tc.timeout)
							cancel(errors.New(tc.timeoutReason))
						})
						ctx = c
					}
				}
				_, _, line, _ := runtime.Caller(0)
				done := WaitFor(ctx, tc.what, tc.checkers...)
				expectOutput := tc.expectOutput
				if expectOutput != "" {
					expectOutput = fmt.Sprintf(expectOutput, os.Getpid(), line+1)
				}
				assert.Equal(t, tc.expectDone, done, "done")
				assert.Equal(t, expectOutput, buffer.String(), "output")
			})
		})
	}
}

// newMockChecker can be created outside of a synctest bubble.
// It constructs the channel inside when Done is first called.
func newMockChecker(name string, delay time.Duration) DoneChecker {
	return &mockChecker{
		name:  name,
		delay: delay,
	}
}

type mockChecker struct {
	name        string
	delay       time.Duration
	initialized bool
	done        <-chan struct{}
}

func (m *mockChecker) Name() string { return m.name }
func (m *mockChecker) Done() <-chan struct{} {
	if !m.initialized {
		switch {
		case m.delay > 0:
			// In the future.
			ctx := context.Background()
			// This leaks a cancel, but is hard to avoid (cannot use the parent t.Cleanup, no other way to delay calling it). Doesn't matter in a unit test.
			//nolint:govet
			ctx, _ = context.WithTimeout(ctx, m.delay)
			m.done = ctx.Done()
		case m.delay == 0:
			// Immediately.
			c := make(chan struct{})
			close(c)
			m.done = c
		default:
			// Never.
			c := make(chan struct{})
			m.done = c
		}
		m.initialized = true
	}
	return m.done
}

func TestSyncResult(t *testing.T) {
	for name, tc := range map[string]struct {
		result        SyncResult
		expectAsError string
	}{
		"empty": {},
		"one": {
			result: SyncResult{
				Err: errors.New("my custom cancellation reason"),
				Synced: map[reflect.Type]bool{
					reflect.TypeFor[int]():    true,
					reflect.TypeFor[string](): false,
				},
			},
			expectAsError: "failed to sync all caches: string: my custom cancellation reason",
		},
		"many": {
			result: SyncResult{
				Err: errors.New("my custom cancellation reason"),
				Synced: map[reflect.Type]bool{
					reflect.TypeFor[int]():    false,
					reflect.TypeFor[string](): false,
				},
			},
			expectAsError: "failed to sync all caches: int, string: my custom cancellation reason",
		},
	} {

		t.Run(name, func(t *testing.T) {
			actual := tc.result.AsError()
			switch {
			case tc.expectAsError == "" && actual != nil:
				t.Fatalf("expected no error, got %v", actual)
			case tc.expectAsError != "" && actual == nil:
				t.Fatalf("expected %q, got no error", actual)
			case tc.expectAsError != "" && actual != nil && actual.Error() != tc.expectAsError:
				t.Fatalf("expected %q, got %q", tc.expectAsError, actual.Error())
			}
			if tc.result.Err != nil && !errors.Is(actual, tc.result.Err) {
				t.Errorf("actual error %+v should wrap %v but doesn't", actual, tc.result.Err)
			}
		})
	}
}
