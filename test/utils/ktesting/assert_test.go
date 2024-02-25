/*
Copyright 2024 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"regexp"
	"testing"
	"time"

	"github.com/onsi/gomega"
	"github.com/stretchr/testify/assert"
)

func TestAsync(t *testing.T) {
	for name, tc := range map[string]struct {
		cb             func(TContext)
		expectNoFail   bool
		expectError    string
		expectDuration time.Duration
	}{
		"eventually-timeout": {
			cb: func(tCtx TContext) {
				Eventually(tCtx, func(tCtx TContext) int {
					// Canceling here is a nop.
					tCtx.Cancel("testing")
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1))
			},
			expectDuration: time.Second,
			expectError: `Timed out after x.y s.
Expected
    <int>: 0
to equal
    <int>: 1`,
		},
		"eventually-final": {
			cb: func(tCtx TContext) {
				Eventually(tCtx, func(tCtx TContext) float64 {
					gomega.StopTrying("final error").Now()
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: 0,
			expectError: `Told to stop trying after x.y s.
final error`,
		},
		"eventually-error": {
			cb: func(tCtx TContext) {
				Eventually(tCtx, func(tCtx TContext) float64 {
					tCtx.Fatal("some error")
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: time.Second,
			expectError: `Timed out after x.y s.
The function passed to Eventually returned the following error:
    <*errors.joinError | 0xXXXX>: 
    some error
    {
        errs: [
            <*errors.errorString | 0xXXXX>{s: "some error"},
        ],
    }`,
		},
		"eventually-success": {
			cb: func(tCtx TContext) {
				Eventually(tCtx, func(tCtx TContext) float64 {
					return 1.0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: 0,
			expectNoFail:   true,
			expectError:    ``,
		},
		"eventually-retry": {
			cb: func(tCtx TContext) {
				Eventually(tCtx, func(tCtx TContext) float64 {
					gomega.TryAgainAfter(time.Millisecond).Now()
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: time.Second,
			expectError: `Timed out after x.y s.
told to try again after 1ms`,
		},
		"consistently-timeout": {
			cb: func(tCtx TContext) {
				Consistently(tCtx, func(tCtx TContext) float64 {
					// Canceling here is a nop.
					tCtx.Cancel("testing")
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: 0,
			expectError: `Failed after x.y s.
Expected
    <float64>: 0
to equal
    <float64>: 1`,
		},
		"consistently-final": {
			cb: func(tCtx TContext) {
				Consistently(tCtx, func(tCtx TContext) float64 {
					gomega.StopTrying("final error").Now()
					tCtx.FailNow()
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: 0,
			expectError: `Told to stop trying after x.y s.
final error`,
		},
		"consistently-error": {
			cb: func(tCtx TContext) {
				Consistently(tCtx, func(tCtx TContext) float64 {
					tCtx.Fatal("some error")
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: 0,
			expectError: `Failed after x.y s.
The function passed to Consistently returned the following error:
    <*errors.joinError | 0xXXXX>: 
    some error
    {
        errs: [
            <*errors.errorString | 0xXXXX>{s: "some error"},
        ],
    }`,
		},
		"consistently-success": {
			cb: func(tCtx TContext) {
				Consistently(tCtx, func(tCtx TContext) float64 {
					return 1.0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: time.Second,
			expectNoFail:   true,
			expectError:    ``,
		},
		"consistently-retry": {
			cb: func(tCtx TContext) {
				Consistently(tCtx, func(tCtx TContext) float64 {
					gomega.TryAgainAfter(time.Millisecond).Wrap(errors.New("intermittent error")).Now()
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: time.Second,
			expectError: `Timed out while waiting on TryAgainAfter after x.y s.
told to try again after 1ms: intermittent error`,
		},
	} {
		tc := tc
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			tCtx := Init(t)
			var err error
			tCtx, finalize := WithError(tCtx, &err)
			start := time.Now()
			func() {
				defer finalize()
				tc.cb(tCtx)
			}()
			duration := time.Since(start)
			assert.InDelta(t, tc.expectDuration.Seconds(), duration.Seconds(), 0.1, fmt.Sprintf("callback invocation duration %s", duration))
			assert.Equal(t, !tc.expectNoFail, tCtx.Failed(), "Failed()")
			if tc.expectError == "" {
				assert.NoError(t, err)
			} else if assert.NotNil(t, err) {
				t.Logf("Result:\n%s", err.Error())
				errMsg := err.Error()
				errMsg = regexp.MustCompile(`[[:digit:]]+\.[[:digit:]]+s`).ReplaceAllString(errMsg, "x.y s")
				errMsg = regexp.MustCompile(`0x[[:xdigit:]]+`).ReplaceAllString(errMsg, "0xXXXX")
				assert.Equal(t, tc.expectError, errMsg)
			}
		})
	}
}
