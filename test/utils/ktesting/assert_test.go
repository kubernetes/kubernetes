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
	"testing"
	"time"

	"github.com/onsi/gomega"
)

func TestAssert(t *testing.T) {
	for name, tc := range map[string]testcase{
		"eventually-timeout": {
			cb: func(tCtx TContext) {
				Eventually(tCtx, func(tCtx TContext) int {
					// Canceling here is a nop.
					tCtx.Cancel("testing")
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1))
			},
			expectDuration: time.Second,
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: 
Timed out after x.y s.
Expected
    <int>: 0
to equal
    <int>: 1
`,
		},
		"eventually-final": {
			cb: func(tCtx TContext) {
				Eventually(tCtx, func(tCtx TContext) float64 {
					gomega.StopTrying("final error").Now()
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: 0,
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: 
Told to stop trying after x.y s.
final error
`,
		},
		"eventually-error": {
			cb: func(tCtx TContext) {
				Eventually(tCtx, func(tCtx TContext) float64 {
					tCtx.Fatal("some error")
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: time.Second,
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: 
Timed out after x.y s.
The function passed to Eventually returned the following error:
    <*errors.joinError | 0xXXXX>: 
    some error
    {
        errs: [
            <*errors.errorString | 0xXXXX>{s: "some error"},
        ],
    }
`,
		},
		"eventually-success": {
			cb: func(tCtx TContext) {
				Eventually(tCtx, func(tCtx TContext) float64 {
					return 1.0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: 0,
			expectTrace:    ``,
		},
		"eventually-retry": {
			cb: func(tCtx TContext) {
				Eventually(tCtx, func(tCtx TContext) float64 {
					gomega.TryAgainAfter(time.Millisecond).Now()
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: time.Second,
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: 
Timed out after x.y s.
told to try again after 1ms
`,
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
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: 
Failed after x.y s.
Expected
    <float64>: 0
to equal
    <float64>: 1
`,
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
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: 
Told to stop trying after x.y s.
final error
`,
		},
		"consistently-error": {
			cb: func(tCtx TContext) {
				Consistently(tCtx, func(tCtx TContext) float64 {
					tCtx.Fatal("some error")
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: 0,
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: 
Failed after x.y s.
The function passed to Consistently returned the following error:
    <*errors.joinError | 0xXXXX>: 
    some error
    {
        errs: [
            <*errors.errorString | 0xXXXX>{s: "some error"},
        ],
    }
`,
		},
		"consistently-success": {
			cb: func(tCtx TContext) {
				Consistently(tCtx, func(tCtx TContext) float64 {
					return 1.0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: time.Second,
			expectTrace:    ``,
		},
		"consistently-retry": {
			cb: func(tCtx TContext) {
				Consistently(tCtx, func(tCtx TContext) float64 {
					gomega.TryAgainAfter(time.Millisecond).Wrap(errors.New("intermittent error")).Now()
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: time.Second,
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: 
Timed out while waiting on TryAgainAfter after x.y s.
told to try again after 1ms: intermittent error
`,
		},
		"expect-equal": {
			cb: func(tCtx TContext) {
				tCtx.Expect(1).To(gomega.Equal(42))
				tCtx.Log("not reached")
			},
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: 
Expected
    <int>: 1
to equal
    <int>: 42
`,
		},
		"require-equal": {
			cb: func(tCtx TContext) {
				tCtx.Require(1).To(gomega.Equal(42))
				tCtx.Log("not reached")
			},
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: 
Expected
    <int>: 1
to equal
    <int>: 42
`,
		},
		"assert-equal": {
			cb: func(tCtx TContext) {
				tCtx.Assert(1).To(gomega.Equal(42))
				tCtx.Log("reached")
			},
			expectTrace: `(ERROR) <klog header>: ERROR: 
Expected
    <int>: 1
to equal
    <int>: 42
(LOG) <klog header>: reached
`,
		},
		"expect-no-error-success": {
			cb: func(tCtx TContext) {
				tCtx.ExpectNoError(nil)
			},
		},
		"expect-no-error-normal-error": {

			cb: func(tCtx TContext) {
				tCtx.ExpectNoError(errors.New("fake error"))
			},
			expectTrace: `(LOG) <klog header>: Unexpected error:
    <*errors.errorString | 0xXXXX>: 
    fake error
    {s: "fake error"}
(FATAL) <klog header>: FATAL ERROR: Unexpected error: fake error
`,
		},
		"expect-no-error-failure": {
			cb: func(tCtx TContext) {
				tCtx.ExpectNoError(fmt.Errorf("doing something: %w", FailureError{Msg: "fake error"}))
			},
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: doing something: fake error
`,
		},
		"expect-no-error-explanation-string": {
			cb: func(tCtx TContext) {
				tCtx.ExpectNoError(fmt.Errorf("doing something: %w", FailureError{Msg: "fake error"}), "testing error checking")
			},
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: testing error checking: doing something: fake error
`,
		},
		"expect-no-error-explanation-printf": {
			cb: func(tCtx TContext) {
				tCtx.ExpectNoError(fmt.Errorf("doing something: %w", FailureError{Msg: "fake error"}), "testing %s %d checking", "error", 42)
			},
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: testing error 42 checking: doing something: fake error
`,
		},
		"expect-no-error-explanation-callback": {
			cb: func(tCtx TContext) {
				tCtx.ExpectNoError(fmt.Errorf("doing something: %w", FailureError{Msg: "fake error"}), func() string { return "testing error checking" })
			},
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: testing error checking: doing something: fake error
`,
		},
		"expect-no-error-backtrace": {
			cb: func(tCtx TContext) {
				tCtx.ExpectNoError(fmt.Errorf("doing something: %w", FailureError{Msg: "fake error", FullStackTrace: "abc\nxyz"}))
			},
			expectTrace: `(LOG) <klog header>: Failed at:
    abc
    xyz
(FATAL) <klog header>: FATAL ERROR: doing something: fake error
`,
		},
		"expect-no-error-backtrace-and-explanation": {
			cb: func(tCtx TContext) {
				tCtx.ExpectNoError(fmt.Errorf("doing something: %w", FailureError{Msg: "fake error", FullStackTrace: "abc\nxyz"}), "testing error checking")
			},
			expectTrace: `(LOG) <klog header>: testing error checking
(LOG) <klog header>: Failed at:
    abc
    xyz
(FATAL) <klog header>: FATAL ERROR: testing error checking: doing something: fake error
`,
		},

		"output": {
			cb: func(tCtx TContext) {
				tCtx.Log("Log", "a", "b", 42)
				tCtx.Logf("Logf %s %s %d", "a", "b", 42)
				tCtx.Error("Error", "a", "b", 42)
				tCtx.Errorf("Errorf %s %s %d", "a", "b", 42)
			},
			expectTrace: `(LOG) <klog header>: Log a b 42
(LOG) <klog header>: Logf a b 42
(ERROR) <klog header>: ERROR: Error a b 42
(ERROR) <klog header>: ERROR: Errorf a b 42
`,
		},
		"fatal": {
			cb: func(tCtx TContext) {
				tCtx.Fatal("Error", "a", "b", 42)
				// not reached
				tCtx.Log("Log")
			},
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: Error a b 42
`,
		},
		"fatalf": {
			cb: func(tCtx TContext) {
				tCtx.Fatalf("Error %s %s %d", "a", "b", 42)
				// not reached
				tCtx.Log("Log")
			},
			expectTrace: `(FATAL) <klog header>: FATAL ERROR: Error a b 42
`,
		},
	} {
		tc := tc
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			tc.run(t)
		})
	}
}
