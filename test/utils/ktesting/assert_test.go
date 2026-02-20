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
				tCtx.Eventually(func(tCtx TContext) int {
					// Canceling here is a nop.
					tCtx.Cancel("testing")
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1))
			},
			expectDuration: time.Second,
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	Timed out after x.y s.
	Expected
	    <int>: 0
	to equal
	    <int>: 1
`,
		},
		"eventually-final": {
			cb: func(tCtx TContext) {
				tCtx.Eventually(func(tCtx TContext) float64 {
					gomega.StopTrying("final error").Now()
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: 0,
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	Told to stop trying after x.y s.
	final error
`,
		},
		"eventually-error": {
			cb: func(tCtx TContext) {
				tCtx.Eventually(func(tCtx TContext) float64 {
					tCtx.Fatal("some error")
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
				tCtx.Log("not reached")
			},
			expectDuration: time.Second,
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	Timed out after x.y s.
	The function passed to Eventually returned the following error:
	    <ktesting.failures>: 
	    some error
`,
		},
		"assert-eventually-error": {
			cb: func(tCtx TContext) {
				tCtx.AssertEventually(func(tCtx TContext) float64 {
					tCtx.Fatal("some error")
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
				tCtx.Log("reached")
			},
			expectDuration: time.Second,
			expectTrace: `(ERROR) ERROR: <klog header>:
	Timed out after x.y s.
	The function passed to Eventually returned the following error:
	    <ktesting.failures>: 
	    some error
(LOG) <klog header>: reached
`,
		},
		"eventually-no-return-okay": {
			cb: func(tCtx TContext) {
				tCtx.Eventually(func(tCtx TContext) {}).WithTimeout(time.Second).Should(gomega.Succeed())
			},
			expectDuration: 0,
			expectTrace:    ``,
		},
		"eventually-no-return-failure": {
			cb: func(tCtx TContext) {
				tCtx.Eventually(func(tCtx TContext) {
					tCtx.Assert(1).To(gomega.Equal(2))
					tCtx.Assert("hello").To(gomega.Equal("world"))
				}).WithTimeout(time.Second).Should(gomega.Succeed())
			},
			expectDuration: time.Second,
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	Timed out after x.y s.
	Expected success, but got an error:
	    <ktesting.failures>: 
	    Expected
	        <int>: 1
	    to equal
	        <int>: 2
	    Expected
	        <string>: hello
	    to equal
	        <string>: world
`,
		},
		"eventually-return-okay": {
			cb: func(tCtx TContext) {
				tCtx.Eventually(func(tCtx TContext) error { return nil }).WithTimeout(time.Second).Should(gomega.Succeed())
			},
			expectDuration: 0,
			expectTrace:    ``,
		},
		"eventually-return-failure": {
			cb: func(tCtx TContext) {
				tCtx.Eventually(func(tCtx TContext) error {
					tCtx.Assert(1).To(gomega.Equal(2))
					tCtx.Assert("hello").To(gomega.Equal("world"))
					return nil
				}).WithTimeout(time.Second).Should(gomega.Succeed())
			},
			expectDuration: time.Second,
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	Timed out after x.y s.
	Expected success, but got an error:
	    <ktesting.failures>: 
	    Expected
	        <int>: 1
	    to equal
	        <int>: 2
	    Expected
	        <string>: hello
	    to equal
	        <string>: world
`,
		},
		"eventually-success": {
			cb: func(tCtx TContext) {
				tCtx.Eventually(func(tCtx TContext) float64 {
					return 1.0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: 0,
			expectTrace:    ``,
		},
		"eventually-retry": {
			cb: func(tCtx TContext) {
				tCtx.Eventually(func(tCtx TContext) float64 {
					gomega.TryAgainAfter(time.Millisecond).Now()
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: time.Second,
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	Timed out after x.y s.
	told to try again after 1ms
`,
		},
		"consistently-timeout": {
			cb: func(tCtx TContext) {
				tCtx.Consistently(func(tCtx TContext) float64 {
					// Canceling here is a nop.
					tCtx.Cancel("testing")
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: 0,
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	Failed after x.y s.
	Expected
	    <float64>: 0
	to equal
	    <float64>: 1
`,
		},
		"consistently-final": {
			cb: func(tCtx TContext) {
				tCtx.Consistently(func(tCtx TContext) float64 {
					gomega.StopTrying("final error").Now()
					tCtx.FailNow()
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: 0,
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	Told to stop trying after x.y s.
	final error
`,
		},
		"consistently-error": {
			cb: func(tCtx TContext) {
				tCtx.Consistently(func(tCtx TContext) float64 {
					tCtx.Fatal("some error")
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
				tCtx.Log("not reached")
			},
			expectDuration: 0,
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	Failed after x.y s.
	The function passed to Consistently returned the following error:
	    <ktesting.failures>: 
	    some error
`,
		},
		"assert-consistently-error": {
			cb: func(tCtx TContext) {
				tCtx.AssertConsistently(func(tCtx TContext) float64 {
					tCtx.Fatal("some error")
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
				tCtx.Log("reached")
			},
			expectDuration: 0,
			expectTrace: `(ERROR) ERROR: <klog header>:
	Failed after x.y s.
	The function passed to Consistently returned the following error:
	    <ktesting.failures>: 
	    some error
(LOG) <klog header>: reached
`,
		},
		"consistently-success": {
			cb: func(tCtx TContext) {
				tCtx.Consistently(func(tCtx TContext) float64 {
					return 1.0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: time.Second,
			expectTrace:    ``,
		},
		"consistently-retry": {
			cb: func(tCtx TContext) {
				tCtx.Consistently(func(tCtx TContext) float64 {
					gomega.TryAgainAfter(time.Millisecond).Wrap(errors.New("intermittent error")).Now()
					return 0
				}).WithTimeout(time.Second).Should(gomega.Equal(1.0))
			},
			expectDuration: time.Second,
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	Timed out while waiting on TryAgainAfter after x.y s.
	told to try again after 1ms: intermittent error
`,
		},
		"expect-equal": {
			cb: func(tCtx TContext) {
				tCtx.Expect(1).To(gomega.Equal(42))
				tCtx.Log("not reached")
			},
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
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
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
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
			expectTrace: `(ERROR) ERROR: <klog header>:
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
(FATAL) FATAL ERROR: <klog header>:
	Unexpected error: fake error
`,
			// {s: fake error} depends on indentation (https://github.com/onsi/gomega/issues/886).
		},
		"expect-no-error-failure": {
			cb: func(tCtx TContext) {
				tCtx.ExpectNoError(fmt.Errorf("doing something: %w", FailureError{Msg: "fake error"}))
				tCtx.Log("not reached")
			},
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	doing something: fake error
`,
		},
		"assert-no-error-failure": {
			cb: func(tCtx TContext) {
				tCtx.AssertNoError(fmt.Errorf("doing something: %w", FailureError{Msg: "fake error"}))
				tCtx.Log("reached")
			},
			expectTrace: `(ERROR) ERROR: <klog header>:
	doing something: fake error
(LOG) <klog header>: reached
`,
		},
		"expect-no-error-explanation-string": {
			cb: func(tCtx TContext) {
				tCtx.ExpectNoError(fmt.Errorf("doing something: %w", FailureError{Msg: "fake error"}), "testing error checking")
			},
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	testing error checking: doing something: fake error
`,
		},
		"expect-no-error-explanation-printf": {
			cb: func(tCtx TContext) {
				tCtx.ExpectNoError(fmt.Errorf("doing something: %w", FailureError{Msg: "fake error"}), "testing %s %d checking", "error", 42)
			},
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	testing error 42 checking: doing something: fake error
`,
		},
		"expect-no-error-explanation-callback": {
			cb: func(tCtx TContext) {
				tCtx.ExpectNoError(fmt.Errorf("doing something: %w", FailureError{Msg: "fake error"}), func() string { return "testing error checking" })
			},
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	testing error checking: doing something: fake error
`,
		},
		"expect-no-error-backtrace": {
			cb: func(tCtx TContext) {
				tCtx.ExpectNoError(fmt.Errorf("doing something: %w", FailureError{Msg: "fake error", FullStackTrace: "abc\nxyz"}))
			},
			expectTrace: `(LOG) <klog header>: Failed at:
	abc
	xyz
(FATAL) FATAL ERROR: <klog header>:
	doing something: fake error
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
(FATAL) FATAL ERROR: <klog header>:
	testing error checking: doing something: fake error
`,
		},

		"output": {
			cb: func(tCtx TContext) {
				tCtx.Log("Log", "a", "b", 42)
				tCtx.Logf("Logf %s %s %d", "a", "b", 42)
				tCtx.Log("multi\nline")
				tCtx.Logf("multi\n%s", "line")
				tCtx.Error("Error", "a", "b", 42)
				tCtx.Errorf("Errorf %s %s %d", "a", "b", 42)
				tCtx.Logger().Info("Hello", "what", "world")
				tCtx.Logger().Info("Hello", "msg", "multi\nline")
			},
			expectTrace: `(LOG) <klog header>: Log a b 42
(LOG) <klog header>: Logf a b 42
(LOG) <klog header>: multi
	line
(LOG) <klog header>: multi
	line
(ERROR) ERROR: <klog header>:
	Error a b 42
(ERROR) ERROR: <klog header>:
	Errorf a b 42
(LOG) <klog header>: Hello what="world"
(LOG) <klog header>: Hello msg=<
	multi
	line
 >
`,
		},
		"fatal": {
			cb: func(tCtx TContext) {
				tCtx.Fatal("Error", "a", "b", 42)
				// not reached
				tCtx.Log("Log")
			},
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	Error a b 42
`,
		},
		"fatalf": {
			cb: func(tCtx TContext) {
				tCtx.Fatalf("Error %s %s %d", "a", "b", 42)
				// not reached
				tCtx.Log("Log")
			},
			expectTrace: `(FATAL) FATAL ERROR: <klog header>:
	Error a b 42
`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			tc.run(t)
		})
	}
}
