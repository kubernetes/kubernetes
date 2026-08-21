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

// Package ktesting is the main package in the family of different ktesting packages.
// Its documentation covers all aspects of ktesting, including an introduction of
// concepts and details that are implemented in the other variants.
//
// # Examples
//
// The examples in [k8s.io/kubernetes/test/utils/ktesting/examples] are unit tests which
// intentionally contain failures and tests which get stuck. To run them, use:
//
//	go test -timeout=10s -tags example k8s.io/kubernetes/test/utils/ktesting/examples/...
//
// Individual examples will be called out below where applicable.
//
// # Side effects of importing
//
// In contrast to [k8s.io/klog/v2/ktesting] this package is opinionated. Importing it:
//   - adds the -v and -vmodule command line flags
//   - adds a custom formatter to [github.com/onsi/gomega/format] which
//     outputs Kubernetes API structs as YAML
//   - optionally sets the default logging verbosity (see below)
//
// # Generic testing API
//
// [TContext] combines aspects of [testing] (logging, reporting failures,
// sub-tests), [testing/synctest] (running code in a synctest bubble) and
// Ginkgo (timeouts and interrupt handling, progress reporting).
//
// [k8s.io/kubernetes/test/utils/client-go/ktesting] adds type-safe passing
// of all relevant client-go instances through a single TContext parameter,
// similar to [k8s.io/kubernetes/test/e2e/framework.Framework].
//
// Code using only TContext can be included in tests run by `go test` and Ginkgo suites.
// To create a TContext instance in a traditional Go test, use [Init](t).
// [NewTestContext] exists for source-compatibility with [k8s.io/klog/v2/ktesting.NewTestContext].
// To ensure that TContext is used consistently, this code pattern makes the `t` parameter
// inaccessible inside the actual test code:
//
//	func TestSomething(t *testing.T) { testSomething(ktesting.Init(t)) }
//	func testSomething(tCtx ktesting.TContext) { ... }
//
// Go benchmarks are also supported via [Init](b). Benchmark specific methods are not exposed
// via TContext. To get access to them one can use [TContext.TB]:
//
//	b := tCtx.TB().(*testing.B)
//
// To create a TContext instance in a Kubernetes E2E suite, use [k8s.io/kubernetes/test/e2e/framework.Framework.TContext].
//
// # Context + timeout + interrupts
//
// TContext implements [context.Context] and thus can be passed to any function
// which needs a context. To distinguish a normal context from a test context,
// [TContext] instances are typically called `tCtx`.
//
// In plain Go tests, the `go test -timeout` parameter is reflected in the deadline
// associated with the TContext, with one small difference: by default, [DefaultCleanupGracePeriod] =
// 5 second are reserved to run cleanup code before the final timeout kills the test binary.
// This can be configured with [initoption.WithCleanupGracePeriod] in tests with long running
// cleanup code.
//
// CTRL-C = SIGINT is caught and cancels all currently active TContext instances.
//
// Code running inside a test should honor context cancellation and return immediately when
// the context is canceled. If it doesn't, the cleanup code cannot be executed.
// Cleanup code registered with [TContext.CleanupCtx] is given a new TContext which has the
// final deadline.
//
// Determining why a context is canceled or why a test starts to shut down can be tricky.
// ktesting tries to make this more obvious in different ways:
//   - ktesting ensures that [context.Cause] returns a useful description of what
//     caused cancellation or a timeout. Code failing because of those should report
//     the result of that call instead of the less descriptive [context.Canceled] or
//     [context.DeadlineExceeded]. [gomega.Eventually] uses [context.Cause],
//     [k8s.io/apimachinery/pkg/util/wait] doesn't.
//   - Not all code does that and even if it does, the error only gets surfaced
//     after cleanup is complete. Therefore ktesting also logs the cause in an INFO message
//     at the time of cancellation.
//   - When catching SIGINT, "canceling test context: received interrupt signal" is printed.
//     That goes to /dev/console if available where it will be visible also when `go test`
//     was invoked without `-v`. If unavailable, stderr is used as fallback, but that
//     may be buffered by `go test`.
//
// These two examples demonstrate the difference in behavior with and without ktesting:
//
//	go test -tags example -timeout=10s -v k8s.io/kubernetes/test/utils/ktesting/examples/with_ktesting
//	go test -tags example -timeout=10s -v k8s.io/kubernetes/test/utils/ktesting/examples/without_ktesting
//
// You can try interrupting before the 10 second timeout, too.
//
// TContext has similar methods as [context] for creating new TContext instances:
//   - [TContext.WithCancel]
//   - [TContext.WithoutCancel]
//   - [TContext.WithTimeout]
//
// In contrast to [context.WithCancel], [TContext.WithCancel] does not return a separate
// cancel function. Instead, [TContext.Cancel] can be used to cancel the new TContext
// instance. [TContext.Cancel] works in all instances, regardless how they were created.
//
// A context created by [Init] automatically gets canceled as soon as the
// test or sub-test function returns, just like [testing.T.Context]. Use
// [TContext.CleanupCtx] to register cleanup callbacks when a TContext instance
// is needed during cleanup: in contrast to the test's TContext, that TContext
// instance passed to the cleanup callback will not be canceled.
//
// In contrast to [context.WithTimeout], a timeout explanation can be specified
// for [context.Cause] by passing a non-empty string to [TContext.WithTimeout].
//
// # Logging
//
// ktesting provides a per-test structured logger which can be accessed via [TContext.Logger] or
// [klog.FromContext]. Per-test logging is particularly useful in packages which test production
// code which supports contextual logging (https://kubernetes.io/blog/2022/05/25/contextual-logging/#contextual-logging).
// Suppose a package has multiple different tests and the production code logs via traditional
// klog to stderr: when one test fails, `go test` shows *all* log output emitted to stderr
// from any test, not just the output of the failed test.
//
// The per-test logger avoids that by writing log messages via [testing.T.Log].
// That method panics when used after a test has terminated, for example when a
// leaked goroutine logs something. Well-written tests should block until all
// started goroutines have terminated. But this is not always easy and/or not
// always done in existing Kubernetes tests, so TContext is more tolerant and
// redirects log output after test termination to stderr, together with a
// warning about the leaked goroutine.
//
// When using Ginkgo, the output goes to the Ginkgo output stream, which is
// also per-test because each worker process only runs one test at a time.
//
// In both cases, the output format is similar to klog's text format and includes
// time stamp and severity (`[IE]<date> <time>`). The [testing.TB] log methods use the same format.
// The log message is not quoted to make structured and unstructured output
// similar to normal test output.
//
// [TContext.Error] always includes "ERROR" in the header to make test failures stand out
// more. [TContext.Fatal] includes "FATAL ERROR". Spyglass then highlights that line
// in the job output, which isn't guaranteed when using [testing.T.Error]: it depends on
// whether the failure text contains one of the keywords recognized by Spyglass.
// For example, `t.Fatal(err)` don't make it clear that the error string
// is the test failure.
//
//	go test -tags example -v -run=TestFormat k8s.io/kubernetes/test/utils/ktesting/examples/logging -args -v=1
//	=== RUN   TestFormat
//	    example_test.go:57: I0820 14:33:51.450589] hello via tCtx.Logf (unstructured logging): x is 1
//	    example_test.go:58: I0820 14:33:51.450605] hello via tCtx.Logger().Info (structured logging) x=1
//	    example_test.go:59: ERROR: I0820 14:33:51.450614]
//	                some thing
//	--- FAIL: TestFormat (0.00s)
//
// Post-processing that output in [k8s.io/kubernetes/cmd/prune-junit-xml] is able to
// distinguish log output from failures because of the header and removes log
// output from the failure message. Output written via [testing.T.Log] cannot be
// removed. Long-term, https://go.dev/doc/go1.27#go-test may solve this, but we are not
// there yet.
//
// During init, the KTESTING_VERBOSITY env variable is set as value for the
// klog verbosity as if `-v=${KTESTING_VERBOSITY}` had been used. This is
// useful in CI jobs which run a large collection of tests where some but not
// all tests have that command line flag. Individual tests can do the same by
// calling [SetDefaultVerbosity]. The env variable has a higher priority than
// such a per-test default.  An actual `-v` parameter has the highest priority.
//
// Note that KTESTING_VERBOSITY and SetDefaultVerbosity immediately
// reconfigures the klog verbosity, already before flag parsing. If the
// verbosity becomes non-zero during init, then other init functions might
// start logging where normally they wouldn't log anything. Should this occur,
// then the right fix is to remove those log calls because logging during init
// is discouraged. It leads to unpredictable output (init functions cannot
// assume that logging is configured) and/or is useless (logging not
// initialized during init and thus conditional log output gets omitted).
//
// With [initoption.BufferLogs] it is possible to create a TContext which
// buffers the log output. This is useful for tests which need to verify that
// some code produces the expected log output.
//
// In benchmarks it might be more desirable to let the production code under
// testing use the original klog. [initoption.PerTestOutput] can be used to
// disable the per-test logger in the TContext instance; see its doc comment
// for its default and an example.
// # TODO: automatically do the right thing in scheduler-perf
//
// # Error checking in helper code
//
// There are two different approaches for implementing checks in helper code,
// with different pros and cons. ktesting has additional support for both of them:
//  1. Return an error as in plain Go, fail with an assertion in the main test code.
//  2. Let helper code use assertions which fail the test.
//
// The first approach needs more code for error checking. Producing informative
// errors also needs more code than calling some assertion helper package (check
// the condition, then construct the error). The advantage is that developers are
// familiar with error wrapping, which (at least in Kubernetes) is suggested by
// linter hints, so typically there will be a trail of where the error came
// from. However, that trail is just plain text which then needs to be mapped
// back to the source code.
//
// [NewFailure] creates a [FailureError] at the source of the problem and adds
// the stack backtrace at that point to the error. Gomega assert failures can
// be turned into such an error (see next section).
//
// [TContext.ExpectNoError] and [TContext.AssertNoError]
// have special support for this error and errors wrapping it:
//   - skip the "unexpected error" prefix that it normally adds for other errors
//   - log the stack backtrace captured at the time of the original failure,
//     without including it in the failure message itself
//
// Direct calls to [TContext.Error], [TContext.Fatal] and their variants (including
// indirectly through Gomega assertions) do not log a stack backtrace: `go test`
// already prints the source code location of the failing call because ktesting
// marks all of these methods as helpers. Only [TContext.ExpectNoError] and
// [TContext.AssertNoError] add a backtrace, because there the location of the
// failure (recorded earlier, when the [FailureError] was created) is different
// from the location where it gets reported.
//
// The second approach makes tests and helpers simpler. One downside is that
// failure texts don't include additional context about the situation in which
// the failure occurred. For example, the loop variable when called repeatedly
// or the complete object when calling a helper for one field of it are often
// useful additional information.
//
// [TContext.WithStep] allows adding some text to all log output and error messages that
// get emitted via the TContext instances returned by WithStep. In this example, "bake cake"
// and "set heat for baking" are passed to two different WithStep calls and "oven not found" to
// [TContext.Fatal]. There are two "FATAL ERRORs" because a cleanup function also fails:
//
//	go test -tags example -v -run=TestWithStep k8s.io/kubernetes/test/utils/ktesting/examples/logging
//	=== RUN   TestWithStep
//	    baking_test.go:29: I0821 17:08:55.019097] bake cake/set heat for baking: Log()
//	    baking_test.go:30: I0821 17:08:55.019114] bake cake/set heat for baking: Logger().Info()
//	    baking_test.go:31: FATAL ERROR: I0821 17:08:55.019128]
//	                bake cake/set heat for baking: oven not found
//	    baking_test.go:37: FATAL ERROR: I0821 17:08:55.019143]
//	                turning off oven not implemented
//	--- FAIL: TestWithStep (0.00s)
//
// The two approaches can be mixed. [TContext.WithError] enables calling a helper function
// which use assertions in another helper function which is supposed to return an error.
//
//	=== RUN   TestWithError
//	    example_test.go:92: I0821 15:01:50.576258] checking oven temperature: failed at:
//	                k8s.io/kubernetes/test/utils/ktesting/examples/logging.checkTemperature({{0x725d20, 0x30e55d1e2bd0}, {{0x72ca58, 0x30e55d1b4b48}}, 0x7201e8, 0x30e55cfd6d50, {0x0, 0x0}, 0x0, {0x0, ...}, ...}, ...)
//	                        /nvme/gopath/src/k8s.io/kubernetes/test/utils/ktesting/examples/logging/example_test.go:100 +0x67
//	                k8s.io/kubernetes/test/utils/ktesting/examples/logging.TestWithError(0x30e55d1b4b48?)
//	                        /nvme/gopath/src/k8s.io/kubernetes/test/utils/ktesting/examples/logging/example_test.go:92 +0x12b
//	    example_test.go:92: ERROR: I0821 15:01:50.576273]
//	                checking oven temperature: oven temperature 42°C is too low for baking
//	    example_test.go:93: I0821 15:01:50.576325] checking oven readiness: failed at:
//	                k8s.io/kubernetes/test/utils/ktesting/examples/logging.checkOvenReady({{0x725d20, 0x30e55d1e2bd0}, {{0x72ca58, 0x30e55d1b4b48}}, 0x7201e8, 0x30e55cfd6d50, {0x0, 0x0}, 0x0, {0x0, ...}, ...}, ...)
//	                        /nvme/gopath/src/k8s.io/kubernetes/test/utils/ktesting/examples/logging/example_test.go:110 +0x306
//	                k8s.io/kubernetes/test/utils/ktesting/examples/logging.TestWithError(0x30e55d1b4b48?)
//	                        /nvme/gopath/src/k8s.io/kubernetes/test/utils/ktesting/examples/logging/example_test.go:93 +0x205
//	    example_test.go:93: ERROR: I0821 15:01:50.576338]
//	                checking oven readiness: oven is not ready yet
//	    baking_test.go:29: I0821 15:01:50.576343] Log()
//	    baking_test.go:30: I0821 15:01:50.576351] Logger().Info()
//	    example_test.go:94: I0821 15:01:50.576384] baking cake: failed at:
//	                k8s.io/kubernetes/test/utils/ktesting/examples/logging.heatOven({{0x725d20, 0x30e55d1e2bd0}, {{0x72ca58, 0x30e55d1b4b48}}, 0x7201e8, 0x30e55cfd6d50, {0x0, 0x0}, 0x0, {0x0, ...}, ...})
//	                        /nvme/gopath/src/k8s.io/kubernetes/test/utils/ktesting/examples/logging/baking_test.go:31 +0x1a5
//	                k8s.io/kubernetes/test/utils/ktesting/examples/logging.bakeCake({{0x725d20, 0x30e55d1e2bd0}, {{0x72ca58, 0x30e55d1b4b48}}, 0x7201e8, 0x30e55cfd6d50, {0x0, 0x0}, 0x0, {0x0, ...}, ...})
//	                        /nvme/gopath/src/k8s.io/kubernetes/test/utils/ktesting/examples/logging/example_test.go:118 +0x22d
//	                k8s.io/kubernetes/test/utils/ktesting/examples/logging.TestWithError(0x30e55d1b4b48?)
//	                        /nvme/gopath/src/k8s.io/kubernetes/test/utils/ktesting/examples/logging/example_test.go:94 +0x2db
//	    example_test.go:94: ERROR: I0821 15:01:50.576390]
//	                baking cake: oven not found
//	    baking_test.go:37: FATAL ERROR: I0821 15:01:50.576397]
//	                turning off oven not implemented
//	--- FAIL: TestWithError (0.00s)
//
// Even with this support in ktesting, the downside of the approach with
// assertions is that the caller must be aware. WithStep should be used
// pro-actively similar to the normal error wrapping and WithError must be used
// when failing the test is not desired.
//
// # Assertions
//
// Gomega can be used to write assertions via [TContext.Expect], its [TContext.Require] alias, or
// via [TContext.Assert]. As the names imply and consistent with testify, Expect and Require abort
// the test in case of a failure while Assert continues:
//
//	import g "github.com/onsi/gomega"
//	tCtx.Assert(2).To(g.Equal(1)) // Logs test failure and continues.
//	tCtx.Expect(1).To(g.Equal(1)) // Is reached.
//
// Note that this does not work in Ginkgo suites because Ginkgo only supports
// one test failure per test, so there all variants abort immediately.
//
// [TContext.Eventually] and [TContext.Consistently] support polling with
// callbacks that take a TContext as parameter. They abort testing when
// they fail. [TContext.AssertEventually] and [TContext.AssertConsistently]
// continue. The TContext in the callback includes the Eventually/Consistently
// timeout.
//
// While polling, sending the Go test binary
// a SIGUSR1 will cause ktesting to print a summary of the current situation,
// which includes the reason why Gomega is still polling, prefixed with the
// name of the test that the polling belongs to, as well as the names of all
// currently running tests and sub-tests. This feature is also supported by
// Ginkgo in E2E suites. Compared to the implementation in Ginkgo, the one in
// ktesting is still pretty rudimentary and, for example, lacks source code
// backtraces.
//
// The following example runs two parallel sub-tests which both get stuck
// in an Eventually call. The progress report lists the parent test and both
// sub-tests as currently running, then shows each sub-test's own
// explanation, indented and prefixed by the name of the sub-test that it
// belongs to:
//
//	go test -tags example -timeout=20s k8s.io/kubernetes/test/utils/ktesting/examples/with_ktesting & pid=$!; sleep 5; killall -USR1 with_ktesting.test; wait $pid
//	...
//	You requested a progress report.
//	Currently running:
//	        TestTimeout
//	        TestTimeout/baking
//	        TestTimeout/heating
//
//	TestTimeout/heating:
//	        waiting for oven to reach baking temperature
//	        Expected
//	            <int>: 1
//	        to equal
//	            <int>: 2
//
//	TestTimeout/baking:
//	        waiting for cake to be done
//	        Expected
//	            <int>: 1
//	        to equal
//	            <int>: 2
//	--- FAIL: TestTimeout (0.00s)
//	    example_test.go:36: I0821 16:19:33.413274] Using "/tmp/TestTimeout2324884183/001" as temporary directory.
//	    example_test.go:41: Will fail shortly before the test suite deadline at 2026-08-21 16:19:53.413099637 +0200 CEST m=+20.000483031.
//	    contexthelper.go:75:
//	        INFO: canceling context: test suite deadline (2026-08-21 16:19:53 +0200 CEST) is close, need to clean up before the 5s cleanup grace period
//
//	    --- FAIL: TestTimeout/heating (15.00s)
//	        example_test.go:48: FATAL ERROR: I0821 16:19:48.414372]
//	                Context was cancelled (cause: test suite deadline (2026-08-21 16:19:53 +0200 CEST) is close, need to clean up before the 5s cleanup grace period) after 15.001s.
//	                waiting for oven to reach baking temperature
//	                Expected
//	                    <int>: 1
//	                to equal
//	                    <int>: 2
//	    --- FAIL: TestTimeout/baking (15.00s)
//	        example_test.go:55: FATAL ERROR: I0821 16:19:48.414730]
//	                Context was cancelled (cause: test suite deadline (2026-08-21 16:19:53 +0200 CEST) is close, need to clean up before the 5s cleanup grace period) after 15.001s.
//	                waiting for cake to be done
//	                Expected
//	                    <int>: 1
//	                to equal
//	                    <int>: 2
//	    example_test.go:38: Cleaning up...
//	FAIL
//	FAIL    k8s.io/kubernetes/test/utils/ktesting/examples/with_ktesting    15.007s
//
// Normally, raising an assertion inside a polling callback is wrong. In the following example,
// a failed assertion aborts the test instead of triggering a retry:
//
//	gomega.Eventually(ctx, func(ctx context.Context) int {
//	   ...
//	   gomega.Expect(...).To(...) // Aborts the test without retrying.
//	   ..
//	}).Should(...)
//
// This is not a problem with ktesting's Gomega integration because the assertion failure
// is turned into an error under the hood. The TContext given to the
// callback must be used for this to work:
//
//	tCtx.Eventually(func(tCtx ktesting.TContext) int {
//	  ...
//	  tCtx.Expect(...).To(...) // Returns early, polling continues.
//	  ...
//	  tCtx.Fatal(...) // Also would return early without stopping polling.
//	  ...
//	}).Should(...)
//
// [github.com/stretchr/testify] can be used by passing the TContext instance as first parameter:
//
//	import "github.com/stretchr/testify/require"
//	require.Equal(tCtx, 1, 1)
//
// Gomega gets special support in ktesting for several reasons:
//   - Its design makes the integration into TContext described above possible.
//   - It's possible to tweak object formatting to use YAML.
//   - In contrast to testify's Equal, it's always clear which parameter is the expected
//     and which is the actual value.
//   - Failure messages are more consistent than in testify.
//   - Gomega has more acceptable dependencies than testify.
//     In particular it does not depend on go-spew, an unwanted dependency in Kubernetes.
//
// # Defining tests
//
// [Init] is only necessary in the top-level Test functions. Sub-tests can be
// defined with [TContext.Run] using callbacks that directly get a TContext:
//
//	func TestSomething(t *testing.T) { testSomething(ktesting.Init(t)) }
//	func testSomething(tCtx ktesting.TContext) {
//	  tCtx.Run("a", func(tCtx ktesting.TContext) { ...})
//	  tCtx.Run("b", func(tCtx ktesting.TContext) { ...})
//	}
//
// [TContext.Parallel] marks the current test as one that runs in parallel
// with other parallel tests, exactly like [testing.T.Parallel]. It must be
// called inside each (sub-)test which is meant to run in parallel with its
// siblings:
//
//	func testSomething(tCtx ktesting.TContext) {
//	  tCtx.Run("a", func(tCtx ktesting.TContext) {
//	    tCtx.Parallel()
//	    ...
//	  })
//	  tCtx.Run("b", func(tCtx ktesting.TContext) {
//	    tCtx.Parallel()
//	    ...
//	  })
//	}
//
// A [testing/synctest] bubble can be created with [TContext.SyncTest] either
// in the currently running test or directly when defining a sub test:
//
//	func TestA(t *testing.T) { ktesting.Init(t).SyncTest("", testA) }
//	func testA(tCtx ktesting.TContext) { ... }
//
//	func TestB(t *testing.T) { testA(ktesting.Init(t)) }
//	func testB(tCtx ktesting.TContext) {
//	  tCtx.SyncTest("1", testB1)
//	  tCtx.SyncTest("2", testB1)
//	}
//	func testB1(tCtx ktesting.TContext) { ... }
//	func testB2(tCtx ktesting.TContext) { ... }
//
// [TContext.Wait] is a shortcut for [testing/synctest.Wait]. It panics when called outside
// of a bubble. [TContext.IsSyncTest] can be used to check for that.
//
// Sub-tests and synctest are not supported in a Ginkgo suite and panic when
// called there.
package ktesting
