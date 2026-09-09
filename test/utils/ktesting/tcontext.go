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
	"flag"
	"fmt"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/test/utils/ktesting/format"
	"k8s.io/kubernetes/test/utils/ktesting/initoption"
	"k8s.io/kubernetes/test/utils/ktesting/internal"
)

// Underlier is the additional interface implemented by the per-test LogSink
// behind [TContext.Logger]. Together with [initoption.BufferLogs] it can be
// used to capture log output in memory to check it in tests.
type Underlier = ktesting.Underlier

// DefaultCleanupGracePeriod is the time that a [TContext] gets canceled before
// the deadline of its underlying test suite (usually determined via "go test
// -timeout"). This gives the running test(s) time to fail with an informative
// timeout error. After that, all cleanup callbacks then have the remaining
// time to complete before the test binary is killed.
//
// For this to work, each blocking call in a test must respect the
// cancellation of the [TContext].
//
// When using Ginkgo to manage the test suite and running tests, the
// cleanup grace period is ignored because Ginkgo itself manages timeouts.
//
// This value can be overridden per-[TContext] via [initoption.WithCleanupGracePeriod].
const DefaultCleanupGracePeriod = 5 * time.Second

// TB is the interface common to [testing.T], [testing.B], [testing.F] and
// [github.com/onsi/ginkgo/v2] which ktesting relies upon itself or
// passes through (like Chdir, TempDir).
//
// In contrast to [testing.TB], it can be implemented also outside of the
// testing package.
//
// Fatal/Error/Skip are used by ktesting instead of Log + FailNow/Failed/SkipNow
// because when implemented by Ginkgo it is better to pass the reason for
// a failure or skip directly to Ginkgo in the method intended for that purpose.
type TB interface {
	Attr(key, value string)
	Chdir(dir string)
	Cleanup(func())
	Error(args ...any)
	Errorf(format string, args ...any)
	Fail()
	FailNow()
	Failed() bool
	Fatal(args ...any)
	Fatalf(format string, args ...any)
	Helper()
	Log(args ...any)
	Logf(format string, args ...any)
	Name() string
	Setenv(key, value string)
	Skip(args ...any)
	Skipf(format string, args ...any)
	SkipNow()
	Skipped() bool
	TempDir() string
}

// ContextTB adds support for cleanup callbacks with explicit context
// parameter. This is used when integrating with Ginkgo: then CleanupCtx
// gets implemented via ginkgo.DeferCleanup.
type ContextTB interface {
	TB
	CleanupCtx(func(ctx context.Context))
}

// TContext implements [context.Context], [testing.TB] and some additional
// methods. [TContext] is the public pointer type for referencing a TC.
// Variables are usually called tCtx. To ensure that test code does not
// use `t` directly unintentionally, it is recommended to use two functions:
//
//	func TestSomething(t *testing.T) { testSomething(ktesting.Init(t)) }
//	func testSomething(tCtx ktesting.TContext) { ... }
//
// Log output is associated with the current test and includes a header similar
// to klog, which enables post-processing to distinguish between log output
// (starts with header) and failure messages (header comes later). Errors
// ([Error], [Errorf]) are recorded with "ERROR" as prefix, fatal errors
// ([Fatal], [Fatalf]) with "FATAL ERROR". Indention is used to ensure that
// follow-up lines belonging to the same log entry can be handled properly
// by post-processing and to make the header stand out more.
//
// tCtx provides features offered by Ginkgo also when using normal Go [testing]:
//   - The context contains a deadline that expires soon enough before
//     the overall timeout that cleanup code can still run.
//   - Cleanup callbacks can get their own, separate contexts when
//     registered via [CleanupCtx].
//   - CTRL-C aborts, prints a progress report, and then cleans up
//     before terminating.
//   - SIGUSR1 prints a progress report without aborting.
//
// Progress reporting is more informative when doing polling with
// [gomega.Eventually] and [gomega.Consistently]. Without that, it
// can only report which tests are active.
type TContext struct {
	// Context makes the methods of the underlying context
	// available. It must not be modified.
	context.Context

	// testingTB makes the methods of the underlying test implementation
	// available. Its embedded TB must not be modified.
	testingTB

	// perTestHeader is an optional function which produces a klog-like perTestHeader when
	// not using some global logger.
	perTestHeader func() string

	// for Cancel
	cancel func(cause error)

	// steps is a concatenation ("step1/step2/step3: ") of steps passed to WithStep.
	// It's empty if there are no steps.
	steps string

	// for IsSyncTest
	isSyncTest bool

	// for WithNamespace
	namespace string

	// capture, if non-nil, changes Error/Errorf/Fatal/Fatalf/Fail/FailNow so
	// that they intercept the problem and convert to errors. Log messages
	// are passed through.
	//
	// Used by WithError.
	capture *capture

	// cleanupGracePeriod is the effective cleanup grace period for this context.
	// It is always non-zero: both Init and InitCtx resolve it from the supplied
	// options or fall back to DefaultCleanupGracePeriod.
	cleanupGracePeriod time.Duration
}

type capture struct {
	mutex  sync.Mutex
	errors []error
	failed bool
}

// testingTB is needed to avoid a name conflict
// between field and method in tContext.
type testingTB struct {
	// TB makes the methods of the underlying test implementation available.
	// In particular Helper must be called directly, not via a wrapper.
	// It must not be modified.
	TB
}

// InitOption is an alias for the options provided through the [initoption] package.
//
// They are in a separate package to separate the main API from the less
// commonly used configuration and to simplify auto-completion.
type InitOption = initoption.InitOption

// Init can be called in a unit or integration test to create
// a test context which:
// - has a per-test logger with verbosity derived from the -v command line flag
// - gets canceled automatically when the test ends
//
// Note that "test ends" is defined as "main test function returned".
// In other words, the test execution order is:
//   - test function
//   - all defer callbacks in that function
//   - automatic TContext cancellation is triggered (i.e. not guaranteed to
//     be instantaneous, but cleanup callbacks can rely on it to happen)
//   - all [Cleanup] and [CleanupCtx] callbacks in LIFO order;
//     they can still log to tb and record additional failures
//   - underlying tb gets finalized, making it unusable
//
// Note that the test context supports the interfaces of [TB] and
// [context.Context] and thus can be used like one of those where needed.
// It also has additional methods for retrieving the logger and canceling
// the context early, which can be useful in tests which want to wait
// for goroutines to terminate after cancellation.
//
// If the [TB] implementation also implements [ContextTB], then
// [TContext.CleanupCtx] uses [ContextTB.CleanupCtx] and uses
// the context passed into that callback. This can be used to let
// Ginkgo create a fresh context for cleanup code.
//
// The default behavior described above can be
// modified via optional functional options defined in [initoption].
// Can be called more than once per test to get different contexts with
// independent cancellation: Cancel only affects the instance it is
// called on. The automatic cancellation is triggered for all instances
// when the test ends (as defined above), regardless of when Init was called.
// Each instance can have a different configuration.
//
// Can be called inside a synctest bubble. Signal handling (cleaning up on
// SIGINT, progress reporting on SIGUSR1) then does not get initialized because
// code running inside a bubble should not depend on outside input. Progress
// reporting still works when some parent test already initialized it.
// Therefore the recommended pattern is to initialize ktesting first, then
// create the synctest bubble:
//
//	func TestSomething(t *testing.T) { ktesting.Init(t).SyncTest("", testSomething) }
//	func testSomething(tCtx ktesting.TContext) { ... }
//
// This pattern also has the advantage that the test code cannot accidentally
// use the testing.T instance directly. The same works for normal tests:
//
//	func TestSomething(t *testing.T) { testSomething(ktesting.Init(t)) }
//	func testSomething(tCtx ktesting.TContext) { ... }
func Init(tb TB, opts ...InitOption) TContext {
	tb.Helper()

	c := internal.InitConfig{
		PerTestOutput: true,
	}
	for _, opt := range opts {
		opt(&c)
	}

	// Resolve the effective cleanup grace period: use the caller-supplied value
	// if positive, otherwise fall back to DefaultCleanupGracePeriod.
	gracePeriod := c.CleanupGracePeriod
	if gracePeriod <= 0 {
		gracePeriod = DefaultCleanupGracePeriod
	}

	isSyncTest, deadline := analyzeTB(tb)

	ctx := defaultProgressReporter.init(tb, isSyncTest)
	var header func() string
	if c.PerTestOutput {
		logger := newLogger(tb, c.BufferLogs)
		ctx = klog.NewContext(ctx, logger)
		header = klogHeader
	}

	var cancelTimeout func(cause error)
	if deadline != nil {
		timeLeft := time.Until(*deadline)
		timeLeft -= gracePeriod
		ctx, cancelTimeout = withTimeout(ctx, tb, timeLeft, fmt.Sprintf("test suite deadline (%s) is close, need to clean up before the %s cleanup grace period", deadline.Truncate(time.Second), gracePeriod))
	}

	// Construct new TContext with context and settings as determined above.
	tCtx := newTContext(ctx, tb, gracePeriod, isSyncTest)
	tCtx.isSyncTest = isSyncTest
	if cancelTimeout != nil {
		tCtx.cancel = cancelTimeout
	} else {
		tCtx = tCtx.WithCancel()
		runWhenDone(tb, func() {
			tCtx.Cancel(cleanupErr(tCtx.Name()).Error())
		})
	}
	tCtx.perTestHeader = header

	return tCtx
}

var timeNow = time.Now // Can be stubbed out for testing.

func klogHeader() string {
	now := timeNow()
	_, month, day := now.Date()
	hour, minute, second := now.Clock()
	return fmt.Sprintf("I%02d%02d %02d:%02d:%02d.%06d]",
		month, day, hour, minute, second, now.Nanosecond()/1000)
}

// InitCtx is a variant of [Init] which uses an already existing context and
// whatever logger and timeouts are stored there.
func InitCtx(ctx context.Context, tb TB, opts ...InitOption) TContext {
	tb.Helper()
	c := internal.InitConfig{}
	for _, opt := range opts {
		opt(&c)
	}
	gracePeriod := c.CleanupGracePeriod
	if gracePeriod <= 0 {
		gracePeriod = DefaultCleanupGracePeriod
	}
	isSyncTest, _ := analyzeTB(tb)
	defaultProgressReporter.init(tb, isSyncTest)
	return newTContext(ctx, tb, gracePeriod, isSyncTest)
}

func newTContext(ctx context.Context, tb TB, gracePeriod time.Duration, isSyncTest bool) TContext {
	return TContext{
		Context:            ctx,
		testingTB:          testingTB{TB: tb},
		cleanupGracePeriod: gracePeriod,
		isSyncTest:         isSyncTest,
	}
}

func analyzeTB(tb TB) (isSyncTest bool, deadline *time.Time) {
	// We don't need a Deadline implementation, testing.B doesn't have it.
	// But if we have one, we use it to determine the deadline and
	// set a timeout shortly before it.
	//
	// This also allows us to detect a synctest bubble.
	if deadlineTB, deadlineOK := tb.(interface {
		Deadline() (time.Time, bool)
	}); deadlineOK {
		func() {
			defer func() {
				// Calling testing.T.Deadline panics inside a synctest bubble.
				// There's no API to detect that in advance, so here we react
				// by catching the panic.
				if r := recover(); r != nil {
					isSyncTest = true
				}
			}()
			if d, ok := deadlineTB.Deadline(); ok {
				deadline = &d
			}
		}()
	}

	return
}

func newLogger(tb TB, bufferLogs bool) klog.Logger {
	config := ktesting.NewConfig(
		ktesting.AnyToString(func(v interface{}) string {
			// For basic types where the string
			// representation is "obvious" we use
			// fmt.Sprintf because format.Object always
			// adds a <"type"> prefix, which is too long
			// for simple values.
			switch v := v.(type) {
			case int, int32, int64, uint, uint32, uint64, float32, float64, bool:
				return fmt.Sprintf("%v", v)
			case string:
				return v
			default:
				return strings.TrimSpace(format.Object(v, 1))
			}
		}),
		ktesting.VerbosityFlagName("v"),
		ktesting.VModuleFlagName("vmodule"),
		ktesting.BufferLogs(bufferLogs),
	)

	// Copy klog settings instead of making the ktesting logger
	// configurable directly.
	var fs flag.FlagSet
	config.AddFlags(&fs)
	for _, name := range []string{"v", "vmodule"} {
		from := flag.CommandLine.Lookup(name)
		to := fs.Lookup(name)
		if err := to.Value.Set(from.Value.String()); err != nil {
			panic(err)
		}
	}

	// Ensure consistent logging: this klog.Logger writes to tb, adding the
	// date/time header, and our own wrapper emulates that behavior for
	// Log/Logf/...
	logger := ktesting.NewLogger(tb, config)
	return logger
}

// withTB constructs a new TContext with a different TB instance.
//
// This is used internally to set up some of the context, in particular
// clients, in the root test and then run sub-tests:
//
//	func TestSomething(t *testing.T) {
//	   tCtx := ktesting.Init(t)
//	   ...
//	   tCtx = ktesting.WithRESTConfig(tCtx, config)
//
//	   t.Run("sub", func (t *testing.T) {
//	       tCtx := ktesting.WithTB(tCtx, t)
//	       ...
//	   })
//
// withTB sets up cancellation for the sub-test and uses per-test output.
//
// Like [Init], it cancels the returned TContext automatically when the
// sub-test ends.
func (tCtx TContext) withTB(tb TB) TContext {
	tCtx.testingTB.TB = tb
	if tCtx.perTestHeader != nil {
		logger := newLogger(tb, false /* don't buffer logs in sub-test */)
		tCtx.Context = klog.NewContext(tCtx.Context, logger)
	}

	// Sub-tests don't go through Init, so without this call they
	// wouldn't show up in the "Currently running" list of a
	// progress report.
	defaultProgressReporter.trackRunningTest(tb)

	// Cancellation for sync tests has to be handled differently,
	// see run below.
	tCtx = tCtx.WithCancel()
	if !tCtx.isSyncTest {
		runWhenDone(tb, func() {
			tCtx.Cancel(cleanupErr(tCtx.Name()).Error())
		})
	}
	return tCtx
}

// WithContext constructs a new TContext with a different Context instance.
// This can be used in callbacks which receive a Context, for example
// from Gomega:
//
//	gomega.Eventually(tCtx, func(ctx context.Context) {
//	   tCtx := ktesting.WithContext(tCtx, ctx)
//	   ...
//
// Cancellation and deadline are determined by the new context.
// Values are looked up first in the new context, then the old one.
// In other words, values set previous via WithValue are still
// available.
func (tCtx TContext) WithContext(ctx context.Context) TContext {
	tCtx.Context = &chainContext{Context: ctx, previousCtx: tCtx.Context}
	return tCtx
}

type chainContext struct {
	context.Context
	previousCtx context.Context
}

func (ctx *chainContext) Value(key any) any {
	if val := ctx.Context.Value(key); val != nil {
		return val
	}
	return ctx.previousCtx.Value(key)
}

// WithValue wraps [context.WithValue] such that the result is again a TContext.
func (tCtx TContext) WithValue(key, val any) TContext {
	ctx := context.WithValue(tCtx.Context, key, val)
	return tCtx.WithContext(ctx)
}

// WithCancel sets up cancellation in a [TContext.Cleanup] callback and
// constructs a new TContext where [TContext.Cancel] cancels only the new
// context.
func (tCtx TContext) WithCancel() TContext {
	ctx, cancel := context.WithCancelCause(tCtx.Context)

	tCtx.Context = ctx
	tCtx.cancel = cancel
	return tCtx
}

// WithoutCancel causes the returned context to ignore cancellation of its parent.
// Calling Cancel will only cancel the new context.
// This matches [context.WithoutCancel].
func (tCtx TContext) WithoutCancel() TContext {
	ctx := context.WithoutCancel(tCtx.Context)

	tCtx.Context = ctx
	tCtx = tCtx.WithCancel() // Re-create a cancelable TContext.
	return tCtx
}

// WithTimeout sets up new context with a timeout. Canceling the timeout gets
// registered in a cleanup callback. [TContext.Cancel] cancels only
// the new context. The cause is used as reason why the context is canceled
// once the timeout is reached. It may be empty, in which case the usual
// "context canceled" error is used.
func (tCtx TContext) WithTimeout(timeout time.Duration, timeoutCause string) TContext {
	ctx, cancel := withTimeout(tCtx.Context, tCtx.TB(), timeout, timeoutCause)

	tCtx.Context = ctx
	tCtx.cancel = cancel
	return tCtx
}

// Parallel signals that this test is to be run in parallel with (and
// only with) other parallel tests. In other words, it needs to be
// called in each test which is meant to run in parallel.
//
// Only supported in Go unit tests, calling it elsewhere causes a test failure.
//
// When a unit test is run multiple times due to use of -test.count or -test.cpu,
// multiple instances of a single test never run in parallel with each other.
func (tCtx TContext) Parallel() {
	if tb, ok := tCtx.TB().(interface{ Parallel() }); ok {
		tb.Parallel()
	} else {
		tCtx.Fatalf("Parallel not implemented, underlying %T does not support it", tCtx.TB())
	}
}

// Cancel can be invoked to cancel the context before the test is completed.
// Tests which use the context to control goroutines and then wait for
// termination of those goroutines must call Cancel to avoid a deadlock.
//
// The cause, if non-empty, is turned into an error which is equivalent
// to context.Canceled. context.Cause will return that error for the
// context.
func (tCtx TContext) Cancel(cause string) {
	if tCtx.cancel != nil {
		var cancelCause error
		if cause != "" {
			cancelCause = canceledError(cause)
		}
		// nil is okay here, context.Canceled will be used instead.
		tCtx.cancel(cancelCause)
	}
}

// CancelBecause is like Cancel except that it directly uses
// the provided error. It is up to the caller whether that
// error is a context.Canceled error.
func (tCtx TContext) CancelBecause(cause error) {
	if tCtx.cancel != nil {
		tCtx.cancel(cause)
	}
}

// CleanupCtx registers a callback that will get invoked when the test
// has finished. Callbacks get invoked in last-in-first-out order (LIFO).
//
// Using CleanupCtx is preferred because of the automatic context cancellation.
// The following broken (!) cleanup code will use a canceled context,
// which is not desirable:
//
//	// tCtx gets canceled when the test ends and before the callback runs.
//	tCtx.Cleanup(func() { /* do something with the test's tCtx */ })
//
// A safer way to run cleanup code is:
//
//	tCtx.CleanupCtx(func (tCtx ktesting.TContext) { /* do something with the cleanup's tCtx */ })
//
// The logger and clients are the same as in the TContext that CleanupCtx
// is invoked on.
func (tCtx TContext) CleanupCtx(cb func(TContext)) {
	tCtx.Helper()

	if tb, ok := tCtx.TB().(ContextTB); ok {
		// Use context from base TB (most likely Ginkgo).
		tb.CleanupCtx(func(ctx context.Context) {
			tCtx := tCtx.WithContext(ctx)
			cb(tCtx)
		})
		return
	}

	tCtx.Cleanup(func() {
		// Use new context. This is the code path for "go test". The
		// context then has *no* deadline. In the code path above for
		// Ginkgo, Ginkgo is more sophisticated and also applies
		// timeouts to cleanup calls which accept a context.
		childCtx := tCtx.WithContext(context.WithoutCancel(tCtx.Context))
		cb(childCtx)
	})
}

// Run runs cb as a subtest called name. It blocks until cb returns or
// calls t.Parallel to become a parallel test.
//
// Only supported in Go unit tests or benchmarks. It fails the current
// test when called elsewhere.
func (tCtx TContext) Run(name string, cb func(tCtx TContext)) bool {
	return tCtx.run(name, false, cb)
}

// SyncTest uses [synctest.Test] to execute the callback inside a bubble.
// Creates a new subtest if the name is non-empty, otherwise it creates
// the bubble directly in the current test context.
//
// Only works in Go unit tests.
//
// Cleaning up on SIGINT is not available because code running inside a bubble
// should not depend on outside input.
func (tCtx TContext) SyncTest(name string, cb func(tCtx TContext)) bool {
	return tCtx.run(name, true, cb)
}

func (tCtx TContext) run(name string, syncTest bool, cb func(tCtx TContext)) bool {
	tCtx.Helper()
	switch tb := tCtx.TB().(type) {
	case *testing.T:
		if syncTest {
			f := func(t *testing.T) {
				// We must not propagate the parent's
				// cancellation channel into the bubble,
				// it causes "panic: receive on synctest channel from outside bubble".
				//
				// Sync tests shouldn't need the overall suite timeout,
				// so this seems okay.
				tCtx.isSyncTest = true
				tCtx = tCtx.WithoutCancel().withTB(t)
				// runWhenDone's context.AfterFunc runs in a new goroutine,
				// which synctest has no reason to schedule before its
				// deadlock check fires the instant cb returns. Cancel
				// synchronously here instead.
				defer tCtx.Cancel(cleanupErr(tCtx.Name()).Error())
				cb(tCtx)
			}
			if name != "" {
				return tb.Run(name, func(t *testing.T) { synctest.Test(t, f) })
			}
			synctest.Test(tb, f)
			return true
		}
		return tb.Run(name, func(t *testing.T) {
			cb(tCtx.withTB(t))
		})
	case *testing.B:
		if !syncTest {
			return tb.Run(name, func(b *testing.B) {
				cb(tCtx.withTB(b))
			})
		}
	}

	what := "Run"
	if syncTest {
		what = "SyncTest"
	}
	tCtx.Fatalf("%s not implemented, underlying %T does not support it", what, tCtx.TB())

	return false
}

// IsSyncTest returns true if the context was created by SyncTest.
//
// Inside such a context, Wait is usable. This can be used in
// code which runs inside synctest bubbles and outside:
//   - Inside a bubble, Wait can be used to block until
//     background activity has settled down (= "durably blocked").
//     Eventually and Consistently both call Wait and then check
//     the condition.
//   - Outside, polling or some synchronization mechanism has to be used.
func (tCtx TContext) IsSyncTest() bool {
	return tCtx.isSyncTest
}

// Wait calls [synctest.Wait] and thus ensures that all background
// activity has settled down (= "durably blocked").
//
// Only works inside a bubble started by SyncTest (can be checked with
// IsSyncTest), panics elsewhere.
func (tCtx TContext) Wait() {
	synctest.Wait()
}

// TB returns the underlying TB. This can be used to "break the glass"
// and cast back into a testing.T or TB. Calling TB is necessary
// because TContext wraps the underlying TB.
//
// For example, in benchmarks it is necessary to cast back to
// a testing.B because not all benchmark-only methods are provided.
// This examples uses separate functions again to ensure that
// b isn't used unintentionally:
//
//	func BenchmarkSomething(b *testing.B) { benchmarkSomething(ktesting.Init(b) }
//	func benchmarkSomething(tCtx ktesting.TContext) {
//	    ... set up with tCtx ...
//	    for tCtx.TB().(*testing.B).Loop() {
//	        ...
//	    }
//	}
func (tCtx TContext) TB() TB { return tCtx.testingTB.TB }

// WithLogger constructs a new context with a different logger.
func (tCtx TContext) WithLogger(logger klog.Logger) TContext {
	ctx := klog.NewContext(tCtx.Context, logger)

	tCtx.Context = ctx
	return tCtx
}

// Logger returns a logger for the current test. This is a shortcut
// for calling klog.FromContext.
//
// Output emitted via this logger and the TB interface (like Logf)
// is formatted consistently. The TB interface generates a single
// message string, while Logger enables structured logging and can
// be passed down into code which expects a logger.
//
// To skip intermediate helper functions during stack unwinding,
// TB.Helper can be called in those functions.
func (tCtx TContext) Logger() klog.Logger {
	return klog.FromContext(tCtx.Context)
}

// WithNamespace creates a new context with a Kubernetes namespace name for retrieval through [Namespace].
func (tCtx TContext) WithNamespace(namespace string) TContext {
	tCtx.namespace = namespace
	return tCtx
}

// Namespace returns the Kubernetes namespace name that was set previously
// through WithNamespace and the empty string if none is available.
//
// This namespace is the one to be used by tests which need to create namespace-scoped
// objects. The name is guaranteed to be unique for the test context, so tests running
// in parallel need to be set up so that each test has its own namespace.
func (tCtx TContext) Namespace() string {
	return tCtx.namespace
}

// WithError creates a context where test failures are collected and stored in
// the provided error instance when the caller is done. Use it like this:
//
//	func doSomething(tCtx ktesting.TContext) (finalErr error) {
//	     tCtx, finalize := WithError(tCtx, &finalErr)
//	     defer finalize()
//	     ...
//	     tCtx.Fatal("some failure")
//
// Any error already stored in the variable will get overwritten by finalize if
// there were test failures, otherwise the variable is left unchanged.
// If there were multiple test errors, then the error will wrap all of
// them with errors.Join.
//
// Test failures are not propagated to the parent context.
// WithRESTConfig initializes all client-go clients with new clients
// created for the config. The current test name gets included in the UserAgent.
func (tCtx TContext) WithError(err *error) (TContext, func()) {
	tCtx.capture = &capture{}

	return tCtx, func() {
		// Recover has to be called in the deferred function. When called inside
		// a function called by a deferred function (like finalize below), it
		// returns nil.
		if e := recover(); e != nil {
			if _, ok := e.(fatalWithError); !ok {
				// Not our own panic, pass it on instead of setting the error.
				panic(e)
			}
		}

		tCtx.finalize(err)
	}
}

func (tCtx TContext) finalize(err *error) {
	tCtx.capture.mutex.Lock()
	defer tCtx.capture.mutex.Unlock()

	errs := tCtx.capture.errors
	if tCtx.capture.failed && len(errs) == 0 {
		errs = []error{errFailedWithNoExplanation}
	}
	if len(errs) == 0 {
		return
	}
	*err = failures{errors.Join(errs...)}
}

type failures struct {
	error
}

// Unwrap gives errors.Is and errors.As access to the individual errors
// joined together by errors.Join in finalize. Embedding only promotes
// Error() because the embedded field has the static type error, which
// doesn't declare Unwrap, so it has to be forwarded explicitly here.
func (e failures) Unwrap() []error {
	if joined, ok := e.error.(interface{ Unwrap() []error }); ok {
		return joined.Unwrap()
	}
	return nil
}

func (e failures) GomegaString() string {
	// We don't need to repeat the string. Errors already get formatted once by Gomega itself,
	// then it calls GomegaString for a summary that isn't necessary anymore.
	return ""
}

// WithStep creates a context where a prefix is added to all errors and log
// messages, similar to how errors are wrapped. This can be nested, leaving a
// trail of "bread crumbs" that help figure out where in a test some problem
// occurred or why some log output gets written:
//
//	ERROR: bake cake/set heat for baking: oven not found
//
// The string should describe the operation that is about to happen ("starting
// the controller", "list items") or what is being operated on ("HTTP server").
// Multiple different prefixes get concatenated with a slash, the same
// separator klog uses for logger names (see below).
//
// The context's logger (as retrieved through [TContext.Logger] or
// [klog.FromContext]) also gets updated by adding the step as name via
// logr.Logger.WithName.
func (tCtx TContext) WithStep(step string) TContext {
	if tCtx.steps == "" {
		tCtx.steps = step + ": "
	} else {
		tCtx.steps = strings.TrimSuffix(tCtx.steps, ": ") + "/" + step + ": "
	}
	logger := klog.FromContext(tCtx.Context).WithName(step)
	tCtx.Context = klog.NewContext(tCtx.Context, logger)
	return tCtx
}

// Step is useful when the context with the step information is
// used more than once:
//
//	ktesting.Step(tCtx, "step 1", func(tCtx ktesting.TContext) {
//	 tCtx.Log(...)
//	    if (... ) {
//	       tCtx.Failf(...)
//	    }
//	)}
//
// Inside the callback, the tCtx variable is the one where the step
// has been added. This avoids the need to introduce multiple different
// context variables and risk of using the wrong one.
func (tCtx TContext) Step(step string, cb func(tCtx TContext)) {
	tCtx.Helper()
	cb(tCtx.WithStep(step))
}

// Value intercepts a search for the special "GINKGO_SPEC_CONTEXT" and
// wraps the underlying reporter so that the recorded steps and the name of
// the running test are visible in the progress report.
func (tCtx TContext) Value(key any) any {
	if s, ok := key.(string); ok && s == ginkgoSpecContextKey {
		// When we construct a new TContext, we have to be careful to not wrap
		// our own TContext instance. Otherwise this tCtx.Context.Value call
		// here will call TContext.Value once more and wrap a ginkgoReporter inside
		// a ginkgoReporter recursively.
		if reporter, ok := tCtx.Context.Value(key).(ginkgoReporter); ok {
			return ginkgoReporter(&stepReporter{reporter: reporter, testName: tCtx.Name(), steps: tCtx.steps})
		}
	}
	return tCtx.Context.Value(key)
}

type stepReporter struct {
	reporter ginkgoReporter
	testName string
	steps    string
}

var _ ginkgoReporter = &stepReporter{}

func (s *stepReporter) AttachProgressReporter(reporter func() string) func() {
	return s.reporter.AttachProgressReporter(func() string {
		report := s.steps + reporter()
		return s.testName + ":\n" + indent(report, true)
	})
}

// buildHeader handles:
// - "ERROR:<non-empty prefix><optional header><suffix>" -> use both prefix and suffix when we have a header, otherwise just the suffix
// - "<empty prefix><optional header><suffix>" -> use suffix only if we have a header
func (tCtx TContext) buildHeader(prefix, suffix string) string {
	if tCtx.perTestHeader != nil {
		return prefix + tCtx.perTestHeader() + suffix
	}
	if prefix != "" {
		return suffix
	}
	return ""
}

// indent either indents all follow-up lines or all lines including the first one.
func indent(msg string, all bool) string {
	header := ""
	if all {
		header = "\t"
	}
	return header + strings.ReplaceAll(msg, "\n", "\n\t")
}

func (tCtx TContext) Skip(args ...any) {
	tCtx.Helper()
	// Enable `go vet printf` by directly calling fmt.Sprintln.
	msg := strings.TrimSpace(fmt.Sprintln(args...))
	tCtx.TB().Skip("SKIP:", tCtx.buildHeader(" ", " ")+tCtx.steps+indent(msg, false))
}

func (tCtx TContext) Skipf(format string, args ...any) {
	tCtx.Helper()
	// Enable `go vet printf` by directly calling fmt.Sprintf.
	msg := strings.TrimSpace(fmt.Sprintf(format, args...))
	tCtx.TB().Skip("SKIP:", tCtx.buildHeader(" ", " ")+tCtx.steps+indent(msg, false))
}

func (tCtx TContext) Log(args ...any) {
	tCtx.Helper()
	// Enable `go vet printf` by directly calling fmt.Sprintln.
	msg := strings.TrimSpace(fmt.Sprintln(args...))
	tCtx.TB().Log(tCtx.buildHeader("", " ") + tCtx.steps + indent(msg, false))
}

func (tCtx TContext) Logf(format string, args ...any) {
	tCtx.Helper()
	// Enable `go vet printf` by directly calling fmt.Sprintf.
	msg := strings.TrimSpace(fmt.Sprintf(format, args...))
	tCtx.TB().Log(tCtx.buildHeader("", " ") + tCtx.steps + indent(msg, false))
}

// reportFailure writes msg to the underlying TB as an "ERROR" or "FATAL
// ERROR", depending on fatal. testing.T itself already logs the source code
// location of the call (via t.Helper() bookkeeping), so this does not also
// dump a stack backtrace: only [TContext.ExpectNoError] and
// [TContext.AssertNoError] do that, for a [FailureError] with a captured
// backtrace from its original occurrence.
func (tCtx TContext) reportFailure(fatal bool, msg string) {
	tCtx.Helper()
	if fatal {
		// FATAL ERROR *before* header to make it stand out as failure.
		tCtx.TB().Fatal("FATAL ERROR:" + tCtx.buildHeader(" ", "\n") + indent(tCtx.steps+msg, true))
		return
	}
	// ERROR *before* header to make it stand out as failure.
	tCtx.TB().Error("ERROR:" + tCtx.buildHeader(" ", "\n") + indent(tCtx.steps+msg, true))
}

func (tCtx TContext) Error(args ...any) {
	if tCtx.capture == nil {
		tCtx.Helper()
		msg := strings.TrimSpace(fmt.Sprintln(args...))
		tCtx.reportFailure(false, msg)
		return
	}

	tCtx.capture.mutex.Lock()
	defer tCtx.capture.mutex.Unlock()

	// Gomega adds a leading newline in https://github.com/onsi/gomega/blob/f804ac6ada8d36164ecae0513295de8affce1245/internal/gomega.go#L37
	// Let's strip that at start and end because ktesting will make errors
	// stand out more with the "ERROR" prefix, so there's no need for additional
	// line breaks. Besides, Sprintln (required for `go vet printf`) also
	// adds a trailing newline that we don't want.
	msg := strings.TrimSpace(fmt.Sprintln(args...))
	tCtx.capture.errors = append(tCtx.capture.errors, FailureError{
		Msg:            tCtx.steps + msg,
		FullStackTrace: captureBacktrace(),
	})
	tCtx.capture.failed = true
}

func (tCtx TContext) Errorf(format string, args ...any) {
	if tCtx.capture == nil {
		tCtx.Helper()
		// Enable `go vet printf` by directly calling fmt.Sprintln.
		msg := strings.TrimSpace(fmt.Sprintf(format, args...))
		tCtx.reportFailure(false, msg)
		return
	}

	tCtx.capture.mutex.Lock()
	defer tCtx.capture.mutex.Unlock()

	msg := strings.TrimSpace(fmt.Sprintf(format, args...))
	tCtx.capture.errors = append(tCtx.capture.errors, FailureError{
		Msg:            tCtx.steps + msg,
		FullStackTrace: captureBacktrace(),
	})
	tCtx.capture.failed = true
}

func (tCtx TContext) Fail() {
	if tCtx.capture == nil {
		tCtx.TB().Fail()
		return
	}

	tCtx.capture.mutex.Lock()
	defer tCtx.capture.mutex.Unlock()

	tCtx.capture.errors = append(tCtx.capture.errors, FailureError{
		Msg:            errFailedWithNoExplanation.Error(),
		FullStackTrace: captureBacktrace(),
	})
	tCtx.capture.failed = true
}

func (tCtx TContext) FailNow() {
	if tCtx.capture == nil {
		tCtx.TB().FailNow()
		return
	}

	tCtx.capture.mutex.Lock()
	defer tCtx.capture.mutex.Unlock()

	if !tCtx.capture.failed {
		tCtx.capture.errors = append(tCtx.capture.errors, FailureError{
			Msg:            errFailedWithNoExplanation.Error(),
			FullStackTrace: captureBacktrace(),
		})
	}
	tCtx.capture.failed = true
	panic(failed)
}

func (tCtx TContext) Failed() bool {
	if tCtx.capture == nil {
		return tCtx.TB().Failed()
	}

	tCtx.capture.mutex.Lock()
	defer tCtx.capture.mutex.Unlock()

	return tCtx.capture.failed
}

func (tCtx TContext) Fatal(args ...any) {
	if tCtx.capture == nil {
		tCtx.Helper()
		// Enable `go vet printf` by directly calling fmt.Sprintln.
		msg := strings.TrimSpace(fmt.Sprintln(args...))
		tCtx.reportFailure(true, msg)
	}

	tCtx.Error(args...)
	tCtx.FailNow()
}

func (tCtx TContext) Fatalf(format string, args ...any) {
	if tCtx.capture == nil {
		tCtx.Helper()
		// Enable `go vet printf` by directly calling fmt.Sprintf.
		msg := strings.TrimSpace(fmt.Sprintf(format, args...))
		tCtx.reportFailure(true, msg)
		return
	}

	tCtx.Errorf(format, args...)
	tCtx.FailNow()
}

// fatalWithError is the internal type that should never get propagated up. The
// only case where that can happen is when the developer forgot to call
// finalize via defer. The string explains that, in case that developers get to
// see it.
type fatalWithError string

const failed = fatalWithError("WithError TContext encountered a fatal error, but the finalize function was not called via defer as it should have been.")

var errFailedWithNoExplanation = errors.New("WithError context was marked as failed without recording an error")
