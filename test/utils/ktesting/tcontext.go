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
	"flag"
	"fmt"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/go-logr/logr"
	"github.com/onsi/gomega"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/test/utils/format"
	"k8s.io/kubernetes/test/utils/ktesting/initoption"
	"k8s.io/kubernetes/test/utils/ktesting/internal"
)

// Underlier is the additional interface implemented by the per-test LogSink
// behind [TContext.Logger]. Together with [initoption.BufferLogs] it can be
// used to capture log output in memory to check it in tests.
type Underlier = ktesting.Underlier

// CleanupGracePeriod is the time that a [TContext] gets canceled before the
// deadline of its underlying test suite (usually determined via "go test
// -timeout"). This gives the running test(s) time to fail with an informative
// timeout error. After that, all cleanup callbacks then have the remaining
// time to complete before the test binary is killed.
//
// For this to work, each blocking calls in a test must respect the
// cancellation of the [TContext].
//
// When using Ginkgo to manage the test suite and running tests, the
// CleanupGracePeriod is ignored because Ginkgo itself manages timeouts.
const CleanupGracePeriod = 5 * time.Second

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

// Init can be called in a unit or integration test to create
// a test context which:
// - has a per-test logger with verbosity derived from the -v command line flag
// - gets canceled when the test finishes (via [TB.Cleanup])
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
// Can be called more than once per test to get different contexts with
// independent cancellation. The default behavior describe above can be
// modified via optional functional options defined in [initoption].
func Init(tb TB, opts ...InitOption) TContext {
	tb.Helper()

	c := internal.InitConfig{
		PerTestOutput: true,
	}
	for _, opt := range opts {
		opt(&c)
	}

	// We don't need a Deadline implementation, testing.B doesn't have it.
	// But if we have one, we'll use it to set a timeout shortly before
	// the deadline. This needs to come before we wrap tb.
	deadlineTB, deadlineOK := tb.(interface {
		Deadline() (time.Time, bool)
	})

	ctx := defaultProgressReporter.init(tb)
	var header func() string
	if c.PerTestOutput {
		logger := newLogger(tb, c.BufferLogs)
		ctx = klog.NewContext(ctx, logger)
		header = klogHeader
	}

	var cancelTimeout func(cause string)
	if deadlineOK {
		if deadline, ok := deadlineTB.Deadline(); ok {
			timeLeft := time.Until(deadline)
			timeLeft -= CleanupGracePeriod
			ctx, cancelTimeout = withTimeout(ctx, tb, timeLeft, fmt.Sprintf("test suite deadline (%s) is close, need to clean up before the %s cleanup grace period", deadline.Truncate(time.Second), CleanupGracePeriod))
		}
	}

	// Construct new TContext with context and settings as determined above.
	tCtx := InitCtx(ctx, tb)
	if cancelTimeout != nil {
		tCtx.cancel = cancelTimeout
	} else {
		tCtx = WithCancel(tCtx)
		tCtx.Cleanup(func() {
			tCtx.Cancel(cleanupErr(tCtx.Name()).Error())
		})
	}
	tCtx.perTestHeader = header

	return tCtx
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

type InitOption = initoption.InitOption

// InitCtx is a variant of [Init] which uses an already existing context and
// whatever logger and timeouts are stored there.
// Functional options are part of the API, but currently
// there are none which have an effect.
func InitCtx(ctx context.Context, tb TB, _ ...InitOption) TContext {
	tc := TC{
		Context:   ctx,
		testingTB: testingTB{TB: tb},
	}
	return &tc
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
func (tc *TC) withTB(tb TB) TContext {
	tc = tc.clone()
	tc.testingTB.TB = tb
	if tc.perTestHeader != nil {
		logger := newLogger(tb, false /* don't buffer logs in sub-test */)
		tc.Context = klog.NewContext(tc.Context, logger)
	}
	tc = WithCancel(tc)
	return tc
}

// run implements the different Run and SyncTest methods. It's not an exported
// method because tCtx.Run is more discoverable (same usage as
// with normal Go).
func run(tc *TC, name string, syncTest bool, cb func(tc *TC)) bool {
	tc.Helper()
	switch tb := tc.TB().(type) {
	case *testing.T:
		if syncTest {
			f := func(t *testing.T) {
				// We must not propagate the parent's
				// cancellation channel into the bubble,
				// it causes "panic: receive on synctest channel from outside bubble".
				//
				// Sync tests shouldn't need the overall suite timeout,
				// so this seems okay.
				tc = tc.clone()
				tc.isSyncTest = true
				tc = tc.WithoutCancel().withTB(t)
				cb(tc)
			}
			if name != "" {
				return tb.Run(name, func(t *testing.T) { synctest.Test(t, f) })
			}
			synctest.Test(tb, f)
			return true
		}
		return tb.Run(name, func(t *testing.T) {
			cb(tc.withTB(t))
		})
	case *testing.B:
		if !syncTest {
			return tb.Run(name, func(b *testing.B) {
				cb(tc.withTB(b))
			})
		}
	}

	what := "Run"
	if syncTest {
		what = "SyncTest"
	}
	tc.Fatalf("%s not implemented, underlying %T does not support it", what, tc.TB())

	return false
}

// Deprecated: use tCtx.WithContext instead
func WithContext(tCtx TContext, ctx context.Context) TContext {
	return tCtx.WithContext(ctx)
}

// WithContext constructs a new TContext with a different Context instance.
// This can be used in callbacks which receive a Context, for example
// from Gomega:
//
//	gomega.Eventually(tCtx, func(ctx context.Context) {
//	   tCtx := ktesting.WithContext(tCtx, ctx)
//	   ...
//
// This is important because the Context in the callback could have
// a different deadline than in the parent TContext.
func (tc *TC) WithContext(ctx context.Context) TContext {
	tc = tc.clone()
	logger := tc.Logger()
	tc.Context = ctx
	if _, err := logr.FromContext(ctx); err != nil {
		// Keep using the logger from the parent context.
		tc = tc.WithLogger(logger)
	}
	return tc
}

// Deprecated: use tCtx.WithValue instead
func WithValue(tCtx TContext, key, val any) TContext {
	return tCtx.WithValue(key, val)
}

// WithValue wraps [context.WithValue] such that the result is again a TContext.
func (tc *TC) WithValue(key, val any) TContext {
	ctx := context.WithValue(tc, key, val)
	return tc.WithContext(ctx)
}

// TContext is the recommended type for storing a [TC] instance.
// The type alias is necessary because TContext used to be an interface.
type TContext = *TC

// TC implements [context.Context], [testing.TB] and some additional
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
type TC struct {
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
	cancel func(cause string)

	// steps is a concatenation ("step1: step2: step3: ") of steps passed to WithStep.
	// It's empty if there are no steps.
	steps string

	// for SyncTest
	isSyncTest bool

	// for WithClient
	restConfig    *rest.Config
	restMapper    *restmapper.DeferredDiscoveryRESTMapper
	client        clientset.Interface
	dynamic       dynamic.Interface
	apiextensions apiextensions.Interface

	// for WithNamespace
	namespace string

	// capture, if non-nil, changes Error/Errorf/Fatal/Fatalf/Fail/FailNow so
	// that they intercept the problem and convert to errors. Log messages
	// are passed through.
	//
	// Used by WithError.
	capture *capture
}

type capture struct {
	mutex  sync.Mutex
	errors []error
	failed bool
}

// tc makes a shallow copy.
func (tc *TC) clone() *TC {
	clone := *tc
	return &clone
}

// testingTB is needed to avoid a name conflict
// between field and method in tContext.
type testingTB struct {
	// TB makes the methods of the underlying test implementation available.
	// In particular Helper must be called directly, not via a wrapper.
	// It must not be modified.
	TB
}

var _ TContext = &TC{}

// Parallel signals that this test is to be run in parallel with (and
// only with) other parallel tests. In other words, it needs to be
// called in each test which is meant to run in parallel.
//
// Only supported in Go unit tests, calling it elsewhere causes a test failure.
//
// When a unit test is run multiple times due to use of -test.count or -test.cpu,
// multiple instances of a single test never run in parallel with each other.
func (tc *TC) Parallel() {
	if tb, ok := tc.TB().(interface{ Parallel() }); ok {
		tb.Parallel()
	} else {
		tc.Fatalf("Parallel not implemented, underlying %T does not support it", tc.TB())
	}
}

// Cancel can be invoked to cancel the context before the test is completed.
// Tests which use the context to control goroutines and then wait for
// termination of those goroutines must call Cancel to avoid a deadlock.
//
// The cause, if non-empty, is turned into an error which is equivalent
// to context.Canceled. context.Cause will return that error for the
// context.
func (tc *TC) Cancel(cause string) {
	if tc.cancel != nil {
		tc.cancel(cause)
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
func (tc *TC) CleanupCtx(cb func(TContext)) {
	tc.Helper()

	if tb, ok := tc.TB().(ContextTB); ok {
		// Use context from base TB (most likely Ginkgo).
		tb.CleanupCtx(func(ctx context.Context) {
			tCtx := WithContext(tc, ctx)
			cb(tCtx)
		})
		return
	}

	tc.Cleanup(func() {
		// Use new context. This is the code path for "go test". The
		// context then has *no* deadline. In the code path above for
		// Ginkgo, Ginkgo is more sophisticated and also applies
		// timeouts to cleanup calls which accept a context.
		childCtx := WithContext(tc, context.WithoutCancel(tc))
		cb(childCtx)
	})
}

// Run runs cb as a subtest called name. It blocks until cb returns or
// calls t.Parallel to become a parallel test.
//
// Only supported in Go unit tests or benchmarks. It fails the current
// test when called elsewhere.
func (tc *TC) Run(name string, cb func(tCtx TContext)) bool {
	return run(tc, name, false, cb)
}

// SyncTest uses [synctest.Test] to execute the callback inside a bubble.
// Creates a new subtest if the name is non-empty, otherwise it creates
// the bubble directly in the current test context.
//
// Only works in Go unit tests.
func (tc *TC) SyncTest(name string, cb func(tCtx TContext)) bool {
	return run(tc, name, true, cb)
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
func (tc *TC) IsSyncTest() bool {
	return tc.isSyncTest
}

// Wait calls [synctest.Wait] and thus ensures that all background
// activity has settled down (= "durably blocked").
//
// Only works inside a bubble started by SyncTest (can be checked with
// IsSyncTest), panics elsewhere.
func (tc *TC) Wait() {
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
func (tc *TC) TB() TB { return tc.testingTB.TB }

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
func (tc *TC) Logger() klog.Logger {
	return klog.FromContext(tc.Context)
}

// RESTConfig returns a copy of the config for a rest client with the UserAgent
// set to include the current test name or nil if not available. Several typed
// clients using this config are available through [Client], [Dynamic],
// [APIExtensions].
func (tc *TC) RESTConfig() *rest.Config {
	return rest.CopyConfig(tc.restConfig)
}

func (tc *TC) RESTMapper() *restmapper.DeferredDiscoveryRESTMapper { return tc.restMapper }
func (tc *TC) Client() clientset.Interface                         { return tc.client }
func (tc *TC) Dynamic() dynamic.Interface                          { return tc.dynamic }
func (tc *TC) APIExtensions() apiextensions.Interface              { return tc.apiextensions }

// Expect wraps [gomega.Expect] such that a failure will be reported via
// [TContext.Fatal]. As with [gomega.Expect], additional values
// may get passed. Those values then all must be nil for the assertion
// to pass. This can be used with functions which return a value
// plus error. The error gets checked automatically.
//
//	myAmazingThing := func(int, error) { ...}
//	tCtx.Expect(myAmazingThing()).Should(gomega.Equal(1))
func (tc *TC) Expect(actual interface{}, extra ...interface{}) gomega.Assertion {
	return gomegaAssertion(tc, true, actual, extra...)
}

// Require is an alias for Expect.
func (tc *TC) Require(actual interface{}, extra ...interface{}) gomega.Assertion {
	return gomegaAssertion(tc, true, actual, extra...)
}

// Assert also wraps [gomega.Expect], but in contrast to Expect = Require,
// it reports a failure through [TContext.Error]. This makes it possible
// to test several different assertions.
func (tc *TC) Assert(actual interface{}, extra ...interface{}) gomega.Assertion {
	return gomegaAssertion(tc, false, actual, extra...)
}

// WithNamespace creates a new context with a Kubernetes namespace name for retrieval through [Namespace].
func (tc *TC) WithNamespace(namespace string) TContext {
	tc = tc.clone()
	tc.namespace = namespace
	return tc
}

// Namespace returns the Kubernetes namespace name that was set previously
// through WithNamespace and the empty string if none is available.
//
// This namespace is the one to be used by tests which need to create namespace-scoped
// objects. The name is guaranteed to be unique for the test context, so tests running
// in parallel need to be set up so that each test has its own namespace.
func (tc *TC) Namespace() string {
	return tc.namespace
}
