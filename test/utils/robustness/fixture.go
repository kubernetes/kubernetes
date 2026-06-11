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

package robustness

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/controller"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// ExpectationsClockName is the registry name of the wrapped ControllerExpectations
// clock installed by NewFixture. Target it with ClockMatch{Clock: ExpectationsClockName}.
const ExpectationsClockName = "expectations"

// activityTracker records mutating API traffic seen by the wrapped transport. It
// is the signal used to detect that the controller has settled: it issued at
// least one write and then gone idle. All methods are safe on a nil receiver
// (the transport may be constructed without a tracker, e.g. in unit tests).
type activityTracker struct {
	mutations    atomic.Int64 // count of mutating requests observed
	lastMutation atomic.Int64 // UnixNano of the most recent mutating request
}

// recordMutation notes that a mutating request (POST/PUT/PATCH/DELETE) was issued.
func (a *activityTracker) recordMutation() {
	if a == nil {
		return
	}
	a.lastMutation.Store(time.Now().UnixNano())
	a.mutations.Add(1)
}

// RobustnessTestFixture coordinates the integration API server lifecycle,
// wrapped clients/queues, fault injection registry, and safety invariant monitoring.
type RobustnessTestFixture struct {
	t          *testing.T
	registry   *FaultRegistry
	testCtx    *testutils.TestContext
	cancelCtx  context.CancelFunc
	ctx        context.Context
	tearDownFn func()

	activity *activityTracker

	mu                   sync.RWMutex
	continuousInvariants []NamedInvariant
	invariantErr         error
}

// NewFixture creates a new test fixture, initializes the APIServer, and starts continuous invariant checks.
func NewFixture(t *testing.T, testName string) *RobustnessTestFixture {
	// Initialize API Server
	apiCtx := testutils.InitTestAPIServer(t, testName, nil)

	ctx, cancel := context.WithCancel(apiCtx.Ctx)

	f := &RobustnessTestFixture{
		t:         t,
		registry:  NewFaultRegistry(),
		testCtx:   apiCtx,
		ctx:       ctx,
		cancelCtx: cancel,
		activity:  &activityTracker{},
	}

	// Inject our fault-injecting expectations clock globally for this test run!
	originalClock := controller.ExpectationsClock
	controller.ExpectationsClock = NewFaultInjectingClock(originalClock, f.registry, ExpectationsClockName)

	f.tearDownFn = func() {
		// Restore the original expectations clock to avoid side effects on other tests
		controller.ExpectationsClock = originalClock
		cancel()
	}

	f.startContinuousInvariantMonitor(50 * time.Millisecond)
	return f
}

// Context returns the test execution context (cancelled if any safety invariant is violated).
func (f *RobustnessTestFixture) Context() context.Context {
	return f.ctx
}

// Registry returns the FaultRegistry for declaring fault rules.
func (f *RobustnessTestFixture) Registry() *FaultRegistry {
	return f.registry
}

// KubeConfig returns the REST client configuration, wrapped with the dynamic fault interceptor.
func (f *RobustnessTestFixture) KubeConfig() *restclient.Config {
	config := restclient.CopyConfig(f.testCtx.KubeConfig)
	config.QPS = -1 // Disable throttling for rapid test loops
	config.Wrap(func(rt http.RoundTripper) http.RoundTripper {
		return NewFaultInjectingTransport(rt, f.registry, f.activity)
	})
	return config
}

// ClientSet returns a client-go Clientset that is wrapped with the fault injection transport.
func (f *RobustnessTestFixture) ClientSet() clientset.Interface {
	return clientset.NewForConfigOrDie(f.KubeConfig())
}

// AdminClientSet returns an un-wrapped, direct API clientset (for arranging test state and verifying results).
func (f *RobustnessTestFixture) AdminClientSet() clientset.Interface {
	return f.testCtx.ClientSet
}

// WrapIndexer wraps a cache.Indexer with our fault injection hook.
func (f *RobustnessTestFixture) WrapIndexer(realIndexer cache.Indexer, name string) cache.Indexer {
	return NewFaultInjectingIndexer(realIndexer, f.registry, name)
}

// WrapQueue wraps a workqueue.RateLimitingInterface with our fault injection hook.
func (f *RobustnessTestFixture) WrapQueue(realQueue workqueue.RateLimitingInterface, name string) workqueue.RateLimitingInterface {
	return NewFaultInjectingWorkQueue(realQueue, f.registry, name)
}

// InjectFault registers a new fault rule in the central registry. It validates
// that the rule's action can actually be served at its injection point, failing
// the test immediately on a mis-targeted fault (e.g. a clock fault attached to a
// transport point) rather than letting it silently misbehave at runtime.
func (f *RobustnessTestFixture) InjectFault(rule FaultRule) {
	f.t.Helper()
	if err := validateFaultDomain(rule); err != nil {
		f.t.Fatalf("InjectFault(%q): %v", rule.Name, err)
	}
	f.registry.Register(rule)
}

// AddContinuousInvariant registers a safety condition checked repeatedly in the background.
func (f *RobustnessTestFixture) AddContinuousInvariant(name string, fn Invariant) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.continuousInvariants = append(f.continuousInvariants, NamedInvariant{Name: name, Fn: fn})
}

// AssertEventually blocks until the given liveness invariant passes, or times out.
func (f *RobustnessTestFixture) AssertEventually(name string, fn Invariant, timeout time.Duration) {
	f.t.Helper()
	err := wait.PollUntilContextTimeout(f.ctx, 100*time.Millisecond, timeout, true, func(ctx context.Context) (bool, error) {
		// Check for background continuous invariant failures first
		f.mu.RLock()
		invErr := f.invariantErr
		f.mu.RUnlock()
		if invErr != nil {
			return false, fmt.Errorf("aborted: continuous invariant was violated: %v", invErr)
		}

		if err := fn(ctx, f.AdminClientSet()); err != nil {
			f.t.Logf("[Wait] Invariant %q not met yet: %v", name, err)
			return false, nil
		}
		return true, nil
	})

	if err != nil {
		f.t.Fatalf("Liveness invariant %q failed to converge within %v: %v", name, timeout, err)
	}
}

// WaitUntilSettled blocks until the controller-under-test has settled, defined as:
// it issued at least one mutating request through the wrapped client and then went
// idle (no further mutating requests) for quietWindow. Returns true once settled,
// or false if the timeout or a background safety violation is hit first.
//
// This is a controller-agnostic proxy for "end of reconciliation" — it needs no
// access to the controller's work queue, and it naturally waits out retry storms
// under fault injection (each retried write resets the idle window). It assumes
// the controller performs at least one write on its way to convergence;
// controllers that converge without writing should use AssertEventually instead.
func (f *RobustnessTestFixture) WaitUntilSettled(quietWindow, timeout time.Duration) bool {
	f.t.Helper()
	startMutations := f.activity.mutations.Load()
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-f.ctx.Done():
			return false
		case <-ticker.C:
			if time.Now().After(deadline) {
				return false
			}
			f.mu.RLock()
			invErr := f.invariantErr
			f.mu.RUnlock()
			if invErr != nil {
				return false
			}
			// Require that the controller has acted since we began waiting, then
			// that it has been idle for the full quiet window.
			if f.activity.mutations.Load() <= startMutations {
				continue
			}
			idle := time.Since(time.Unix(0, f.activity.lastMutation.Load()))
			if idle >= quietWindow {
				return true
			}
		}
	}
}

// AssertWhenSettled waits for the controller to settle (see WaitUntilSettled), then
// evaluates the invariant exactly once. Unlike AssertEventually (which retries
// until the condition holds), this checks the final, steady state deterministically
// — the right semantics for "X holds at the end of reconciliation". It fails if
// the controller does not settle in time or if the invariant does not hold once
// settled.
func (f *RobustnessTestFixture) AssertWhenSettled(name string, fn Invariant, quietWindow, timeout time.Duration) {
	f.t.Helper()
	if !f.WaitUntilSettled(quietWindow, timeout) {
		f.mu.RLock()
		invErr := f.invariantErr
		f.mu.RUnlock()
		if invErr != nil {
			f.t.Fatalf("Settle wait for %q aborted: continuous invariant was violated: %v", name, invErr)
		}
		f.t.Fatalf("Controller did not settle (a %v write-idle window) within %v while waiting to check %q", quietWindow, timeout, name)
	}
	if err := fn(f.ctx, f.AdminClientSet()); err != nil {
		f.t.Errorf("Invariant %q failed once settled: %v", name, err)
	}
}

// AssertFaultHitCount verifies that an injected fault was triggered the expected number of times.
func (f *RobustnessTestFixture) AssertFaultHitCount(ruleName string, expected int) {
	actual := f.registry.GetHitCount(ruleName)
	if actual != expected {
		f.t.Errorf("Fault verification failed for %q: expected %d hits, got %d", ruleName, expected, actual)
	}
}

// AssertAllFaultsMatched fails the test if any registered non-optional fault rule
// was never exercised. An unmatched rule means its Match never lined up with a
// real injection site, so it injected nothing and the scenario silently asserted
// nothing. Mark a rule Optional to exempt it (e.g. feature-gated faults).
func (f *RobustnessTestFixture) AssertAllFaultsMatched() {
	f.t.Helper()
	if unmatched := f.registry.UnmatchedRules(); len(unmatched) > 0 {
		f.t.Errorf("fault rule(s) registered but never matched any injection site (injected nothing): %v", unmatched)
	}
}

// AssertExpectedFaultsTriggered fails the test if any fault rule registered with
// ExpectTriggered never actually fired. Matching proves a rule lined up with a
// real injection site (see AssertAllFaultsMatched); this proves the declared
// faults were also delivered, without per-rule hit-count boilerplate.
func (f *RobustnessTestFixture) AssertExpectedFaultsTriggered() {
	f.t.Helper()
	if silent := f.registry.ExpectedRulesNotTriggered(); len(silent) > 0 {
		f.t.Errorf("fault rule(s) declared ExpectTriggered but never fired: %v", silent)
	}
}

// AssertFaultHitCountGreaterThan verifies that an injected fault was triggered at least once (or more).
func (f *RobustnessTestFixture) AssertFaultHitCountGreaterThan(ruleName string, min int) {
	actual := f.registry.GetHitCount(ruleName)
	if actual <= min {
		f.t.Errorf("Fault verification failed for %q: expected > %d hits, got %d", ruleName, min, actual)
	}
}

func (f *RobustnessTestFixture) startContinuousInvariantMonitor(pollInterval time.Duration) {
	go func() {
		ticker := time.NewTicker(pollInterval)
		defer ticker.Stop()
		for {
			select {
			case <-f.ctx.Done():
				return
			case <-ticker.C:
				// Non-blocking check: if the context was cancelled during sleep, exit immediately to prevent races
				select {
				case <-f.ctx.Done():
					return
				default:
				}

				if err := f.checkContinuousInvariants(); err != nil {
					f.mu.Lock()
					f.invariantErr = err
					f.mu.Unlock()
					f.t.Errorf("[Invariant Safety Violation] %v", err)
					f.cancelCtx() // Cancel context to abort test immediately
					return
				}
			}
		}
	}()
}

func (f *RobustnessTestFixture) checkContinuousInvariants() error {
	f.mu.RLock()
	defer f.mu.RUnlock()
	for _, inv := range f.continuousInvariants {
		if err := inv.Fn(f.ctx, f.AdminClientSet()); err != nil {
			return fmt.Errorf("safety invariant %q failed: %v", inv.Name, err)
		}
	}
	return nil
}

// TearDown cancels context and shuts down the in-memory API server.
func (f *RobustnessTestFixture) TearDown() {
	f.tearDownFn()
}
