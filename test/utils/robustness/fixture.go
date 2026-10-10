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
	"errors"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/controller"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// ExpectationsClockName is the registry name of the wrapped ControllerExpectations clock.
const ExpectationsClockName = "expectations"

// activityTracker records mutating API traffic seen by the wrapped transport to
// detect when the controller has settled. Safe on a nil receiver.
type activityTracker struct {
	mutations    atomic.Int64
	lastMutation atomic.Int64 // UnixNano of most recent mutation
}

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
//
// CONCURRENCY WARNING: Mutates controller.ExpectationsClock; tests using this
// fixture must not call t.Parallel().
func NewFixture(t *testing.T, testName string) *RobustnessTestFixture {
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

	originalClock := controller.ExpectationsClock
	controller.ExpectationsClock = NewFaultInjectingClock(originalClock, f.registry, ExpectationsClockName)

	f.tearDownFn = func() {
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

// KubeConfig returns a REST client config wrapped with the fault-injecting transport.
func (f *RobustnessTestFixture) KubeConfig() *restclient.Config {
	config := restclient.CopyConfig(f.testCtx.KubeConfig)
	config.QPS = -1 // Disable client-side throttling for test loops
	config.Wrap(func(rt http.RoundTripper) http.RoundTripper {
		return NewFaultInjectingTransport(rt, f.registry, f.activity)
	})
	return config
}

// ClientSet returns a client-go Clientset wrapped with the fault-injecting transport.
func (f *RobustnessTestFixture) ClientSet() clientset.Interface {
	return clientset.NewForConfigOrDie(f.KubeConfig())
}

// AdminClientSet returns an un-wrapped API clientset for arranging state and checking invariants.
func (f *RobustnessTestFixture) AdminClientSet() clientset.Interface {
	return f.testCtx.ClientSet
}

// WrapIndexer wraps a cache.Indexer with our fault injection hook.
func (f *RobustnessTestFixture) WrapIndexer(realIndexer cache.Indexer, name string) cache.Indexer {
	return NewFaultInjectingIndexer(realIndexer, f.registry, name)
}

// WrapQueue wraps a workqueue.TypedRateLimitingInterface[any] with our fault injection hook.
func (f *RobustnessTestFixture) WrapQueue(realQueue workqueue.TypedRateLimitingInterface[any], name string) workqueue.TypedRateLimitingInterface[any] {
	return NewFaultInjectingWorkQueue(realQueue, f.registry, name)
}

// StartInformers starts the given InformerFactory and blocks until all informer caches have synced.
func (f *RobustnessTestFixture) StartInformers(factory informers.SharedInformerFactory) {
	f.t.Helper()
	factory.Start(f.ctx.Done())
	synced := factory.WaitForCacheSync(f.ctx.Done())
	for typ, ok := range synced {
		if !ok {
			f.t.Fatalf("failed to sync cache for informer: %v", typ)
		}
	}
}

// InjectFault validates rule's domain and registers it in the central registry.
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

// AssertEventually polls until the given liveness invariant passes or times out.
func (f *RobustnessTestFixture) AssertEventually(name string, fn Invariant, timeout time.Duration) {
	f.t.Helper()
	err := wait.PollUntilContextTimeout(f.ctx, 100*time.Millisecond, timeout, true, func(ctx context.Context) (bool, error) {
		f.mu.RLock()
		invErr := f.invariantErr
		f.mu.RUnlock()
		if invErr != nil {
			return false, fmt.Errorf("aborted: continuous invariant was violated: %w", invErr)
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

// WaitUntilSettled blocks until the controller has issued at least one mutating
// request through the wrapped client and then gone write-idle for quietWindow.
func (f *RobustnessTestFixture) WaitUntilSettled(quietWindow, timeout time.Duration) bool {
	f.t.Helper()
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
			if f.activity.mutations.Load() == 0 {
				continue
			}
			idle := time.Since(time.Unix(0, f.activity.lastMutation.Load()))
			if idle >= quietWindow {
				return true
			}
		}
	}
}

// AssertWhenSettled waits for the controller to settle (see WaitUntilSettled)
// and then evaluates the invariant once against the steady state.
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

// AssertAllFaultsMatched fails the test if any non-optional fault rule never matched an injection site.
func (f *RobustnessTestFixture) AssertAllFaultsMatched() {
	f.t.Helper()
	if unmatched := f.registry.UnmatchedRules(); len(unmatched) > 0 {
		f.t.Errorf("fault rule(s) registered but never matched any injection site (injected nothing): %v", unmatched)
	}
}

// AssertExpectedFaultsTriggered fails the test if any ExpectTriggered fault rule never fired.
func (f *RobustnessTestFixture) AssertExpectedFaultsTriggered() {
	f.t.Helper()
	if silent := f.registry.ExpectedRulesNotTriggered(); len(silent) > 0 {
		f.t.Errorf("fault rule(s) declared ExpectTriggered but never fired: %v", silent)
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
				select {
				case <-f.ctx.Done():
					return
				default:
				}

				if err := f.checkContinuousInvariants(); err != nil {
					if errors.Is(err, context.Canceled) || f.ctx.Err() != nil {
						return
					}
					f.mu.Lock()
					f.invariantErr = err
					f.mu.Unlock()
					f.t.Errorf("[Invariant Safety Violation] %v", err)
					f.cancelCtx()
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
			return fmt.Errorf("safety invariant %q failed: %w", inv.Name, err)
		}
	}
	return nil
}

// TearDown cancels context and restores global state.
func (f *RobustnessTestFixture) TearDown() {
	f.tearDownFn()
}
