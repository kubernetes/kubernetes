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
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/ktesting"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/test/integration/framework"
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
	tCtx     ktesting.TContext
	registry *FaultRegistry
	ctx      context.Context
	cancel   context.CancelFunc

	adminClient    clientset.Interface
	kubeConfig     *restclient.Config
	tearDownServer framework.TearDownFunc

	activity    *activityTracker
	monitorDone chan struct{}  // closed when the invariant monitor goroutine exits
	goroutines  sync.WaitGroup // goroutines started via Go

	mu                   sync.RWMutex
	continuousInvariants []NamedInvariant
	invariantErr         error
}

// NewFixture starts a fresh API server and the continuous invariant monitor.
// It wraps controller.ExpectationsClock, so fixtures must not overlap (no t.Parallel).
func NewFixture(tCtx ktesting.TContext) *RobustnessTestFixture {
	return newFixture(tCtx, NewFaultRegistry())
}

// newFixture is NewFixture with a caller-supplied registry.
func newFixture(tCtx ktesting.TContext, registry *FaultRegistry) *RobustnessTestFixture {
	tCtx.Logf("Fault registry seed: %d (export %s=%d to replay probabilistic faults)", registry.Seed(), SeedEnvVar, registry.Seed())
	if _, nested := controller.ExpectationsClock.(*FaultInjectingClock); nested {
		tCtx.Fatalf("controller.ExpectationsClock is already wrapped: robustness fixtures must not overlap")
	}

	adminClient, kubeConfig, tearDownServer := framework.StartTestServer(tCtx, tCtx, framework.TestServerSetup{
		ModifyServerRunOptions: func(o *options.ServerRunOptions) {
			// Pods need no service account, and no admission-added node taints.
			o.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount", "TaintNodesByCondition", "Priority", "StorageObjectInUseProtection"}
		},
	})
	ctx, cancel := context.WithCancel(tCtx)

	f := &RobustnessTestFixture{
		tCtx:           tCtx,
		registry:       registry,
		ctx:            ctx,
		cancel:         cancel,
		adminClient:    adminClient,
		kubeConfig:     kubeConfig,
		tearDownServer: tearDownServer,
		activity:       &activityTracker{},
		monitorDone:    make(chan struct{}),
	}

	originalClock := controller.ExpectationsClock
	controller.ExpectationsClock = NewFaultInjectingClock(ctx, originalClock, f.registry, ExpectationsClockName)
	tCtx.Cleanup(func() { controller.ExpectationsClock = originalClock })

	f.startContinuousInvariantMonitor(50 * time.Millisecond)
	return f
}

// Go runs fn with the fixture context in a goroutine that TearDown joins, so a
// controller that does not stop when its context is cancelled fails its own cell.
func (f *RobustnessTestFixture) Go(fn func(ctx context.Context)) {
	f.goroutines.Go(func() { fn(f.ctx) })
}

// TCtx returns the scenario subtest's context.
func (f *RobustnessTestFixture) TCtx() ktesting.TContext {
	return f.tCtx
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
	config := restclient.CopyConfig(f.kubeConfig)
	config.QPS = -1                         // Disable client-side throttling for test loops
	config.ContentType = "application/json" // watch events are intercepted as JSON lines
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
	return f.adminClient
}

// WrapIndexer wraps a cache.Indexer with our fault injection hook.
func (f *RobustnessTestFixture) WrapIndexer(realIndexer cache.Indexer, name string) cache.Indexer {
	return NewFaultInjectingIndexer(f.ctx, realIndexer, f.registry, name)
}

// StartInformers starts the given InformerFactory and blocks until all informer caches have synced.
func (f *RobustnessTestFixture) StartInformers(factory informers.SharedInformerFactory) {
	f.tCtx.Helper()
	factory.Start(f.ctx.Done())
	synced := factory.WaitForCacheSync(f.ctx.Done())
	for typ, ok := range synced {
		if !ok {
			f.tCtx.Fatalf("failed to sync cache for informer: %v", typ)
		}
	}
}

// InjectFault validates rule's domain and registers it in the central registry.
func (f *RobustnessTestFixture) InjectFault(rule FaultRule) {
	f.tCtx.Helper()
	if err := validateFaultDomain(rule); err != nil {
		f.tCtx.Fatalf("InjectFault(%q): %v", rule.Name, err)
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
	f.tCtx.Helper()
	err := wait.PollUntilContextTimeout(f.ctx, 100*time.Millisecond, timeout, true, func(ctx context.Context) (bool, error) {
		f.mu.RLock()
		invErr := f.invariantErr
		f.mu.RUnlock()
		if invErr != nil {
			return false, fmt.Errorf("aborted: continuous invariant was violated: %w", invErr)
		}

		if err := fn(ctx, f.AdminClientSet()); err != nil {
			f.tCtx.Logf("[Wait] Invariant %q not met yet: %v", name, err)
			return false, nil
		}
		return true, nil
	})

	if err != nil {
		f.tCtx.Fatalf("Liveness invariant %q failed to converge within %v: %v", name, timeout, err)
	}
}

// WaitUntilSettled blocks until the controller has issued at least one mutating
// request through the wrapped client and then gone write-idle for quietWindow.
func (f *RobustnessTestFixture) WaitUntilSettled(quietWindow, timeout time.Duration) bool {
	f.tCtx.Helper()
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
	f.tCtx.Helper()
	if !f.WaitUntilSettled(quietWindow, timeout) {
		f.mu.RLock()
		invErr := f.invariantErr
		f.mu.RUnlock()
		if invErr != nil {
			f.tCtx.Fatalf("Settle wait for %q aborted: continuous invariant was violated: %v", name, invErr)
		}
		f.tCtx.Fatalf("Controller did not settle (a %v write-idle window) within %v while waiting to check %q", quietWindow, timeout, name)
	}
	if err := fn(f.ctx, f.AdminClientSet()); err != nil {
		f.tCtx.Errorf("Invariant %q failed once settled: %v", name, err)
	}
}

// WaitForBlockingFaults blocks until every fired InjectDelay has released (plus
// a short margin for the controller to observe the backlog). It reports whether
// it waited at all. Returns early if the fixture context is cancelled.
func (f *RobustnessTestFixture) WaitForBlockingFaults(timeout time.Duration) bool {
	f.tCtx.Helper()
	deadline := f.registry.BlockingFaultDeadline()
	if deadline.IsZero() {
		return false
	}
	wait := time.Until(deadline.Add(time.Second))
	if wait <= 0 {
		return true
	}
	if wait > timeout {
		f.tCtx.Fatalf("injected stall releases in %v, beyond the %v liveness timeout", wait, timeout)
	}
	select {
	case <-time.After(wait):
	case <-f.ctx.Done():
	}
	return true
}

// AssertAllFaultsMatched fails the test if any non-optional fault rule never matched an injection site.
func (f *RobustnessTestFixture) AssertAllFaultsMatched() {
	f.tCtx.Helper()
	if unmatched := f.registry.UnmatchedRules(); len(unmatched) > 0 {
		f.tCtx.Errorf("fault rule(s) registered but never matched any injection site (injected nothing): %v", unmatched)
	}
}

// AssertExpectedFaultsTriggered fails the test if any ExpectTriggered fault rule never fired.
func (f *RobustnessTestFixture) AssertExpectedFaultsTriggered() {
	f.tCtx.Helper()
	if silent := f.registry.ExpectedRulesNotTriggered(); len(silent) > 0 {
		f.tCtx.Errorf("fault rule(s) declared ExpectTriggered but never fired: %v", silent)
	}
}

func (f *RobustnessTestFixture) startContinuousInvariantMonitor(pollInterval time.Duration) {
	go func() {
		defer close(f.monitorDone)
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
					f.tCtx.Errorf("[Invariant Safety Violation] %v", err)
					f.cancel()
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

// TearDown stops the controller goroutines, the invariant monitor and the API
// server, in that order. Goroutines that ignore cancellation fail the test.
func (f *RobustnessTestFixture) TearDown() {
	f.tCtx.Helper()
	f.cancel()
	<-f.monitorDone
	done := make(chan struct{})
	go func() {
		f.goroutines.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		f.tCtx.Errorf("goroutines started with fixture.Go did not exit within 10s of context cancellation")
	}
	f.tearDownServer()
}
