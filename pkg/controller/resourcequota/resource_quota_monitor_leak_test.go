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

package resourcequota

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
)

// countingInformer is a minimal SharedIndexInformer that records how many event
// handlers have been added and removed, so a test can detect handlers that are
// leaked when a monitor is torn down. Only the methods exercised by
// QuotaMonitor.controllerFor and the monitor teardown are implemented; the rest
// are inherited from the embedded interface and panic if called.
type countingInformer struct {
	cache.SharedIndexInformer
	added   int
	removed int
	// addErr, when set, is returned by AddEventHandlerWithOptions, as a stopped
	// informer refuses new handlers.
	addErr error
}

func (c *countingInformer) AddEventHandlerWithOptions(handler cache.ResourceEventHandler, options cache.HandlerOptions) (cache.ResourceEventHandlerRegistration, error) {
	if c.addErr != nil {
		return nil, c.addErr
	}
	c.added++
	return noopRegistration{}, nil
}

func (c *countingInformer) RemoveEventHandler(handle cache.ResourceEventHandlerRegistration) error {
	c.removed++
	return nil
}

func (c *countingInformer) GetController() cache.Controller { return blockingController{} }

// live is the number of event handlers currently registered on the informer.
func (c *countingInformer) live() int { return c.added - c.removed }

// blockingController is a cache.Controller that only waits for its stop
// signal, which is all a monitor needs to be started and stopped.
type blockingController struct{}

func (blockingController) Run(stopCh <-chan struct{})          { <-stopCh }
func (blockingController) RunWithContext(ctx context.Context)  { <-ctx.Done() }
func (blockingController) HasSynced() bool                     { return true }
func (blockingController) HasSyncedChecker() cache.DoneChecker { return nil }
func (blockingController) LastSyncResourceVersion() string     { return "" }

type noopRegistration struct{}

func (noopRegistration) HasSynced() bool                     { return true }
func (noopRegistration) HasSyncedChecker() cache.DoneChecker { return nil }

// genericCountingInformer adapts countingInformer to informers.GenericInformer.
type genericCountingInformer struct {
	informer *countingInformer
}

func (g *genericCountingInformer) Informer() cache.SharedIndexInformer { return g.informer }

func (g *genericCountingInformer) Lister() cache.GenericLister { return nil }

// countingInformerFactory hands out one cached countingInformer per resource,
// so that re-adding the same resource reuses the same informer. This mirrors the
// real (metadata) informer factory, which caches informers by GVR and therefore
// reuses the same informer when a deleted resource (e.g. a CRD) is recreated
// with the same GroupVersionResource.
type countingInformerFactory struct {
	informers map[schema.GroupVersionResource]*countingInformer
}

func newCountingInformerFactory() *countingInformerFactory {
	return &countingInformerFactory{informers: map[schema.GroupVersionResource]*countingInformer{}}
}

func (f *countingInformerFactory) ForResource(resource schema.GroupVersionResource) (informers.GenericInformer, error) {
	ci, ok := f.informers[resource]
	if !ok {
		ci = &countingInformer{}
		f.informers[resource] = ci
	}
	return &genericCountingInformer{informer: ci}, nil
}

func (f *countingInformerFactory) Start(stopCh <-chan struct{}) {}

// TestSyncMonitorsRemovesEventHandlerOnResourceRemoval is a regression test for
// the resource quota controller's QuotaMonitor: SyncMonitors added an event
// handler to the (per-GVR cached) shared informer for every monitored resource,
// but on teardown it only stopped the monitor without removing that handler.
// When a resource such as a CRD is repeatedly created and deleted, the handlers
// pile up on the reused informer and leak, growing kube-controller-manager
// memory. This is the same class of bug fixed for the garbage collector in
// kubernetes/kubernetes#142039 (issue #114066).
func TestSyncMonitorsRemovesEventHandlerOnResourceRemoval(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "widgets"}

	factory := newCountingInformerFactory()
	qm := &QuotaMonitor{
		informerFactory:  factory,
		ignoredResources: map[schema.GroupResource]struct{}{},
		resyncPeriod:     controller.StaticResyncPeriodFunc(0),
		registry:         generic.NewRegistry(nil),
	}

	withResource := map[schema.GroupVersionResource]struct{}{gvr: {}}
	withoutResource := map[schema.GroupVersionResource]struct{}{}

	// Add then remove the resource several times. Each add registers one event
	// handler on the cached informer; each removal must unregister it.
	const cycles = 3
	for i := range cycles {
		if err := qm.SyncMonitors(ctx, withResource); err != nil {
			t.Fatalf("cycle %d: SyncMonitors(add) returned an error: %v", i, err)
		}
		if err := qm.SyncMonitors(ctx, withoutResource); err != nil {
			t.Fatalf("cycle %d: SyncMonitors(remove) returned an error: %v", i, err)
		}
	}

	ci := factory.informers[gvr]
	if ci == nil {
		t.Fatalf("expected an informer to have been created for %s", gvr)
	}
	if ci.added != cycles {
		t.Fatalf("expected %d handler registrations across %d cycles, got %d", cycles, cycles, ci.added)
	}
	if leaked := ci.live(); leaked != 0 {
		t.Fatalf("event handlers leaked: %d handler(s) still registered after %d add/remove cycles (added=%d, removed=%d); monitor teardown must call RemoveEventHandler", leaked, cycles, ci.added, ci.removed)
	}
}

// TestRunTeardownRemovesEventHandlers covers the other teardown site: when Run
// returns because its context was cancelled, it stops every monitor and must
// unregister each monitor's event handler as well.
func TestRunTeardownRemovesEventHandlers(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "widgets"}

	factory := newCountingInformerFactory()
	informersStarted := make(chan struct{})
	close(informersStarted)
	qm := NewMonitor(ctx, informersStarted, factory, map[schema.GroupResource]struct{}{}, controller.StaticResyncPeriodFunc(0), nil, generic.NewRegistry(nil), nil)

	if err := qm.SyncMonitors(ctx, map[schema.GroupVersionResource]struct{}{gvr: {}}); err != nil {
		t.Fatalf("SyncMonitors returned an error: %v", err)
	}

	done := make(chan struct{})
	go func() {
		qm.Run(ctx)
		close(done)
	}()
	cancel()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatalf("Run did not return after its context was cancelled")
	}

	ci := factory.informers[gvr]
	if ci == nil {
		t.Fatalf("expected an informer to have been created for %s", gvr)
	}
	if ci.added != 1 {
		t.Fatalf("expected 1 handler registration, got %d", ci.added)
	}
	if leaked := ci.live(); leaked != 0 {
		t.Fatalf("event handlers leaked on shutdown: %d handler(s) still registered (added=%d, removed=%d); Run must call RemoveEventHandler for every monitor it stops", leaked, ci.added, ci.removed)
	}
}

// TestSyncMonitorsReportsEventHandlerRegistrationError covers a shared informer
// that refuses the event handler, as one that has already stopped does:
// SyncMonitors reports the resource and keeps no monitor for it, instead of
// running a monitor that never receives events.
func TestSyncMonitorsReportsEventHandlerRegistrationError(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "widgets"}

	factory := newCountingInformerFactory()
	factory.informers[gvr] = &countingInformer{addErr: errors.New("informer has stopped")}
	qm := &QuotaMonitor{
		informerFactory:  factory,
		ignoredResources: map[schema.GroupResource]struct{}{},
		resyncPeriod:     controller.StaticResyncPeriodFunc(0),
		registry:         generic.NewRegistry(nil),
	}

	err := qm.SyncMonitors(ctx, map[schema.GroupVersionResource]struct{}{gvr: {}})
	if err == nil {
		t.Fatalf("SyncMonitors returned no error for a resource whose informer refused the event handler")
	}
	for _, want := range []string{gvr.String(), "informer has stopped"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("SyncMonitors error %q does not mention %q", err, want)
		}
	}
	if _, ok := qm.monitors[gvr]; ok {
		t.Fatalf("SyncMonitors kept a monitor for a resource whose event handler was not registered")
	}
}
