/*
Copyright 2026 The Kubernetes Authors.

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

package garbagecollector

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2/ktesting"
)

// countingInformer is a minimal SharedIndexInformer that records how many event
// handlers have been added and removed, so a test can detect handlers that are
// leaked when a monitor is torn down. Only the methods exercised by
// GraphBuilder.controllerFor and the monitor teardown are implemented; the rest
// are inherited from the embedded interface and panic if called.
type countingInformer struct {
	cache.SharedIndexInformer
	added   int
	removed int
}

func (c *countingInformer) AddEventHandlerWithOptions(handler cache.ResourceEventHandler, options cache.HandlerOptions) (cache.ResourceEventHandlerRegistration, error) {
	c.added++
	return noopRegistration{}, nil
}

func (c *countingInformer) RemoveEventHandler(handle cache.ResourceEventHandlerRegistration) error {
	c.removed++
	return nil
}

func (c *countingInformer) GetController() cache.Controller { return nil }

func (c *countingInformer) GetStore() cache.Store { return nil }

// live is the number of event handlers currently registered on the informer.
func (c *countingInformer) live() int { return c.added - c.removed }

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
// so that re-adding the same resource reuses the same informer. This mirrors
// the real (metadata) informer factory, which caches informers by GVR and
// therefore reuses the same informer when a deleted resource (e.g. a CRD) is
// recreated with the same GroupVersionResource.
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
// https://github.com/kubernetes/kubernetes/issues/114066: syncMonitors added an
// event handler to the (per-GVR cached) shared informer for every resource, but
// on teardown it only stopped the monitor without removing that handler. When a
// resource such as a CRD is repeatedly created and deleted, the handlers pile up
// on the reused informer and leak, growing kube-controller-manager memory.
func TestSyncMonitorsRemovesEventHandlerOnResourceRemoval(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "widgets"}
	gvk := schema.GroupVersionKind{Group: "example.com", Version: "v1", Kind: "Widget"}

	rm := meta.NewDefaultRESTMapper(nil)
	rm.AddSpecific(gvk, gvr, schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "widget"}, meta.RESTScopeNamespace)

	factory := newCountingInformerFactory()
	gb := &GraphBuilder{
		restMapper:       rm,
		sharedInformers:  factory,
		ignoredResources: map[schema.GroupResource]struct{}{},
	}

	withResource := map[schema.GroupVersionResource]struct{}{gvr: {}}
	withoutResource := map[schema.GroupVersionResource]struct{}{}

	// Add then remove the resource several times. Each add registers one event
	// handler on the cached informer; each removal must unregister it.
	const cycles = 3
	for i := 0; i < cycles; i++ {
		if err := gb.syncMonitors(logger, withResource); err != nil {
			t.Fatalf("cycle %d: syncMonitors(add) returned an error: %v", i, err)
		}
		if err := gb.syncMonitors(logger, withoutResource); err != nil {
			t.Fatalf("cycle %d: syncMonitors(remove) returned an error: %v", i, err)
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
