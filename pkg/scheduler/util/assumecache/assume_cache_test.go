/*
Copyright 2017 The Kubernetes Authors.

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

package assumecache

import (
	"fmt"
	"slices"
	"sort"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// testInformer implements [Informer] and can be used to feed changes into an assume
// cache during unit testing. Only a single event handler is supported, which is
// sufficient for one assume cache.
type testInformer struct {
	handler cache.ResourceEventHandler
}

func (i *testInformer) AddEventHandler(handler cache.ResourceEventHandler) (cache.ResourceEventHandlerRegistration, error) {
	i.handler = handler
	return nil, nil
}

func (i *testInformer) add(obj interface{}) {
	if i.handler == nil {
		return
	}
	i.handler.OnAdd(obj, false)
}

func (i *testInformer) update(obj interface{}) {
	if i.handler == nil {
		return
	}
	i.handler.OnUpdate(nil, obj)
}

func (i *testInformer) delete(obj interface{}) {
	if i.handler == nil {
		return
	}
	i.handler.OnDelete(obj)
}

func makeObj(name, version, namespace string) metav1.Object {
	return &metav1.ObjectMeta{
		Name:            name,
		Namespace:       namespace,
		ResourceVersion: version,
	}
}

func newTest(t *testing.T) (ktesting.TContext, *AssumeCache, *testInformer) {
	return newTestWithIndexer(t, "", nil)
}

func newTestWithIndexer(t *testing.T, indexName string, indexFunc cache.IndexFunc) (ktesting.TContext, *AssumeCache, *testInformer) {
	tCtx := ktesting.Init(t)
	informer := new(testInformer)
	cache := NewAssumeCache(tCtx.Logger(), informer, "TestObject", indexName, indexFunc)
	return tCtx, cache, informer
}

func verify(tCtx ktesting.TContext, cache *AssumeCache, key string, expectedObject, expectedAPIObject interface{}) {
	tCtx.Helper()
	actualObject, err := cache.Get(key)
	if err != nil {
		tCtx.Fatalf("unexpected error retrieving object for key %s: %v", key, err)
	}
	if actualObject != expectedObject {
		tCtx.Fatalf("Get() returned %v, expected %v", actualObject, expectedObject)
	}
	actualAPIObject, err := cache.GetAPIObj(key)
	if err != nil {
		tCtx.Fatalf("unexpected error retrieving API object for key %s: %v", key, err)
	}
	if actualAPIObject != expectedAPIObject {
		tCtx.Fatalf("GetAPIObject() returned %v, expected %v", actualAPIObject, expectedAPIObject)
	}
}

func verifyList(tCtx ktesting.TContext, assumeCache *AssumeCache, expectedObjs []interface{}, indexObj interface{}) {
	actualObjs := assumeCache.List(indexObj)
	diff := cmp.Diff(expectedObjs, actualObjs, cmpopts.SortSlices(func(x, y interface{}) bool {
		xKey, err := cache.MetaNamespaceKeyFunc(x)
		if err != nil {
			tCtx.Fatalf("unexpected error determining key for %v: %v", x, err)
		}
		yKey, err := cache.MetaNamespaceKeyFunc(y)
		if err != nil {
			tCtx.Fatalf("unexpected error determining key for %v: %v", y, err)
		}
		return xKey < yKey
	}))
	if diff != "" {
		tCtx.Fatalf("List() result differs (- expected, + actual):\n%s", diff)
	}
}

type mockEventHandler struct {
	mutex  sync.Mutex
	events []event
	cache  *AssumeCache
	block  <-chan struct{}
}

type event struct {
	What        string
	OldObj, Obj interface{}
	InitialList bool
}

func (m *mockEventHandler) OnAdd(obj interface{}, initialList bool) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.events = append(m.events, event{
		What:        "add",
		Obj:         obj,
		InitialList: initialList,
	})

	if m.cache != nil {
		// Must not deadlock!
		m.cache.List(nil)
	}
	if m.block != nil {
		<-m.block
	}
}

func (m *mockEventHandler) OnUpdate(oldObj, obj interface{}) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.events = append(m.events, event{
		What:   "update",
		OldObj: oldObj,
		Obj:    obj,
	})
}

func (m *mockEventHandler) OnDelete(obj interface{}) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.events = append(m.events, event{
		What: "delete",
		Obj:  obj,
	})
}

func (m *mockEventHandler) verifyAndFlush(tCtx ktesting.TContext, expectedEvents []event) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	tCtx.Helper()
	if diff := cmp.Diff(expectedEvents, m.events); diff != "" {
		tCtx.Fatalf("unexpected events (- expected, + actual):\n%s", diff)
	}
	m.events = nil
}

func (m *mockEventHandler) sortEvents(cmp func(objI, objJ interface{}) bool) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	sort.Slice(m.events, func(i, j int) bool {
		return cmp(m.events[i].Obj, m.events[j].Obj)
	})
}

func TestAssume(t *testing.T) {
	scenarios := map[string]struct {
		oldObj    metav1.Object
		newObj    interface{}
		expectErr error
	}{
		"success-same-version": {
			oldObj: makeObj("pvc1", "5", ""),
			newObj: makeObj("pvc1", "5", ""),
		},
		"success-new-higher-version": {
			oldObj: makeObj("pvc1", "5", ""),
			newObj: makeObj("pvc1", "6", ""),
		},
		"fail-old-not-found": {
			oldObj:    makeObj("pvc2", "5", ""),
			newObj:    makeObj("pvc1", "5", ""),
			expectErr: ErrNotFound,
		},
		"fail-new-lower-version": {
			oldObj:    makeObj("pvc1", "5", ""),
			newObj:    makeObj("pvc1", "4", ""),
			expectErr: cmpopts.AnyError,
		},
		"fail-new-bad-version": {
			oldObj:    makeObj("pvc1", "5", ""),
			newObj:    makeObj("pvc1", "a", ""),
			expectErr: cmpopts.AnyError,
		},
		"fail-old-bad-version": {
			oldObj:    makeObj("pvc1", "a", ""),
			newObj:    makeObj("pvc1", "5", ""),
			expectErr: cmpopts.AnyError,
		},
		"fail-new-bad-object": {
			oldObj:    makeObj("pvc1", "5", ""),
			newObj:    1,
			expectErr: ErrObjectName,
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			tCtx, cache, informer := newTest(t)
			var events mockEventHandler
			cache.AddEventHandler(&events)

			// Add old object to cache.
			informer.add(scenario.oldObj)
			verify(tCtx, cache, scenario.oldObj.GetName(), scenario.oldObj, scenario.oldObj)

			// Assume new object.
			err := cache.Assume(scenario.newObj)
			if diff := cmp.Diff(scenario.expectErr, err, cmpopts.EquateErrors()); diff != "" {
				t.Errorf("Assume() returned error: %v\ndiff (- expected, + actual):\n%s", err, diff)
			}

			// Check that Get returns correct object and
			// that events were delivered correctly.
			expectEvents := []event{{What: "add", Obj: scenario.oldObj}}
			expectedObj := scenario.newObj
			if scenario.expectErr != nil {
				expectedObj = scenario.oldObj
			} else {
				expectEvents = append(expectEvents, event{What: "update", OldObj: scenario.oldObj, Obj: scenario.newObj})
			}
			verify(tCtx, cache, scenario.oldObj.GetName(), expectedObj, scenario.oldObj)
			events.verifyAndFlush(tCtx, expectEvents)
		})
	}
}

func TestRestore(t *testing.T) {
	tCtx, cache, informer := newTest(t)
	var events mockEventHandler
	cache.AddEventHandler(&events)

	// This test assumes an object with the same version as the API object.
	// The assume cache supports that, but doing so in real code suffers from
	// a race: if an unrelated update is received from the apiserver while
	// such an object is assumed, the local modification gets dropped.
	oldObj := makeObj("pvc1", "5", "")
	newObj := makeObj("pvc1", "5", "")

	// Restore object that doesn't exist
	ktesting.Step(tCtx, "empty cache", func(tCtx ktesting.TContext) {
		cache.Restore("nothing")
		events.verifyAndFlush(tCtx, nil)
	})

	// Add old object to cache.
	ktesting.Step(tCtx, "initial update", func(tCtx ktesting.TContext) {
		informer.add(oldObj)
		verify(tCtx, cache, oldObj.GetName(), oldObj, oldObj)
		events.verifyAndFlush(tCtx, []event{{What: "add", Obj: oldObj}})
	})

	// Restore the same object.
	ktesting.Step(tCtx, "initial Restore", func(tCtx ktesting.TContext) {
		cache.Restore(oldObj.GetName())
		verify(tCtx, cache, oldObj.GetName(), oldObj, oldObj)
		events.verifyAndFlush(tCtx, nil)
	})

	// Assume new object.
	ktesting.Step(tCtx, "Assume", func(tCtx ktesting.TContext) {
		if err := cache.Assume(newObj); err != nil {
			tCtx.Fatalf("Assume() returned error %v", err)
		}
		verify(tCtx, cache, oldObj.GetName(), newObj, oldObj)
		events.verifyAndFlush(tCtx, []event{{What: "update", OldObj: oldObj, Obj: newObj}})
	})

	// Restore the same object.
	ktesting.Step(tCtx, "second Restore", func(tCtx ktesting.TContext) {
		cache.Restore(oldObj.GetName())
		verify(tCtx, cache, oldObj.GetName(), oldObj, oldObj)
		events.verifyAndFlush(tCtx, []event{{What: "update", OldObj: newObj, Obj: oldObj}})
	})
}

func TestEvents(t *testing.T) {
	tCtx, cache, informer := newTest(t)

	oldObj := makeObj("pvc1", "5", "")
	newObj := makeObj("pvc1", "6", "")
	key := oldObj.GetName()

	// Add old object to cache.
	informer.add(oldObj)
	verify(ktesting.WithStep(tCtx, "after initial update"), cache, key, oldObj, oldObj)

	// Receive initial list.
	var events mockEventHandler
	cache.AddEventHandler(&events)
	events.verifyAndFlush(ktesting.WithStep(tCtx, "initial list"), []event{{What: "add", Obj: oldObj, InitialList: true}})

	// Update object.
	ktesting.Step(tCtx, "initial update", func(tCtx ktesting.TContext) {
		informer.update(newObj)
		verify(tCtx, cache, key, newObj, newObj)
		events.verifyAndFlush(tCtx, []event{{What: "update", OldObj: oldObj, Obj: newObj}})
	})

	// Some error cases (don't occur in practice).
	ktesting.Step(tCtx, "nop add", func(tCtx ktesting.TContext) {
		informer.add(1)
		verify(tCtx, cache, key, newObj, newObj)
		events.verifyAndFlush(tCtx, nil)
	})
	ktesting.Step(tCtx, "nil add", func(tCtx ktesting.TContext) {
		informer.add(nil)
		verify(tCtx, cache, key, newObj, newObj)
		events.verifyAndFlush(tCtx, nil)
	})
	ktesting.Step(tCtx, "nop update", func(tCtx ktesting.TContext) {
		informer.update(oldObj)
		events.verifyAndFlush(tCtx, nil)
		verify(tCtx, cache, key, newObj, newObj)
	})
	ktesting.Step(tCtx, "nil update", func(tCtx ktesting.TContext) {
		informer.update(nil)
		verify(tCtx, cache, key, newObj, newObj)
		events.verifyAndFlush(tCtx, nil)
	})
	ktesting.Step(tCtx, "nop delete", func(tCtx ktesting.TContext) {
		informer.delete(nil)
		verify(tCtx, cache, key, newObj, newObj)
		events.verifyAndFlush(tCtx, nil)
	})

	// Delete object.
	ktesting.Step(tCtx, "delete", func(tCtx ktesting.TContext) {
		informer.delete(oldObj)
		events.verifyAndFlush(tCtx, []event{{What: "delete", Obj: newObj}})
		_, err := cache.Get(key)
		if diff := cmp.Diff(ErrNotFound, err, cmpopts.EquateErrors()); diff != "" {
			tCtx.Errorf("Get did not return expected error: %v\ndiff (- expected, + actual):\n%s", err, diff)
		}
	})
}

func TestEventHandlers(t *testing.T) {
	tCtx, cache, informer := newTest(t)
	handlers := make([]mockEventHandler, 5)

	var objs []metav1.Object
	for i := 0; i < 5; i++ {
		objs = append(objs, makeObj(fmt.Sprintf("test-pvc%v", i), "1", ""))
		informer.add(objs[i])
	}

	// Accessing cache during OnAdd must not deadlock!
	handlers[0].cache = cache

	// Order of delivered events is random, we must ensure
	// increasing order by name ourselves.
	var expectedEvents []event
	for _, obj := range objs {
		expectedEvents = append(expectedEvents,
			event{
				What:        "add",
				Obj:         obj,
				InitialList: true,
			},
		)
	}
	for i := range handlers {
		cache.AddEventHandler(&handlers[i])
		handlers[i].sortEvents(func(objI, objJ interface{}) bool {
			return objI.(*metav1.ObjectMeta).Name <
				objJ.(*metav1.ObjectMeta).Name
		})
		handlers[i].verifyAndFlush(tCtx, expectedEvents)
	}

	for i := 5; i < 7; i++ {
		objs = append(objs, makeObj(fmt.Sprintf("test-pvc%v", i), "1", ""))
		informer.add(objs[i])
		for e := range handlers {
			handlers[e].verifyAndFlush(tCtx, []event{{What: "add", Obj: objs[i]}})
		}
	}

	for i, oldObj := range objs {
		newObj := makeObj(fmt.Sprintf("test-pvc%v", i), "2", "")
		objs[i] = newObj
		informer.update(newObj)
		for e := range handlers {
			handlers[e].verifyAndFlush(tCtx, []event{{What: "update", OldObj: oldObj, Obj: newObj}})
		}
	}

	for _, obj := range objs {
		informer.delete(obj)
		for e := range handlers {
			handlers[e].verifyAndFlush(tCtx, []event{{What: "delete", Obj: obj}})
		}
	}
}

func TestEventHandlerConcurrency(t *testing.T) {
	tCtx, cache, informer := newTest(t)
	handlers := make([]mockEventHandler, 5)

	var objs []metav1.Object
	for i := 0; i < 5; i++ {
		objs = append(objs, makeObj(fmt.Sprintf("test-pvc%v", i), "1", ""))
	}

	// Accessing cache during OnAdd must not deadlock!
	handlers[0].cache = cache

	// Each add blocks until this gets cancelled.
	tCancelCtx := ktesting.WithCancel(tCtx)
	var wg sync.WaitGroup

	for i := range handlers {
		handlers[i].block = tCancelCtx.Done()
		cache.AddEventHandler(&handlers[i])
	}

	// Execution of the add calls is random, therefore
	// we have to sort again.
	var expectedEvents []event
	for _, obj := range objs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			informer.add(obj)
		}()
		expectedEvents = append(expectedEvents,
			event{
				What: "add",
				Obj:  obj,
			},
		)
	}

	tCancelCtx.Cancel("proceed")
	wg.Wait()

	for i := range handlers {
		handlers[i].sortEvents(func(objI, objJ interface{}) bool {
			return objI.(*metav1.ObjectMeta).Name <
				objJ.(*metav1.ObjectMeta).Name
		})
		handlers[i].verifyAndFlush(tCtx, expectedEvents)
	}
}

func TestListNoIndexer(t *testing.T) {
	tCtx, cache, informer := newTest(t)

	// Add a bunch of objects.
	var objs []interface{}
	for i := 0; i < 10; i++ {
		obj := makeObj(fmt.Sprintf("test-pvc%v", i), "1", "")
		objs = append(objs, obj)
		informer.add(obj)
	}

	// List them
	verifyList(ktesting.WithStep(tCtx, "after add"), cache, objs, "")

	// Update an object.
	updatedObj := makeObj("test-pvc3", "2", "")
	objs[3] = updatedObj
	informer.update(updatedObj)

	// List them
	verifyList(ktesting.WithStep(tCtx, "after update"), cache, objs, "")

	// Delete a PV
	deletedObj := objs[7]
	objs = slices.Delete(objs, 7, 8)
	informer.delete(deletedObj)

	// List them
	verifyList(ktesting.WithStep(tCtx, "after delete"), cache, objs, "")
}

func TestListWithIndexer(t *testing.T) {
	namespaceIndexer := func(obj interface{}) ([]string, error) {
		objAccessor, err := meta.Accessor(obj)
		if err != nil {
			return nil, err
		}
		return []string{objAccessor.GetNamespace()}, nil
	}
	tCtx, cache, informer := newTestWithIndexer(t, "myNamespace", namespaceIndexer)

	// Add a bunch of objects.
	ns := "ns1"
	var objs []interface{}
	for i := 0; i < 10; i++ {
		obj := makeObj(fmt.Sprintf("test-pvc%v", i), "1", ns)
		objs = append(objs, obj)
		informer.add(obj)
	}

	// Add a bunch of other objects.
	for i := 0; i < 10; i++ {
		obj := makeObj(fmt.Sprintf("test-pvc%v", i), "1", "ns2")
		informer.add(obj)
	}

	// List them
	verifyList(ktesting.WithStep(tCtx, "after add"), cache, objs, objs[0])

	// Update an object.
	updatedObj := makeObj("test-pvc3", "2", ns)
	objs[3] = updatedObj
	informer.update(updatedObj)

	// List them
	verifyList(ktesting.WithStep(tCtx, "after update"), cache, objs, objs[0])

	// Delete a PV
	deletedObj := objs[7]
	objs = slices.Delete(objs, 7, 8)
	informer.delete(deletedObj)

	// List them
	verifyList(ktesting.WithStep(tCtx, "after delete"), cache, objs, objs[0])
}
