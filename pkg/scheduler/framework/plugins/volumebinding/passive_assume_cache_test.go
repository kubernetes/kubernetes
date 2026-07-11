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

package volumebinding

import (
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-helpers/storage/volume"
	"k8s.io/klog/v2/ktesting"
)

// sufficient for one assume cache.
type testInformer struct {
	handler cache.ResourceEventHandler
	indexer cache.Indexer
	t       *testing.T
}

func (i *testInformer) AddEventHandler(handler cache.ResourceEventHandler) (cache.ResourceEventHandlerRegistration, error) {
	i.handler = handler
	return nil, nil
}

func (i *testInformer) GetIndexer() cache.Indexer {
	return i.indexer
}

func (i *testInformer) add(obj interface{}) {
	if err := i.indexer.Add(obj); err != nil {
		i.t.Fatalf("failed to add object into indexer: %v", err)
	}
	i.handler.OnAdd(obj, false)
}

func (i *testInformer) update(oldObj, obj interface{}) {
	if err := i.indexer.Update(obj); err != nil {
		i.t.Fatalf("failed to update object to indexer: %v", err)
	}
	i.handler.OnUpdate(oldObj, obj)
}

func (i *testInformer) delete(obj interface{}) {
	if err := i.indexer.Delete(obj); err != nil {
		i.t.Fatalf("failed to delete object from indexer: %v", err)
	}
	i.handler.OnDelete(obj)
}

func verifyList(t *testing.T, cache testCache, expected map[string]*testObj, indexedValue string) {
	t.Helper()
	got, err := cache.ByIndex(testIndex, indexedValue)
	if err != nil {
		t.Fatalf("failed to get indexed objects: %v", err)
	}
	if len(got) != len(expected) {
		t.Errorf("ByIndex() returned %v objects, expected %v", len(got), len(expected))
	}
	for _, obj := range got {
		expected, ok := expected[obj.Name]
		if !ok {
			t.Errorf("ByIndex() returned unexpected object %q", obj.Name)
		}
		if expected != obj {
			t.Errorf("ByIndex() returned object %p, expected %p", obj, expected)
		}
	}
}

func TestBasicCache(t *testing.T) {
	informer, cache := newTestCache(t)

	// Get object that doesn't exist
	obj, err := cache.Get("nothere")
	if err == nil {
		t.Errorf("Get() returned unexpected success")
	}
	if obj != nil {
		t.Errorf("Get() returned unexpected PV %q", obj.Name)
	}

	// Add a bunch of objects
	objects := map[string]*testObj{}
	for i := range 10 {
		obj := makeObj(fmt.Sprintf("test-%v", i), "1", "")
		objects[obj.Name] = obj
		informer.add(obj)
	}

	// List them
	verifyList(t, cache, objects, "")

	// Update an object
	updated := makeObj("test-3", "1", "")
	informer.update(objects["test-3"], updated)
	objects[updated.Name] = updated

	// List them
	verifyList(t, cache, objects, "")

	// Delete an object
	deleted := objects["test-7"]
	delete(objects, deleted.Name)
	informer.delete(deleted)

	// List them
	verifyList(t, cache, objects, "")
}

func TestPVCacheWithIndex(t *testing.T) {
	informer, cache := newTestCache(t)

	// Add a bunch of objects
	objects1 := map[string]*testObj{}
	for i := range 10 {
		obj := makeObj(fmt.Sprintf("test-%v", i), "1", "")
		obj.Annotations["test"] = "test1"
		objects1[obj.Name] = obj
		informer.add(obj)
	}

	// Add a bunch of objects
	objects2 := map[string]*testObj{}
	for i := range 10 {
		obj := makeObj(fmt.Sprintf("test2-%v", i), "1", "")
		obj.Annotations["test"] = "test2"
		objects2[obj.Name] = obj
		informer.add(obj)
	}

	// List them
	verifyList(t, cache, objects1, "test1")
	verifyList(t, cache, objects2, "test2")

	// Update an object
	updated := makeObj("test-3", "2", "")
	updated.Annotations["test"] = "test1"
	informer.update(objects1[updated.Name], updated)
	objects1[updated.Name] = updated

	// List them
	verifyList(t, cache, objects1, "test1")
	verifyList(t, cache, objects2, "test2")

	// Delete an object
	deletedPV := objects1["test-7"]
	delete(objects1, deletedPV.Name)
	informer.delete(deletedPV)

	// List them
	verifyList(t, cache, objects1, "test1")
	verifyList(t, cache, objects2, "test2")
}

type testObj = metav1.ObjectMeta

func makeObj(name, version, namespace string) *testObj {
	return &testObj{
		Name:            name,
		Namespace:       namespace,
		ResourceVersion: version,
		Annotations:     map[string]string{},
	}
}

type testCache = *passiveAssumeCache[*testObj]

func verifyObj(cache testCache, key string, expected *testObj) error {
	obj, err := cache.Get(key)
	if err != nil {
		return err
	}
	if obj != expected {
		return fmt.Errorf("Get() returned %p, expected %p", obj, expected)
	}
	return nil
}

const testIndex = "testIndex"

func newTestCache(t *testing.T) (*testInformer, testCache) {
	logger, _ := ktesting.NewTestContext(t)
	informer := &testInformer{
		indexer: cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{
			testIndex: func(obj interface{}) ([]string, error) {
				return []string{obj.(*testObj).Annotations["test"]}, nil
			},
		}),
		t: t,
	}
	cache, err := newAssumeCache[*testObj](logger, informer, schema.GroupResource{Resource: "tests"})
	if err != nil {
		t.Fatalf("newAssumeCache() failed: %v", err)
	}
	return informer, cache
}

func TestAssume(t *testing.T) {
	scenarios := map[string]struct {
		old           *testObj
		new           *testObj
		shouldSucceed bool
	}{
		"success-same-version": {
			old:           makeObj("pvc1", "5", "ns1"),
			new:           makeObj("pvc1", "5", "ns1"),
			shouldSucceed: true,
		},
		"fail-new-higher-version": {
			old:           makeObj("pvc1", "5", "ns1"),
			new:           makeObj("pvc1", "6", "ns1"),
			shouldSucceed: false,
		},
		"fail-old-not-found": {
			old:           makeObj("pvc2", "5", "ns1"),
			new:           makeObj("pvc1", "5", "ns1"),
			shouldSucceed: false,
		},
		"fail-new-lower-version": {
			old:           makeObj("pvc1", "5", "ns1"),
			new:           makeObj("pvc1", "4", "ns1"),
			shouldSucceed: false,
		},
		"fail-new-bad-version": {
			old:           makeObj("pvc1", "5", "ns1"),
			new:           makeObj("pvc1", "a", "ns1"),
			shouldSucceed: false,
		},
		"fail-old-bad-version": {
			old:           makeObj("pvc1", "a", "ns1"),
			new:           makeObj("pvc1", "5", "ns1"),
			shouldSucceed: false,
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			informer, cache := newTestCache(t)

			// Add old to cache
			informer.add(scenario.old)
			if err := verifyObj(cache, keyOf(scenario.old), scenario.old); err != nil {
				t.Fatalf("Failed to Get() after initial update: %v", err)
			}

			// Assume new
			err := cache.Assume(scenario.new)
			if scenario.shouldSucceed && err != nil {
				t.Errorf("Test %q failed: Assume() returned error %v", name, err)
			}
			if !scenario.shouldSucceed && err == nil {
				t.Errorf("Test %q failed: Assume() returned success but expected error", name)
			}

			// Check that Get returns correct version
			expectedPV := scenario.new
			if !scenario.shouldSucceed {
				expectedPV = scenario.old
			}
			if err := verifyObj(cache, keyOf(scenario.old), expectedPV); err != nil {
				t.Errorf("Failed to Get() after initial update: %v", err)
			}
		})
	}
}

func TestRestore(t *testing.T) {
	informer, cache := newTestCache(t)

	old := makeObj("pvc1", "5", "ns1")
	new := makeObj("pvc1", "5", "ns1")

	// Restore object that doesn't exist
	cache.Restore(&testObj{})

	// Add old to cache
	informer.add(old)
	if err := verifyObj(cache, keyOf(old), old); err != nil {
		t.Fatalf("Failed to Get() after initial update: %v", err)
	}

	// Restore
	cache.Restore(old)
	if err := verifyObj(cache, keyOf(old), old); err != nil {
		t.Fatalf("Failed to Get() after initial restore: %v", err)
	}

	// Assume new
	if err := cache.Assume(new); err != nil {
		t.Fatalf("Assume() returned error %v", err)
	}
	if err := verifyObj(cache, keyOf(old), new); err != nil {
		t.Fatalf("Failed to Get() after Assume: %v", err)
	}

	// Restore
	cache.Restore(new)
	if err := verifyObj(cache, keyOf(old), old); err != nil {
		t.Fatalf("Failed to Get() after restore: %v", err)
	}
}

func TestConcurrentAssume(t *testing.T) {
	informer, cache := newTestCache(t)

	obj1 := makeObj("pvc1", "5", "ns1")
	obj1Update := makeObj("pvc1", "5", "ns1")
	// Add object to cache
	informer.add(obj1)

	// Update obj 1
	if err := cache.Assume(obj1Update); err != nil {
		t.Fatalf("Assume() returned error %v", err)
	}
	if err := verifyObj(cache, keyOf(obj1Update), obj1Update); err != nil {
		t.Fatalf("Failed to Get() after Assume: %v", err)
	}

	obj2 := makeObj("pvc1", "7", "ns1")
	obj2Update := makeObj("pvc1", "7", "ns1")
	// obj updated externally
	informer.add(obj2)

	// Update obj 2
	if err := cache.Assume(obj2Update); err != nil {
		t.Fatalf("Assume() returned error %v", err)
	}
	// obj 1 failed with conflict
	cache.Restore(obj1Update)
	// Should still have obj 2 in cache
	if err := verifyObj(cache, keyOf(obj2Update), obj2Update); err != nil {
		t.Fatalf("Failed to Get() after restore: %v", err)
	}
}

func TestAssumeUpdateCache(t *testing.T) {
	informer, cache := newTestCache(t)

	// Add a object
	obj := makeObj("test-pvc0", "1", "test-ns")
	informer.add(obj)
	if err := verifyObj(cache, keyOf(obj), obj); err != nil {
		t.Fatalf("failed to get: %v", err)
	}

	// Assume
	newObj := obj.DeepCopy()
	newObj.Annotations[volume.AnnSelectedNode] = "test-node"
	if err := cache.Assume(newObj); err != nil {
		t.Fatalf("failed to assume: %v", err)
	}
	if err := verifyObj(cache, keyOf(obj), newObj); err != nil {
		t.Fatalf("failed to get after assume: %v", err)
	}

	// Add old
	informer.add(obj)
	if err := verifyObj(cache, keyOf(obj), newObj); err != nil {
		t.Fatalf("failed to get after old added: %v", err)
	}
}

func TestDelayedInformerEvent(t *testing.T) {
	informer, cache := newTestCache(t)

	obj1 := makeObj("test-pvc0", "1", "test-ns")
	obj2 := makeObj("test-pvc0", "2", "test-ns")
	// Only add indexer, simulating delayed informer event
	if err := informer.indexer.Add(obj2); err != nil {
		t.Fatalf("failed to add: %v", err)
	}

	newObj := obj2.DeepCopy()
	newObj.Annotations[volume.AnnSelectedNode] = "test-node"
	if err := cache.Assume(newObj); err != nil {
		t.Fatalf("failed to assume: %v", err)
	}

	// Send the delayed event
	informer.handler.OnAdd(obj1, false)
	informer.handler.OnDelete(obj1)
	informer.handler.OnAdd(obj2, false)
	// Expect assumed version not overwritten
	if err := verifyObj(cache, keyOf(newObj), newObj); err != nil {
		t.Fatalf("failed to get after assume: %v", err)
	}
}
