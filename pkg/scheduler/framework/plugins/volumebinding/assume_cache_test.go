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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

func newTestPVCache(t *testing.T) (*testInformer, PVAssumeCache) {
	logger, _ := ktesting.NewTestContext(t)
	informer := &testInformer{
		indexer: cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{}),
		t:       t,
	}
	cache, err := NewPVAssumeCache(logger, informer)
	if err != nil {
		t.Fatalf("NewPVAssumeCache() failed: %v", err)
	}
	return informer, cache
}

func verifyListPVs(t *testing.T, cache PVAssumeCache, expectedPVs map[string]*v1.PersistentVolume, storageClassName string) {
	pvList, err := cache.ListPVs(storageClassName)
	if err != nil {
		t.Errorf("ListPVs() failed: %v", err)
	}
	if len(pvList) != len(expectedPVs) {
		t.Errorf("ListPVs() returned %v PVs, expected %v", len(pvList), len(expectedPVs))
	}
	for _, pv := range pvList {
		expectedPV, ok := expectedPVs[pv.Name]
		if !ok {
			t.Errorf("ListPVs() returned unexpected PV %q", pv.Name)
		}
		if expectedPV != pv {
			t.Errorf("ListPVs() returned PV %p, expected %p", pv, expectedPV)
		}
	}
}

func verifyPV(cache PVAssumeCache, name string, expectedPV *v1.PersistentVolume) error {
	pv, err := cache.Get(name)
	if err != nil {
		return err
	}
	if pv != expectedPV {
		return fmt.Errorf("Get() returned %p, expected %p", pv, expectedPV)
	}
	return nil
}

func TestAssumePV(t *testing.T) {
	scenarios := map[string]struct {
		oldPV         *v1.PersistentVolume
		newPV         *v1.PersistentVolume
		shouldSucceed bool
	}{
		"success-same-version": {
			oldPV:         makePV("pv1", "").withVersion("5").PersistentVolume,
			newPV:         makePV("pv1", "").withVersion("5").PersistentVolume,
			shouldSucceed: true,
		},
		"success-storageclass-same-version": {
			oldPV:         makePV("pv1", "class1").withVersion("5").PersistentVolume,
			newPV:         makePV("pv1", "class1").withVersion("5").PersistentVolume,
			shouldSucceed: true,
		},
		"fail-new-higher-version": {
			oldPV:         makePV("pv1", "").withVersion("5").PersistentVolume,
			newPV:         makePV("pv1", "").withVersion("6").PersistentVolume,
			shouldSucceed: false,
		},
		"fail-old-not-found": {
			oldPV:         makePV("pv2", "").withVersion("5").PersistentVolume,
			newPV:         makePV("pv1", "").withVersion("5").PersistentVolume,
			shouldSucceed: false,
		},
		"fail-new-lower-version": {
			oldPV:         makePV("pv1", "").withVersion("5").PersistentVolume,
			newPV:         makePV("pv1", "").withVersion("4").PersistentVolume,
			shouldSucceed: false,
		},
		"fail-new-bad-version": {
			oldPV:         makePV("pv1", "").withVersion("5").PersistentVolume,
			newPV:         makePV("pv1", "").withVersion("a").PersistentVolume,
			shouldSucceed: false,
		},
		"fail-old-bad-version": {
			oldPV:         makePV("pv1", "").withVersion("a").PersistentVolume,
			newPV:         makePV("pv1", "").withVersion("5").PersistentVolume,
			shouldSucceed: false,
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			informer, cache := newTestPVCache(t)
			// Add oldPV to cache
			informer.add(scenario.oldPV)
			if err := verifyPV(cache, scenario.oldPV.Name, scenario.oldPV); err != nil {
				t.Fatalf("Failed to Get() after initial update: %v", err)
			}

			// Assume newPV
			err := cache.Assume(scenario.newPV)
			if scenario.shouldSucceed && err != nil {
				t.Errorf("Test %q failed: Assume() returned error %v", name, err)
			}
			if !scenario.shouldSucceed && err == nil {
				t.Errorf("Test %q failed: Assume() returned success but expected error", name)
			}

			// Check that Get returns correct PV
			expectedPV := scenario.newPV
			if !scenario.shouldSucceed {
				expectedPV = scenario.oldPV
			}
			if err := verifyPV(cache, scenario.oldPV.Name, expectedPV); err != nil {
				t.Errorf("Failed to Get() after initial update: %v", err)
			}
		})
	}
}

func TestRestorePV(t *testing.T) {
	informer, cache := newTestPVCache(t)

	oldPV := makePV("pv1", "").withVersion("5").PersistentVolume
	newPV := makePV("pv1", "").withVersion("5").PersistentVolume

	// Restore PV that doesn't exist
	cache.Restore(&v1.PersistentVolume{})

	// Add oldPV to cache
	informer.add(oldPV)
	if err := verifyPV(cache, oldPV.Name, oldPV); err != nil {
		t.Fatalf("Failed to Get() after initial update: %v", err)
	}

	// Restore PV
	cache.Restore(oldPV)
	if err := verifyPV(cache, oldPV.Name, oldPV); err != nil {
		t.Fatalf("Failed to Get() after initial restore: %v", err)
	}

	// Assume newPV
	if err := cache.Assume(newPV); err != nil {
		t.Fatalf("Assume() returned error %v", err)
	}
	if err := verifyPV(cache, oldPV.Name, newPV); err != nil {
		t.Fatalf("Failed to Get() after Assume: %v", err)
	}

	// Restore PV
	cache.Restore(newPV)
	if err := verifyPV(cache, oldPV.Name, oldPV); err != nil {
		t.Fatalf("Failed to Get() after restore: %v", err)
	}
}

func TestBasicPVCache(t *testing.T) {
	informer, cache := newTestPVCache(t)

	// Get object that doesn't exist
	pv, err := cache.Get("nothere")
	if err == nil {
		t.Errorf("Get() returned unexpected success")
	}
	if pv != nil {
		t.Errorf("Get() returned unexpected PV %q", pv.Name)
	}

	// Add a bunch of PVs
	pvs := map[string]*v1.PersistentVolume{}
	for i := 0; i < 10; i++ {
		pv := makePV(fmt.Sprintf("test-pv%v", i), "").withVersion("1").PersistentVolume
		pvs[pv.Name] = pv
		informer.add(pv)
	}

	// List them
	verifyListPVs(t, cache, pvs, "")

	// Update a PV
	updatedPV := makePV("test-pv3", "").withVersion("2").PersistentVolume
	informer.update(pvs["test-pv3"], updatedPV)
	pvs[updatedPV.Name] = updatedPV

	// List them
	verifyListPVs(t, cache, pvs, "")

	// Delete a PV
	deletedPV := pvs["test-pv7"]
	delete(pvs, deletedPV.Name)
	informer.delete(deletedPV)

	// List them
	verifyListPVs(t, cache, pvs, "")
}

func TestPVCacheWithStorageClasses(t *testing.T) {
	informer, cache := newTestPVCache(t)

	// Add a bunch of PVs
	pvs1 := map[string]*v1.PersistentVolume{}
	for i := 0; i < 10; i++ {
		pv := makePV(fmt.Sprintf("test-pv%v", i), "class1").withVersion("1").PersistentVolume
		pvs1[pv.Name] = pv
		informer.add(pv)
	}

	// Add a bunch of PVs
	pvs2 := map[string]*v1.PersistentVolume{}
	for i := 0; i < 10; i++ {
		pv := makePV(fmt.Sprintf("test2-pv%v", i), "class2").withVersion("1").PersistentVolume
		pvs2[pv.Name] = pv
		informer.add(pv)
	}

	// List them
	verifyListPVs(t, cache, pvs1, "class1")
	verifyListPVs(t, cache, pvs2, "class2")

	// Update a PV
	updatedPV := makePV("test-pv3", "class1").withVersion("2").PersistentVolume
	informer.update(pvs1[updatedPV.Name], updatedPV)
	pvs1[updatedPV.Name] = updatedPV

	// List them
	verifyListPVs(t, cache, pvs1, "class1")
	verifyListPVs(t, cache, pvs2, "class2")

	// Delete a PV
	deletedPV := pvs1["test-pv7"]
	delete(pvs1, deletedPV.Name)
	informer.delete(deletedPV)

	// List them
	verifyListPVs(t, cache, pvs1, "class1")
	verifyListPVs(t, cache, pvs2, "class2")
}

func TestAssumeUpdatePVCache(t *testing.T) {
	informer, cache := newTestPVCache(t)

	pvName := "test-pv0"

	// Add a PV
	pv := makePV(pvName, "").withVersion("1").PersistentVolume
	informer.add(pv)
	if err := verifyPV(cache, pvName, pv); err != nil {
		t.Fatalf("failed to get PV: %v", err)
	}

	// Assume PV
	newPV := pv.DeepCopy()
	newPV.Spec.ClaimRef = &v1.ObjectReference{Name: "test-claim"}
	if err := cache.Assume(newPV); err != nil {
		t.Fatalf("failed to assume PV: %v", err)
	}
	if err := verifyPV(cache, pvName, newPV); err != nil {
		t.Fatalf("failed to get PV after assume: %v", err)
	}

	// Add old PV (resync)
	informer.add(pv)
	if err := verifyPV(cache, pvName, newPV); err != nil {
		t.Fatalf("failed to get PV after old PV added: %v", err)
	}
}

func makeClaim(name, version, namespace string) *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       namespace,
			ResourceVersion: version,
			Annotations:     map[string]string{},
		},
	}
}

func verifyPVC(cache PVCAssumeCache, pvcKey string, expectedPVC *v1.PersistentVolumeClaim) error {
	pvc, err := cache.Get(pvcKey)
	if err != nil {
		return err
	}
	if pvc != expectedPVC {
		return fmt.Errorf("Get() returned %p, expected %p", pvc, expectedPVC)
	}
	return nil
}

func newTestPVCCache(t *testing.T) (*testInformer, PVCAssumeCache) {
	logger, _ := ktesting.NewTestContext(t)
	informer := &testInformer{
		indexer: cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{}),
		t:       t,
	}
	cache, err := NewPVCAssumeCache(logger, informer)
	if err != nil {
		t.Fatalf("NewPVCAssumeCache() failed: %v", err)
	}
	return informer, cache
}

func TestAssumePVC(t *testing.T) {
	scenarios := map[string]struct {
		oldPVC        *v1.PersistentVolumeClaim
		newPVC        *v1.PersistentVolumeClaim
		shouldSucceed bool
	}{
		"success-same-version": {
			oldPVC:        makeClaim("pvc1", "5", "ns1"),
			newPVC:        makeClaim("pvc1", "5", "ns1"),
			shouldSucceed: true,
		},
		"fail-new-higher-version": {
			oldPVC:        makeClaim("pvc1", "5", "ns1"),
			newPVC:        makeClaim("pvc1", "6", "ns1"),
			shouldSucceed: false,
		},
		"fail-old-not-found": {
			oldPVC:        makeClaim("pvc2", "5", "ns1"),
			newPVC:        makeClaim("pvc1", "5", "ns1"),
			shouldSucceed: false,
		},
		"fail-new-lower-version": {
			oldPVC:        makeClaim("pvc1", "5", "ns1"),
			newPVC:        makeClaim("pvc1", "4", "ns1"),
			shouldSucceed: false,
		},
		"fail-new-bad-version": {
			oldPVC:        makeClaim("pvc1", "5", "ns1"),
			newPVC:        makeClaim("pvc1", "a", "ns1"),
			shouldSucceed: false,
		},
		"fail-old-bad-version": {
			oldPVC:        makeClaim("pvc1", "a", "ns1"),
			newPVC:        makeClaim("pvc1", "5", "ns1"),
			shouldSucceed: false,
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			informer, cache := newTestPVCCache(t)

			// Add oldPVC to cache
			informer.add(scenario.oldPVC)
			if err := verifyPVC(cache, getPVCName(scenario.oldPVC), scenario.oldPVC); err != nil {
				t.Fatalf("Failed to Get() after initial update: %v", err)
			}

			// Assume newPVC
			err := cache.Assume(scenario.newPVC)
			if scenario.shouldSucceed && err != nil {
				t.Errorf("Test %q failed: Assume() returned error %v", name, err)
			}
			if !scenario.shouldSucceed && err == nil {
				t.Errorf("Test %q failed: Assume() returned success but expected error", name)
			}

			// Check that Get returns correct PVC
			expectedPV := scenario.newPVC
			if !scenario.shouldSucceed {
				expectedPV = scenario.oldPVC
			}
			if err := verifyPVC(cache, getPVCName(scenario.oldPVC), expectedPV); err != nil {
				t.Errorf("Failed to Get() after initial update: %v", err)
			}
		})
	}
}

func TestRestorePVC(t *testing.T) {
	informer, cache := newTestPVCCache(t)

	oldPVC := makeClaim("pvc1", "5", "ns1")
	newPVC := makeClaim("pvc1", "5", "ns1")

	// Restore PVC that doesn't exist
	cache.Restore(&v1.PersistentVolumeClaim{})

	// Add oldPVC to cache
	informer.add(oldPVC)
	if err := verifyPVC(cache, getPVCName(oldPVC), oldPVC); err != nil {
		t.Fatalf("Failed to Get() after initial update: %v", err)
	}

	// Restore PVC
	cache.Restore(oldPVC)
	if err := verifyPVC(cache, getPVCName(oldPVC), oldPVC); err != nil {
		t.Fatalf("Failed to Get() after initial restore: %v", err)
	}

	// Assume newPVC
	if err := cache.Assume(newPVC); err != nil {
		t.Fatalf("Assume() returned error %v", err)
	}
	if err := verifyPVC(cache, getPVCName(oldPVC), newPVC); err != nil {
		t.Fatalf("Failed to Get() after Assume: %v", err)
	}

	// Restore PVC
	cache.Restore(newPVC)
	if err := verifyPVC(cache, getPVCName(oldPVC), oldPVC); err != nil {
		t.Fatalf("Failed to Get() after restore: %v", err)
	}
}

func TestConcurrentAssumePVC(t *testing.T) {
	informer, cache := newTestPVCCache(t)

	pvc1 := makeClaim("pvc1", "5", "ns1")
	pvc1Update := makeClaim("pvc1", "5", "ns1")
	// Add PVC to cache
	informer.add(pvc1)

	// Update PVC 1
	if err := cache.Assume(pvc1Update); err != nil {
		t.Fatalf("Assume() returned error %v", err)
	}
	if err := verifyPVC(cache, getPVCName(pvc1Update), pvc1Update); err != nil {
		t.Fatalf("Failed to Get() after Assume: %v", err)
	}

	pvc2 := makeClaim("pvc1", "7", "ns1")
	pvc2Update := makeClaim("pvc1", "7", "ns1")
	// PVC updated externally
	informer.add(pvc2)

	// Update PVC 2
	if err := cache.Assume(pvc2Update); err != nil {
		t.Fatalf("Assume() returned error %v", err)
	}
	// PVC 1 failed with conflict
	cache.Restore(pvc1Update)
	// Should still have pvc 2 in cache
	if err := verifyPVC(cache, getPVCName(pvc2Update), pvc2Update); err != nil {
		t.Fatalf("Failed to Get() after restore: %v", err)
	}
}

func TestAssumeUpdatePVCCache(t *testing.T) {
	informer, cache := newTestPVCCache(t)

	pvcName := "test-pvc0"
	pvcNamespace := "test-ns"

	// Add a PVC
	pvc := makeClaim(pvcName, "1", pvcNamespace)
	informer.add(pvc)
	if err := verifyPVC(cache, getPVCName(pvc), pvc); err != nil {
		t.Fatalf("failed to get PVC: %v", err)
	}

	// Assume PVC
	newPVC := pvc.DeepCopy()
	newPVC.Annotations[volume.AnnSelectedNode] = "test-node"
	if err := cache.Assume(newPVC); err != nil {
		t.Fatalf("failed to assume PVC: %v", err)
	}
	if err := verifyPVC(cache, getPVCName(pvc), newPVC); err != nil {
		t.Fatalf("failed to get PVC after assume: %v", err)
	}

	// Add old PVC
	informer.add(pvc)
	if err := verifyPVC(cache, getPVCName(pvc), newPVC); err != nil {
		t.Fatalf("failed to get PVC after old PVC added: %v", err)
	}
}

func TestDelayedInformerEvent(t *testing.T) {
	informer, cache := newTestPVCCache(t)

	pvcName := "test-pvc0"
	pvcNamespace := "test-ns"

	pvc1 := makeClaim(pvcName, "1", pvcNamespace)
	pvc2 := makeClaim(pvcName, "2", pvcNamespace)
	// Only add indexer, simulating delayed informer event
	if err := informer.indexer.Add(pvc2); err != nil {
		t.Fatalf("failed to add PVC: %v", err)
	}

	newPVC := pvc2.DeepCopy()
	newPVC.Annotations[volume.AnnSelectedNode] = "test-node"
	if err := cache.Assume(newPVC); err != nil {
		t.Fatalf("failed to assume PVC: %v", err)
	}

	// Send the delayed event
	informer.handler.OnAdd(pvc1, false)
	informer.handler.OnDelete(pvc1)
	informer.handler.OnAdd(pvc2, false)
	// Expect assumed version not overwritten
	if err := verifyPVC(cache, getPVCName(newPVC), newPVC); err != nil {
		t.Fatalf("failed to get PVC after assume: %v", err)
	}
}
