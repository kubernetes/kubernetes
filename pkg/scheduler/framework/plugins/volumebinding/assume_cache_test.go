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
	"k8s.io/component-helpers/storage/volume"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

func verifyListPVs(t *testing.T, cache *PVAssumeCache, expectedPVs map[string]*v1.PersistentVolume, storageClassName string) {
	pvList := cache.ListPVs(storageClassName)
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

func verifyPV(cache *PVAssumeCache, name string, expectedPV *v1.PersistentVolume) error {
	pv, err := cache.GetPV(name)
	if err != nil {
		return err
	}
	if pv != expectedPV {
		return fmt.Errorf("GetPV() returned %p, expected %p", pv, expectedPV)
	}
	return nil
}

func TestAssumePV(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
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
		"success-new-higher-version": {
			oldPV:         makePV("pv1", "").withVersion("5").PersistentVolume,
			newPV:         makePV("pv1", "").withVersion("6").PersistentVolume,
			shouldSucceed: true,
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
		cache := NewPVAssumeCache(logger, nil)

		// Add oldPV to cache
		assumecache.AddTestObject(cache.AssumeCache, scenario.oldPV)
		if err := verifyPV(cache, scenario.oldPV.Name, scenario.oldPV); err != nil {
			t.Errorf("Failed to GetPV() after initial update: %v", err)
			continue
		}

		// Assume newPV
		err := cache.Assume(scenario.newPV)
		if scenario.shouldSucceed && err != nil {
			t.Errorf("Test %q failed: Assume() returned error %v", name, err)
		}
		if !scenario.shouldSucceed && err == nil {
			t.Errorf("Test %q failed: Assume() returned success but expected error", name)
		}

		// Check that GetPV returns correct PV
		expectedPV := scenario.newPV
		if !scenario.shouldSucceed {
			expectedPV = scenario.oldPV
		}
		if err := verifyPV(cache, scenario.oldPV.Name, expectedPV); err != nil {
			t.Errorf("Failed to GetPV() after initial update: %v", err)
		}
	}
}

func TestRestorePV(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cache := NewPVAssumeCache(logger, nil)

	oldPV := makePV("pv1", "").withVersion("5").PersistentVolume
	newPV := makePV("pv1", "").withVersion("5").PersistentVolume

	// Restore PV that doesn't exist
	cache.Restore("nothing")

	// Add oldPV to cache
	assumecache.AddTestObject(cache.AssumeCache, oldPV)
	if err := verifyPV(cache, oldPV.Name, oldPV); err != nil {
		t.Fatalf("Failed to GetPV() after initial update: %v", err)
	}

	// Restore PV
	cache.Restore(oldPV.Name)
	if err := verifyPV(cache, oldPV.Name, oldPV); err != nil {
		t.Fatalf("Failed to GetPV() after initial restore: %v", err)
	}

	// Assume newPV
	if err := cache.Assume(newPV); err != nil {
		t.Fatalf("Assume() returned error %v", err)
	}
	if err := verifyPV(cache, oldPV.Name, newPV); err != nil {
		t.Fatalf("Failed to GetPV() after Assume: %v", err)
	}

	// Restore PV
	cache.Restore(oldPV.Name)
	if err := verifyPV(cache, oldPV.Name, oldPV); err != nil {
		t.Fatalf("Failed to GetPV() after restore: %v", err)
	}
}

func TestBasicPVCache(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cache := NewPVAssumeCache(logger, nil)

	// Get object that doesn't exist
	pv, err := cache.GetPV("nothere")
	if err == nil {
		t.Errorf("GetPV() returned unexpected success")
	}
	if pv != nil {
		t.Errorf("GetPV() returned unexpected PV %q", pv.Name)
	}

	// Add a bunch of PVs
	pvs := map[string]*v1.PersistentVolume{}
	for i := 0; i < 10; i++ {
		pv := makePV(fmt.Sprintf("test-pv%v", i), "").withVersion("1").PersistentVolume
		pvs[pv.Name] = pv
		assumecache.AddTestObject(cache.AssumeCache, pv)
	}

	// List them
	verifyListPVs(t, cache, pvs, "")

	// Update a PV
	updatedPV := makePV("test-pv3", "").withVersion("2").PersistentVolume
	pvs[updatedPV.Name] = updatedPV
	assumecache.UpdateTestObject(cache.AssumeCache, updatedPV)

	// List them
	verifyListPVs(t, cache, pvs, "")

	// Delete a PV
	deletedPV := pvs["test-pv7"]
	delete(pvs, deletedPV.Name)
	assumecache.DeleteTestObject(cache.AssumeCache, deletedPV)

	// List them
	verifyListPVs(t, cache, pvs, "")
}

func TestPVCacheWithStorageClasses(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cache := NewPVAssumeCache(logger, nil)

	// Add a bunch of PVs
	pvs1 := map[string]*v1.PersistentVolume{}
	for i := 0; i < 10; i++ {
		pv := makePV(fmt.Sprintf("test-pv%v", i), "class1").withVersion("1").PersistentVolume
		pvs1[pv.Name] = pv
		assumecache.AddTestObject(cache.AssumeCache, pv)
	}

	// Add a bunch of PVs
	pvs2 := map[string]*v1.PersistentVolume{}
	for i := 0; i < 10; i++ {
		pv := makePV(fmt.Sprintf("test2-pv%v", i), "class2").withVersion("1").PersistentVolume
		pvs2[pv.Name] = pv
		assumecache.AddTestObject(cache.AssumeCache, pv)
	}

	// List them
	verifyListPVs(t, cache, pvs1, "class1")
	verifyListPVs(t, cache, pvs2, "class2")

	// Update a PV
	updatedPV := makePV("test-pv3", "class1").withVersion("2").PersistentVolume
	pvs1[updatedPV.Name] = updatedPV
	assumecache.UpdateTestObject(cache.AssumeCache, updatedPV)

	// List them
	verifyListPVs(t, cache, pvs1, "class1")
	verifyListPVs(t, cache, pvs2, "class2")

	// Delete a PV
	deletedPV := pvs1["test-pv7"]
	delete(pvs1, deletedPV.Name)
	assumecache.DeleteTestObject(cache.AssumeCache, deletedPV)

	// List them
	verifyListPVs(t, cache, pvs1, "class1")
	verifyListPVs(t, cache, pvs2, "class2")
}

func TestAssumeUpdatePVCache(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cache := NewPVAssumeCache(logger, nil)

	pvName := "test-pv0"

	// Add a PV
	pv := makePV(pvName, "").withVersion("1").PersistentVolume
	assumecache.AddTestObject(cache.AssumeCache, pv)
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

	// Add old PV
	assumecache.AddTestObject(cache.AssumeCache, pv)
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

func verifyPVC(cache *PVCAssumeCache, pvcKey string, expectedPVC *v1.PersistentVolumeClaim) error {
	pvc, err := cache.GetPVC(pvcKey)
	if err != nil {
		return err
	}
	if pvc != expectedPVC {
		return fmt.Errorf("GetPVC() returned %p, expected %p", pvc, expectedPVC)
	}
	return nil
}

func TestAssumePVC(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
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
		"success-new-higher-version": {
			oldPVC:        makeClaim("pvc1", "5", "ns1"),
			newPVC:        makeClaim("pvc1", "6", "ns1"),
			shouldSucceed: true,
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
		cache := NewPVCAssumeCache(logger, nil)

		// Add oldPVC to cache
		assumecache.AddTestObject(cache.AssumeCache, scenario.oldPVC)
		if err := verifyPVC(cache, getPVCName(scenario.oldPVC), scenario.oldPVC); err != nil {
			t.Errorf("Failed to GetPVC() after initial update: %v", err)
			continue
		}

		// Assume newPVC
		err := cache.Assume(scenario.newPVC)
		if scenario.shouldSucceed && err != nil {
			t.Errorf("Test %q failed: Assume() returned error %v", name, err)
		}
		if !scenario.shouldSucceed && err == nil {
			t.Errorf("Test %q failed: Assume() returned success but expected error", name)
		}

		// Check that GetPVC returns correct PVC
		expectedPV := scenario.newPVC
		if !scenario.shouldSucceed {
			expectedPV = scenario.oldPVC
		}
		if err := verifyPVC(cache, getPVCName(scenario.oldPVC), expectedPV); err != nil {
			t.Errorf("Failed to GetPVC() after initial update: %v", err)
		}
	}
}

func TestRestorePVC(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cache := NewPVCAssumeCache(logger, nil)

	oldPVC := makeClaim("pvc1", "5", "ns1")
	newPVC := makeClaim("pvc1", "5", "ns1")

	// Restore PVC that doesn't exist
	cache.Restore("nothing")

	// Add oldPVC to cache
	assumecache.AddTestObject(cache.AssumeCache, oldPVC)
	if err := verifyPVC(cache, getPVCName(oldPVC), oldPVC); err != nil {
		t.Fatalf("Failed to GetPVC() after initial update: %v", err)
	}

	// Restore PVC
	cache.Restore(getPVCName(oldPVC))
	if err := verifyPVC(cache, getPVCName(oldPVC), oldPVC); err != nil {
		t.Fatalf("Failed to GetPVC() after initial restore: %v", err)
	}

	// Assume newPVC
	if err := cache.Assume(newPVC); err != nil {
		t.Fatalf("Assume() returned error %v", err)
	}
	if err := verifyPVC(cache, getPVCName(oldPVC), newPVC); err != nil {
		t.Fatalf("Failed to GetPVC() after Assume: %v", err)
	}

	// Restore PVC
	cache.Restore(getPVCName(oldPVC))
	if err := verifyPVC(cache, getPVCName(oldPVC), oldPVC); err != nil {
		t.Fatalf("Failed to GetPVC() after restore: %v", err)
	}
}

func TestAssumeUpdatePVCCache(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cache := NewPVCAssumeCache(logger, nil)

	pvcName := "test-pvc0"
	pvcNamespace := "test-ns"

	// Add a PVC
	pvc := makeClaim(pvcName, "1", pvcNamespace)
	assumecache.AddTestObject(cache.AssumeCache, pvc)
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
	assumecache.AddTestObject(cache.AssumeCache, pvc)
	if err := verifyPVC(cache, getPVCName(pvc), newPVC); err != nil {
		t.Fatalf("failed to get PVC after old PVC added: %v", err)
	}
}
