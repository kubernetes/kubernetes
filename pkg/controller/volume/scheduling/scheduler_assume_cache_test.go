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

package scheduling

import (
	"fmt"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pvutil "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/util"
)

func makePV(name, version, storageClass string) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			ResourceVersion: version,
		},
		Spec: v1.PersistentVolumeSpec{
			StorageClassName: storageClass,
		},
	}
}

func verifyListPVs(t *testing.T, cache PVAssumeCache, expectedPVs map[string]*v1.PersistentVolume, storageClassName string) {
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

func verifyPV(cache PVAssumeCache, name string, expectedPV *v1.PersistentVolume) error {
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
	scenarios := map[string]struct {
		oldPV         *v1.PersistentVolume
		newPV         *v1.PersistentVolume
		shouldSucceed bool
	}{
		"success-same-version": {
			oldPV:         makePV("pv1", "5", ""),
			newPV:         makePV("pv1", "5", ""),
			shouldSucceed: true,
		},
		"success-storageclass-same-version": {
			oldPV:         makePV("pv1", "5", "class1"),
			newPV:         makePV("pv1", "5", "class1"),
			shouldSucceed: true,
		},
		"success-new-higher-version": {
			oldPV:         makePV("pv1", "5", ""),
			newPV:         makePV("pv1", "6", ""),
			shouldSucceed: true,
		},
		"fail-old-not-found": {
			oldPV:         makePV("pv2", "5", ""),
			newPV:         makePV("pv1", "5", ""),
			shouldSucceed: false,
		},
		"fail-new-lower-version": {
			oldPV:         makePV("pv1", "5", ""),
			newPV:         makePV("pv1", "4", ""),
			shouldSucceed: false,
		},
		"fail-new-bad-version": {
			oldPV:         makePV("pv1", "5", ""),
			newPV:         makePV("pv1", "a", ""),
			shouldSucceed: false,
		},
		"fail-old-bad-version": {
			oldPV:         makePV("pv1", "a", ""),
			newPV:         makePV("pv1", "5", ""),
			shouldSucceed: false,
		},
	}

	for name, scenario := range scenarios {
		cache := NewPVAssumeCache(nil)
		internalCache, ok := cache.(*pvAssumeCache).AssumeCache.(*assumeCache)
		if !ok {
			t.Fatalf("Failed to get internal cache")
		}

		// Add oldPV to cache
		internalCache.add(scenario.oldPV)
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
	cache := NewPVAssumeCache(nil)
	internalCache, ok := cache.(*pvAssumeCache).AssumeCache.(*assumeCache)
	if !ok {
		t.Fatalf("Failed to get internal cache")
	}

	oldPV := makePV("pv1", "5", "")
	newPV := makePV("pv1", "5", "")

	// Restore PV that doesn't exist
	cache.Restore("nothing")

	// Add oldPV to cache
	internalCache.add(oldPV)
	if err := verifyPV(cache, oldPV.Name, oldPV); err != nil {
		t.Fatalf("Failed to GetPV() after initial update: %v", err)
	}

	// Restore PV
	cache.Restore(oldPV.Name)
	if err := verifyPV(cache, oldPV.Name, oldPV); err != nil {
		t.Fatalf("Failed to GetPV() after iniital restore: %v", err)
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
	cache := NewPVAssumeCache(nil)
	internalCache, ok := cache.(*pvAssumeCache).AssumeCache.(*assumeCache)
	if !ok {
		t.Fatalf("Failed to get internal cache")
	}

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
		pv := makePV(fmt.Sprintf("test-pv%v", i), "1", "")
		pvs[pv.Name] = pv
		internalCache.add(pv)
	}

	// List them
	verifyListPVs(t, cache, pvs, "")

	// Update a PV
	updatedPV := makePV("test-pv3", "2", "")
	pvs[updatedPV.Name] = updatedPV
	internalCache.update(nil, updatedPV)

	// List them
	verifyListPVs(t, cache, pvs, "")

	// Delete a PV
	deletedPV := pvs["test-pv7"]
	delete(pvs, deletedPV.Name)
	internalCache.delete(deletedPV)

	// List them
	verifyListPVs(t, cache, pvs, "")
}

func TestPVCacheWithStorageClasses(t *testing.T) {
	cache := NewPVAssumeCache(nil)
	internalCache, ok := cache.(*pvAssumeCache).AssumeCache.(*assumeCache)
	if !ok {
		t.Fatalf("Failed to get internal cache")
	}

	// Add a bunch of PVs
	pvs1 := map[string]*v1.PersistentVolume{}
	for i := 0; i < 10; i++ {
		pv := makePV(fmt.Sprintf("test-pv%v", i), "1", "class1")
		pvs1[pv.Name] = pv
		internalCache.add(pv)
	}

	// Add a bunch of PVs
	pvs2 := map[string]*v1.PersistentVolume{}
	for i := 0; i < 10; i++ {
		pv := makePV(fmt.Sprintf("test2-pv%v", i), "1", "class2")
		pvs2[pv.Name] = pv
		internalCache.add(pv)
	}

	// List them
	verifyListPVs(t, cache, pvs1, "class1")
	verifyListPVs(t, cache, pvs2, "class2")

	// Update a PV
	updatedPV := makePV("test-pv3", "2", "class1")
	pvs1[updatedPV.Name] = updatedPV
	internalCache.update(nil, updatedPV)

	// List them
	verifyListPVs(t, cache, pvs1, "class1")
	verifyListPVs(t, cache, pvs2, "class2")

	// Delete a PV
	deletedPV := pvs1["test-pv7"]
	delete(pvs1, deletedPV.Name)
	internalCache.delete(deletedPV)

	// List them
	verifyListPVs(t, cache, pvs1, "class1")
	verifyListPVs(t, cache, pvs2, "class2")
}

func TestAssumeUpdatePVCache(t *testing.T) {
	cache := NewPVAssumeCache(nil)
	internalCache, ok := cache.(*pvAssumeCache).AssumeCache.(*assumeCache)
	if !ok {
		t.Fatalf("Failed to get internal cache")
	}

	pvName := "test-pv0"

	// Add a PV
	pv := makePV(pvName, "1", "")
	internalCache.add(pv)
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
	internalCache.add(pv)
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
		cache := NewPVCAssumeCache(nil)
		internalCache, ok := cache.(*pvcAssumeCache).AssumeCache.(*assumeCache)
		if !ok {
			t.Fatalf("Failed to get internal cache")
		}

		// Add oldPVC to cache
		internalCache.add(scenario.oldPVC)
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
	cache := NewPVCAssumeCache(nil)
	internalCache, ok := cache.(*pvcAssumeCache).AssumeCache.(*assumeCache)
	if !ok {
		t.Fatalf("Failed to get internal cache")
	}

	oldPVC := makeClaim("pvc1", "5", "ns1")
	newPVC := makeClaim("pvc1", "5", "ns1")

	// Restore PVC that doesn't exist
	cache.Restore("nothing")

	// Add oldPVC to cache
	internalCache.add(oldPVC)
	if err := verifyPVC(cache, getPVCName(oldPVC), oldPVC); err != nil {
		t.Fatalf("Failed to GetPVC() after initial update: %v", err)
	}

	// Restore PVC
	cache.Restore(getPVCName(oldPVC))
	if err := verifyPVC(cache, getPVCName(oldPVC), oldPVC); err != nil {
		t.Fatalf("Failed to GetPVC() after iniital restore: %v", err)
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
	cache := NewPVCAssumeCache(nil)
	internalCache, ok := cache.(*pvcAssumeCache).AssumeCache.(*assumeCache)
	if !ok {
		t.Fatalf("Failed to get internal cache")
	}

	pvcName := "test-pvc0"
	pvcNamespace := "test-ns"

	// Add a PVC
	pvc := makeClaim(pvcName, "1", pvcNamespace)
	internalCache.add(pvc)
	if err := verifyPVC(cache, getPVCName(pvc), pvc); err != nil {
		t.Fatalf("failed to get PVC: %v", err)
	}

	// Assume PVC
	newPVC := pvc.DeepCopy()
	newPVC.Annotations[pvutil.AnnSelectedNode] = "test-node"
	if err := cache.Assume(newPVC); err != nil {
		t.Fatalf("failed to assume PVC: %v", err)
	}
	if err := verifyPVC(cache, getPVCName(pvc), newPVC); err != nil {
		t.Fatalf("failed to get PVC after assume: %v", err)
	}

	// Add old PVC
	internalCache.add(pvc)
	if err := verifyPVC(cache, getPVCName(pvc), newPVC); err != nil {
		t.Fatalf("failed to get PVC after old PVC added: %v", err)
	}
}
