/*
Copyright 2016 The Kubernetes Authors.

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

package validation

import (
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/storage"
)

var (
	deleteReclaimPolicy = api.PersistentVolumeReclaimDelete
	immediateMode1      = storage.VolumeBindingImmediate
	immediateMode2      = storage.VolumeBindingImmediate
	waitingMode         = storage.VolumeBindingWaitForFirstConsumer
	invalidMode         = storage.VolumeBindingMode("foo")
)

func TestValidateStorageClass(t *testing.T) {
	deleteReclaimPolicy := api.PersistentVolumeReclaimPolicy("Delete")
	retainReclaimPolicy := api.PersistentVolumeReclaimPolicy("Retain")
	recycleReclaimPolicy := api.PersistentVolumeReclaimPolicy("Recycle")
	successCases := []storage.StorageClass{
		{
			// empty parameters
			ObjectMeta:    metav1.ObjectMeta{Name: "foo"},
			Provisioner:   "kubernetes.io/foo-provisioner",
			Parameters:    map[string]string{},
			ReclaimPolicy: &deleteReclaimPolicy,
		},
		{
			// nil parameters
			ObjectMeta:    metav1.ObjectMeta{Name: "foo"},
			Provisioner:   "kubernetes.io/foo-provisioner",
			ReclaimPolicy: &deleteReclaimPolicy,
		},
		{
			// some parameters
			ObjectMeta:  metav1.ObjectMeta{Name: "foo"},
			Provisioner: "kubernetes.io/foo-provisioner",
			Parameters: map[string]string{
				"kubernetes.io/foo-parameter": "free/form/string",
				"foo-parameter":               "free-form-string",
				"foo-parameter2":              "{\"embedded\": \"json\", \"with\": {\"structures\":\"inside\"}}",
			},
			ReclaimPolicy: &deleteReclaimPolicy,
		},
		{
			// retain reclaimPolicy
			ObjectMeta:    metav1.ObjectMeta{Name: "foo"},
			Provisioner:   "kubernetes.io/foo-provisioner",
			ReclaimPolicy: &retainReclaimPolicy,
		},
	}

	// Success cases are expected to pass validation.
	for k, v := range successCases {
		if errs := ValidateStorageClass(&v); len(errs) != 0 {
			t.Errorf("Expected success for %d, got %v", k, errs)
		}
	}

	// generate a map longer than maxProvisionerParameterSize
	longParameters := make(map[string]string)
	totalSize := 0
	for totalSize < maxProvisionerParameterSize {
		k := fmt.Sprintf("param/%d", totalSize)
		v := fmt.Sprintf("value-%d", totalSize)
		longParameters[k] = v
		totalSize = totalSize + len(k) + len(v)
	}

	errorCases := map[string]storage.StorageClass{
		"namespace is present": {
			ObjectMeta:    metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Provisioner:   "kubernetes.io/foo-provisioner",
			ReclaimPolicy: &deleteReclaimPolicy,
		},
		"invalid provisioner": {
			ObjectMeta:    metav1.ObjectMeta{Name: "foo"},
			Provisioner:   "kubernetes.io/invalid/provisioner",
			ReclaimPolicy: &deleteReclaimPolicy,
		},
		"invalid empty parameter name": {
			ObjectMeta:  metav1.ObjectMeta{Name: "foo"},
			Provisioner: "kubernetes.io/foo",
			Parameters: map[string]string{
				"": "value",
			},
			ReclaimPolicy: &deleteReclaimPolicy,
		},
		"provisioner: Required value": {
			ObjectMeta:    metav1.ObjectMeta{Name: "foo"},
			Provisioner:   "",
			ReclaimPolicy: &deleteReclaimPolicy,
		},
		"too long parameters": {
			ObjectMeta:    metav1.ObjectMeta{Name: "foo"},
			Provisioner:   "kubernetes.io/foo",
			Parameters:    longParameters,
			ReclaimPolicy: &deleteReclaimPolicy,
		},
		"invalid reclaimpolicy": {
			ObjectMeta:    metav1.ObjectMeta{Name: "foo"},
			Provisioner:   "kubernetes.io/foo",
			ReclaimPolicy: &recycleReclaimPolicy,
		},
	}

	// Error cases are not expected to pass validation.
	for testName, storageClass := range errorCases {
		if errs := ValidateStorageClass(&storageClass); len(errs) == 0 {
			t.Errorf("Expected failure for test: %s", testName)
		}
	}
}

func TestAlphaExpandPersistentVolumesFeatureValidation(t *testing.T) {
	deleteReclaimPolicy := api.PersistentVolumeReclaimPolicy("Delete")
	falseVar := false
	testSC := &storage.StorageClass{
		// empty parameters
		ObjectMeta:           metav1.ObjectMeta{Name: "foo"},
		Provisioner:          "kubernetes.io/foo-provisioner",
		Parameters:           map[string]string{},
		ReclaimPolicy:        &deleteReclaimPolicy,
		AllowVolumeExpansion: &falseVar,
	}

	// Enable alpha feature ExpandPersistentVolumes
	err := utilfeature.DefaultFeatureGate.Set("ExpandPersistentVolumes=true")
	if err != nil {
		t.Errorf("Failed to enable feature gate for ExpandPersistentVolumes: %v", err)
		return
	}
	if errs := ValidateStorageClass(testSC); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
	// Disable alpha feature ExpandPersistentVolumes
	err = utilfeature.DefaultFeatureGate.Set("ExpandPersistentVolumes=false")
	if err != nil {
		t.Errorf("Failed to disable feature gate for ExpandPersistentVolumes: %v", err)
		return
	}
	if errs := ValidateStorageClass(testSC); len(errs) == 0 {
		t.Errorf("expected failure, but got no error")
	}

}

func makeClassWithBinding(mode *storage.VolumeBindingMode) *storage.StorageClass {
	return &storage.StorageClass{
		ObjectMeta:        metav1.ObjectMeta{Name: "foo", ResourceVersion: "foo"},
		Provisioner:       "kubernetes.io/foo-provisioner",
		ReclaimPolicy:     &deleteReclaimPolicy,
		VolumeBindingMode: mode,
	}
}

// TODO: Remove these tests once feature gate is not required
func TestValidateVolumeBindingModeAlphaDisabled(t *testing.T) {
	errorCases := map[string]*storage.StorageClass{
		"immediate mode": makeClassWithBinding(&immediateMode1),
		"waiting mode":   makeClassWithBinding(&waitingMode),
		"invalid mode":   makeClassWithBinding(&invalidMode),
	}

	for testName, storageClass := range errorCases {
		if errs := ValidateStorageClass(storageClass); len(errs) == 0 {
			t.Errorf("Expected failure for test: %v", testName)
		}
	}
}

type bindingTest struct {
	class         *storage.StorageClass
	shouldSucceed bool
}

func TestValidateVolumeBindingMode(t *testing.T) {
	cases := map[string]bindingTest{
		"no mode": {
			class:         makeClassWithBinding(nil),
			shouldSucceed: true,
		},
		"immediate mode": {
			class:         makeClassWithBinding(&immediateMode1),
			shouldSucceed: true,
		},
		"waiting mode": {
			class:         makeClassWithBinding(&waitingMode),
			shouldSucceed: true,
		},
		"invalid mode": {
			class:         makeClassWithBinding(&invalidMode),
			shouldSucceed: false,
		},
	}

	// TODO: remove when feature gate not required
	err := utilfeature.DefaultFeatureGate.Set("VolumeTopologyBinding=true")
	if err != nil {
		t.Fatalf("Failed to enable feature gate for VolumeTopologyBinding: %v", err)
	}

	for testName, testCase := range cases {
		errs := ValidateStorageClass(testCase.class)
		if testCase.shouldSucceed && len(errs) != 0 {
			t.Errorf("Expected success for test %q, got %v", testName, errs)
		}
		if !testCase.shouldSucceed && len(errs) == 0 {
			t.Errorf("Expected failure for test %q, got success", testName)
		}
	}

	err = utilfeature.DefaultFeatureGate.Set("VolumeTopologyBinding=false")
	if err != nil {
		t.Fatalf("Failed to disable feature gate for VolumeTopologyBinding: %v", err)
	}
}

type updateTest struct {
	oldClass      *storage.StorageClass
	newClass      *storage.StorageClass
	shouldSucceed bool
}

func TestValidateUpdateVolumeBindingMode(t *testing.T) {
	noBinding := makeClassWithBinding(nil)
	immediateBinding1 := makeClassWithBinding(&immediateMode1)
	immediateBinding2 := makeClassWithBinding(&immediateMode2)
	waitBinding := makeClassWithBinding(&waitingMode)

	cases := map[string]updateTest{
		"old and new no mode": {
			oldClass:      noBinding,
			newClass:      noBinding,
			shouldSucceed: true,
		},
		"old and new same mode ptr": {
			oldClass:      immediateBinding1,
			newClass:      immediateBinding1,
			shouldSucceed: true,
		},
		"old and new same mode value": {
			oldClass:      immediateBinding1,
			newClass:      immediateBinding2,
			shouldSucceed: true,
		},
		"old no mode, new mode": {
			oldClass:      noBinding,
			newClass:      waitBinding,
			shouldSucceed: false,
		},
		"old mode, new no mode": {
			oldClass:      waitBinding,
			newClass:      noBinding,
			shouldSucceed: false,
		},
		"old and new different modes": {
			oldClass:      waitBinding,
			newClass:      immediateBinding1,
			shouldSucceed: false,
		},
	}

	// TODO: remove when feature gate not required
	err := utilfeature.DefaultFeatureGate.Set("VolumeTopologyBinding=true")
	if err != nil {
		t.Fatalf("Failed to enable feature gate for VolumeTopologyBinding: %v", err)
	}

	for testName, testCase := range cases {
		errs := ValidateStorageClassUpdate(testCase.newClass, testCase.oldClass)
		if testCase.shouldSucceed && len(errs) != 0 {
			t.Errorf("Expected success for %v, got %v", testName, errs)
		}
		if !testCase.shouldSucceed && len(errs) == 0 {
			t.Errorf("Expected failure for %v, got success", testName)
		}
	}

	err = utilfeature.DefaultFeatureGate.Set("VolumeTopologyBinding=false")
	if err != nil {
		t.Fatalf("Failed to disable feature gate for VolumeTopologyBinding: %v", err)
	}
}
