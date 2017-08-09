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

package persistentvolumeclaim

import (
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/features"
)

func TestDropAlphaPVCVolumeMode(t *testing.T) {
	vmode := api.PersistentVolumeFilesystem

	// PersistentVolume with VolumeMode set
	pvc := api.PersistentVolumeClaim{
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			VolumeMode:  &vmode,
		},
	}

	// Enable alpha feature BlockVolumeSupport
	err1 := utilfeature.DefaultFeatureGate.Set("BlockVolumeSupport=true")
	if err1 != nil {
		t.Fatalf("Failed to enable feature gate for BlockVolumeSupport: %v", err1)
	}

	// now test dropping the fields - should not be dropped
	DropDisabledAlphaFields(&pvc.Spec)

	// check to make sure VolumeDevices is still present
	// if featureset is set to true
	if utilfeature.DefaultFeatureGate.Enabled(features.BlockVolumeSupport) {
		if pvc.Spec.VolumeMode == nil {
			t.Error("VolumeMode in pvc.Spec should not have been dropped based on feature-gate")
		}
	}

	// Disable alpha feature BlockVolumeSupport
	err := utilfeature.DefaultFeatureGate.Set("BlockVolumeSupport=false")
	if err != nil {
		t.Fatalf("Failed to disable feature gate for BlockVolumeSupport: %v", err)
	}

	// now test dropping the fields
	DropDisabledAlphaFields(&pvc.Spec)

	// check to make sure VolumeDevices is nil
	// if featureset is set to false
	if !utilfeature.DefaultFeatureGate.Enabled(features.BlockVolumeSupport) {
		if pvc.Spec.VolumeMode != nil {
			t.Error("DropDisabledAlphaFields VolumeMode for pvc.Spec failed")
		}
	}
}
