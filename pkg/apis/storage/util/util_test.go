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

package util

import (
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/storage"
)

func TestDropAlphaFields(t *testing.T) {
	bindingMode := storage.VolumeBindingWaitForFirstConsumer

	// Test that field gets dropped when feature gate is not set
	if err := utilfeature.DefaultFeatureGate.Set("VolumeScheduling=false"); err != nil {
		t.Fatalf("Failed to set feature gate for VolumeScheduling: %v", err)
	}
	class := &storage.StorageClass{
		VolumeBindingMode: &bindingMode,
	}
	DropDisabledAlphaFields(class)
	if class.VolumeBindingMode != nil {
		t.Errorf("VolumeBindingMode field didn't get dropped: %+v", class.VolumeBindingMode)
	}

	// Test that field does not get dropped when feature gate is set
	class = &storage.StorageClass{
		VolumeBindingMode: &bindingMode,
	}
	if err := utilfeature.DefaultFeatureGate.Set("VolumeScheduling=true"); err != nil {
		t.Fatalf("Failed to set feature gate for VolumeScheduling: %v", err)
	}
	DropDisabledAlphaFields(class)
	if class.VolumeBindingMode != &bindingMode {
		t.Errorf("VolumeBindingMode field got unexpectantly modified: %+v", class.VolumeBindingMode)
	}
	if err := utilfeature.DefaultFeatureGate.Set("VolumeScheduling=false"); err != nil {
		t.Fatalf("Failed to disable feature gate for VolumeScheduling: %v", err)
	}
}
