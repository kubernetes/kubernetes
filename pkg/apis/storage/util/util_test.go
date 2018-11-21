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
	"reflect"
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/features"
)

func TestDropAlphaFields(t *testing.T) {
	bindingMode := storage.VolumeBindingWaitForFirstConsumer
	allowedTopologies := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node1"},
				},
			},
		},
	}

	// Test that field gets dropped when feature gate is not set
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeScheduling, false)()
	class := &storage.StorageClass{
		VolumeBindingMode: &bindingMode,
		AllowedTopologies: allowedTopologies,
	}
	DropDisabledAlphaFields(class)
	if class.VolumeBindingMode != nil {
		t.Errorf("VolumeBindingMode field didn't get dropped: %+v", class.VolumeBindingMode)
	}
	if class.AllowedTopologies != nil {
		t.Errorf("AllowedTopologies field didn't get dropped: %+v", class.AllowedTopologies)
	}

	// Test that field does not get dropped when feature gate is set
	class = &storage.StorageClass{
		VolumeBindingMode: &bindingMode,
		AllowedTopologies: allowedTopologies,
	}
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeScheduling, true)()
	DropDisabledAlphaFields(class)
	if class.VolumeBindingMode != &bindingMode {
		t.Errorf("VolumeBindingMode field got unexpectantly modified: %+v", class.VolumeBindingMode)
	}
	if !reflect.DeepEqual(class.AllowedTopologies, allowedTopologies) {
		t.Errorf("AllowedTopologies field got unexpectantly modified: %+v", class.AllowedTopologies)
	}
}
