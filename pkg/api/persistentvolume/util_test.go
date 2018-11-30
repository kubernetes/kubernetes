/*
Copyright 2018 The Kubernetes Authors.

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

package persistentvolume

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

func TestDropDisabledFields(t *testing.T) {
	specWithCSI := func() *api.PersistentVolumeSpec {
		return &api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{CSI: &api.CSIPersistentVolumeSource{}}}
	}
	specWithoutCSI := func() *api.PersistentVolumeSpec {
		return &api.PersistentVolumeSpec{PersistentVolumeSource: api.PersistentVolumeSource{CSI: nil}}
	}
	specWithMode := func(mode *api.PersistentVolumeMode) *api.PersistentVolumeSpec {
		return &api.PersistentVolumeSpec{VolumeMode: mode}
	}

	modeBlock := api.PersistentVolumeBlock

	tests := map[string]struct {
		oldSpec       *api.PersistentVolumeSpec
		newSpec       *api.PersistentVolumeSpec
		expectOldSpec *api.PersistentVolumeSpec
		expectNewSpec *api.PersistentVolumeSpec
		csiEnabled    bool
		blockEnabled  bool
	}{
		"disabled csi clears new": {
			csiEnabled:    false,
			newSpec:       specWithCSI(),
			expectNewSpec: specWithoutCSI(),
			oldSpec:       nil,
			expectOldSpec: nil,
		},
		"disabled csi clears update when old pv did not use csi": {
			csiEnabled:    false,
			newSpec:       specWithCSI(),
			expectNewSpec: specWithoutCSI(),
			oldSpec:       specWithoutCSI(),
			expectOldSpec: specWithoutCSI(),
		},
		"disabled csi preserves update when old pv did use csi": {
			csiEnabled:    false,
			newSpec:       specWithCSI(),
			expectNewSpec: specWithCSI(),
			oldSpec:       specWithCSI(),
			expectOldSpec: specWithCSI(),
		},

		"enabled csi preserves new": {
			csiEnabled:    true,
			newSpec:       specWithCSI(),
			expectNewSpec: specWithCSI(),
			oldSpec:       nil,
			expectOldSpec: nil,
		},
		"enabled csi preserves update when old pv did not use csi": {
			csiEnabled:    true,
			newSpec:       specWithCSI(),
			expectNewSpec: specWithCSI(),
			oldSpec:       specWithoutCSI(),
			expectOldSpec: specWithoutCSI(),
		},
		"enabled csi preserves update when old pv did use csi": {
			csiEnabled:    true,
			newSpec:       specWithCSI(),
			expectNewSpec: specWithCSI(),
			oldSpec:       specWithCSI(),
			expectOldSpec: specWithCSI(),
		},

		"disabled block clears new": {
			blockEnabled:  false,
			newSpec:       specWithMode(&modeBlock),
			expectNewSpec: specWithMode(nil),
			oldSpec:       nil,
			expectOldSpec: nil,
		},
		"disabled block clears update when old pv did not use block": {
			blockEnabled:  false,
			newSpec:       specWithMode(&modeBlock),
			expectNewSpec: specWithMode(nil),
			oldSpec:       specWithMode(nil),
			expectOldSpec: specWithMode(nil),
		},
		// TODO: consider changing this case to preserve
		"disabled block clears old and new on update when old pv did use block": {
			blockEnabled:  false,
			newSpec:       specWithMode(&modeBlock),
			expectNewSpec: specWithMode(nil),
			oldSpec:       specWithMode(&modeBlock),
			expectOldSpec: specWithMode(nil),
		},

		"enabled block preserves new": {
			blockEnabled:  true,
			newSpec:       specWithMode(&modeBlock),
			expectNewSpec: specWithMode(&modeBlock),
			oldSpec:       nil,
			expectOldSpec: nil,
		},
		"enabled block preserves update when old pv did not use block": {
			blockEnabled:  true,
			newSpec:       specWithMode(&modeBlock),
			expectNewSpec: specWithMode(&modeBlock),
			oldSpec:       specWithMode(nil),
			expectOldSpec: specWithMode(nil),
		},
		"enabled block preserves update when old pv did use block": {
			blockEnabled:  true,
			newSpec:       specWithMode(&modeBlock),
			expectNewSpec: specWithMode(&modeBlock),
			oldSpec:       specWithMode(&modeBlock),
			expectOldSpec: specWithMode(&modeBlock),
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIPersistentVolume, tc.csiEnabled)()
			defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BlockVolume, tc.blockEnabled)()

			DropDisabledFields(tc.newSpec, tc.oldSpec)
			if !reflect.DeepEqual(tc.newSpec, tc.expectNewSpec) {
				t.Error(diff.ObjectReflectDiff(tc.newSpec, tc.expectNewSpec))
			}
			if !reflect.DeepEqual(tc.oldSpec, tc.expectOldSpec) {
				t.Error(diff.ObjectReflectDiff(tc.oldSpec, tc.expectOldSpec))
			}
		})
	}
}
