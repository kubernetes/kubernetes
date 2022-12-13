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

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

func TestDropDisabledFields(t *testing.T) {
	secretRef := &api.SecretReference{
		Name:      "expansion-secret",
		Namespace: "default",
	}

	tests := map[string]struct {
		oldSpec             *api.PersistentVolumeSpec
		newSpec             *api.PersistentVolumeSpec
		expectOldSpec       *api.PersistentVolumeSpec
		expectNewSpec       *api.PersistentVolumeSpec
		csiExpansionEnabled bool
	}{
		"disabled csi expansion clears secrets": {
			csiExpansionEnabled: false,
			newSpec:             specWithCSISecrets(secretRef),
			expectNewSpec:       specWithCSISecrets(nil),
			oldSpec:             nil,
			expectOldSpec:       nil,
		},
		"enabled csi expansion preserve secrets": {
			csiExpansionEnabled: true,
			newSpec:             specWithCSISecrets(secretRef),
			expectNewSpec:       specWithCSISecrets(secretRef),
			oldSpec:             nil,
			expectOldSpec:       nil,
		},
		"enabled csi expansion preserve secrets when both old and new have it": {
			csiExpansionEnabled: true,
			newSpec:             specWithCSISecrets(secretRef),
			expectNewSpec:       specWithCSISecrets(secretRef),
			oldSpec:             specWithCSISecrets(secretRef),
			expectOldSpec:       specWithCSISecrets(secretRef),
		},
		"disabled csi expansion old pv had secrets": {
			csiExpansionEnabled: false,
			newSpec:             specWithCSISecrets(secretRef),
			expectNewSpec:       specWithCSISecrets(secretRef),
			oldSpec:             specWithCSISecrets(secretRef),
			expectOldSpec:       specWithCSISecrets(secretRef),
		},
		"enabled csi expansion preserves secrets when old pv did not had secrets": {
			csiExpansionEnabled: true,
			newSpec:             specWithCSISecrets(secretRef),
			expectNewSpec:       specWithCSISecrets(secretRef),
			oldSpec:             specWithCSISecrets(nil),
			expectOldSpec:       specWithCSISecrets(nil),
		},
		"disabled csi expansion neither new pv nor old pv had secrets": {
			csiExpansionEnabled: false,
			newSpec:             specWithCSISecrets(nil),
			expectNewSpec:       specWithCSISecrets(nil),
			oldSpec:             specWithCSISecrets(nil),
			expectOldSpec:       specWithCSISecrets(nil),
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSINodeExpandSecret, tc.csiExpansionEnabled)()

			DropDisabledFields(tc.newSpec, tc.oldSpec)
			if !reflect.DeepEqual(tc.newSpec, tc.expectNewSpec) {
				t.Error(cmp.Diff(tc.newSpec, tc.expectNewSpec))
			}
			if !reflect.DeepEqual(tc.oldSpec, tc.expectOldSpec) {
				t.Error(cmp.Diff(tc.oldSpec, tc.expectOldSpec))
			}
		})
	}
}

func specWithCSISecrets(secret *api.SecretReference) *api.PersistentVolumeSpec {
	pvSpec := &api.PersistentVolumeSpec{
		PersistentVolumeSource: api.PersistentVolumeSource{
			CSI: &api.CSIPersistentVolumeSource{
				Driver:       "com.google.gcepd",
				VolumeHandle: "foobar",
			},
		},
	}

	if secret != nil {
		pvSpec.CSI.NodeExpandSecretRef = secret
	}
	return pvSpec
}

func TestWarnings(t *testing.T) {
	testcases := []struct {
		name     string
		template *api.PersistentVolume
		expected []string
	}{
		{
			name:     "null",
			template: nil,
			expected: nil,
		},
		{
			name: "no warning",
			template: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Status: api.PersistentVolumeStatus{
					Phase: api.VolumeBound,
				},
			},
			expected: nil,
		},
		{
			name: "warning",
			template: &api.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: api.PersistentVolumeSpec{
					NodeAffinity: &api.VolumeNodeAffinity{
						Required: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{
								{
									MatchExpressions: []api.NodeSelectorRequirement{
										{
											Key:      "beta.kubernetes.io/os",
											Operator: "Equal",
											Values:   []string{"windows"},
										},
									},
								},
							},
						},
					},
				},
				Status: api.PersistentVolumeStatus{
					Phase: api.VolumeBound,
				},
			},
			expected: []string{
				`spec.nodeAffinity.required.nodeSelectorTerms[0].matchExpressions[0].key: beta.kubernetes.io/os is deprecated since v1.14; use "kubernetes.io/os" instead`,
			},
		},
	}

	for _, tc := range testcases {
		t.Run("podspec_"+tc.name, func(t *testing.T) {
			actual := sets.NewString(GetWarningsForPersistentVolume(tc.template)...)
			expected := sets.NewString(tc.expected...)
			for _, missing := range expected.Difference(actual).List() {
				t.Errorf("missing: %s", missing)
			}
			for _, extra := range actual.Difference(expected).List() {
				t.Errorf("extra: %s", extra)
			}
		})

	}
}
