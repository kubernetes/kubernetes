/*
Copyright The Kubernetes Authors.

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

package podgroupprotection

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	schedulingapi "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
)

func TestAdmit(t *testing.T) {
	pg := &schedulingapi.PodGroup{}
	pg.Name = "my-podgroup"
	pg.Namespace = "default"

	pgWithFinalizer := pg.DeepCopy()
	pgWithFinalizer.Finalizers = []string{schedulingapi.PodGroupProtectionFinalizer}

	cpg := &schedulingapi.CompositePodGroup{}
	cpg.Name = "my-compositepodgroup"
	cpg.Namespace = "default"

	cpgWithFinalizer := cpg.DeepCopy()
	cpgWithFinalizer.Finalizers = []string{schedulingapi.CompositePodGroupProtectionFinalizer}

	tests := []struct {
		name                   string
		genericWorkloadEnabled bool
		compositeGroupEnabled  bool
		resource               schema.GroupVersionResource
		object                 runtime.Object
		expectedObject         runtime.Object
		namespace              string
	}{
		{
			name:                   "podgroup create with plugin enabled, add finalizer",
			genericWorkloadEnabled: true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("podgroups"),
			object:                 pg,
			expectedObject:         pgWithFinalizer,
			namespace:              pg.Namespace,
		},
		{
			name:                   "podgroup finalizer already exists, no new finalizer",
			genericWorkloadEnabled: true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("podgroups"),
			object:                 pgWithFinalizer,
			expectedObject:         pgWithFinalizer,
			namespace:              pgWithFinalizer.Namespace,
		},
		{
			name:           "podgroup create with plugin disabled, no finalizer added",
			resource:       schedulingapi.SchemeGroupVersion.WithResource("podgroups"),
			object:         pg,
			expectedObject: pg,
			namespace:      pg.Namespace,
		},
		{
			name:                   "compositepodgroup create with plugin enabled, add finalizer",
			genericWorkloadEnabled: true,
			compositeGroupEnabled:  true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("compositepodgroups"),
			object:                 cpg,
			expectedObject:         cpgWithFinalizer,
			namespace:              cpg.Namespace,
		},
		{
			name:                   "compositepodgroup finalizer already exists, no new finalizer",
			genericWorkloadEnabled: true,
			compositeGroupEnabled:  true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("compositepodgroups"),
			object:                 cpgWithFinalizer,
			expectedObject:         cpgWithFinalizer,
			namespace:              cpgWithFinalizer.Namespace,
		},
		{
			name:                   "compositepodgroup create with CompositePodGroup feature disabled, no finalizer added",
			genericWorkloadEnabled: true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("compositepodgroups"),
			object:                 cpg,
			expectedObject:         cpg,
			namespace:              cpg.Namespace,
		},
		{
			name:           "compositepodgroup create with GenericWorkload feature disabled, no finalizer added",
			resource:       schedulingapi.SchemeGroupVersion.WithResource("compositepodgroups"),
			object:         cpg,
			expectedObject: cpg,
			namespace:      cpg.Namespace,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 test.genericWorkloadEnabled,
				features.TopologyAwareWorkloadScheduling: test.compositeGroupEnabled,
				features.CompositePodGroup:               test.compositeGroupEnabled,
			})

			ctrl := newPlugin()
			ctrl.InspectFeatureGates(utilfeature.DefaultFeatureGate)

			obj := test.object.DeepCopyObject()
			attrs := admission.NewAttributesRecord(
				obj,
				obj.DeepCopyObject(),
				schema.GroupVersionKind{},
				test.namespace,
				"foo",
				test.resource,
				"",
				admission.Create,
				&metav1.CreateOptions{},
				false,
				nil,
			)

			if err := ctrl.Admit(t.Context(), attrs, nil); err != nil {
				t.Errorf("got unexpected error: %v", err)
			}
			if diff := cmp.Diff(test.expectedObject, obj); diff != "" {
				t.Errorf("unexpected object diff (-want +got):\n%s", diff)
			}
		})
	}
}

func TestValidateInitialization(t *testing.T) {
	t.Run("uninspected feature gates", func(t *testing.T) {
		ctrl := newPlugin()
		if err := ctrl.ValidateInitialization(); err == nil {
			t.Errorf("expected error for uninspected feature gates")
		}
	})

	t.Run("inspected feature gates", func(t *testing.T) {
		ctrl := newPlugin()
		ctrl.InspectFeatureGates(utilfeature.DefaultFeatureGate)
		if err := ctrl.ValidateInitialization(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})
}
