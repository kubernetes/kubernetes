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

package statefulset

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

func TestStatefulSetStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("StatefulSet must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("StatefulSet should not allow create on update")
	}

	validSelector := map[string]string{"a": "b"}
	validPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: validSelector,
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
	}
	ps := &apps.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: apps.StatefulSetSpec{
			PodManagementPolicy: apps.OrderedReadyPodManagement,
			Selector:            &metav1.LabelSelector{MatchLabels: validSelector},
			Template:            validPodTemplate.Template,
			UpdateStrategy:      apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
		},
		Status: apps.StatefulSetStatus{Replicas: 3},
	}

	Strategy.PrepareForCreate(ctx, ps)
	if ps.Status.Replicas != 0 {
		t.Error("StatefulSet should not allow setting status.replicas on create")
	}
	errs := Strategy.Validate(ctx, ps)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}
	newMinReadySeconds := int32(50)
	// Just Spec.Replicas is allowed to change
	validPs := &apps.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{Name: ps.Name, Namespace: ps.Namespace, ResourceVersion: "1", Generation: 1},
		Spec: apps.StatefulSetSpec{
			PodManagementPolicy: apps.OrderedReadyPodManagement,
			Selector:            ps.Spec.Selector,
			Template:            validPodTemplate.Template,
			UpdateStrategy:      apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
			MinReadySeconds:     newMinReadySeconds,
		},
		Status: apps.StatefulSetStatus{Replicas: 4},
	}
	Strategy.PrepareForUpdate(ctx, validPs, ps)
	t.Run("when minReadySeconds feature gate is enabled", func(t *testing.T) {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StatefulSetMinReadySeconds, true)()
		// Test creation
		ps := &apps.StatefulSet{
			ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
			Spec: apps.StatefulSetSpec{
				PodManagementPolicy: apps.OrderedReadyPodManagement,
				Selector:            &metav1.LabelSelector{MatchLabels: validSelector},
				Template:            validPodTemplate.Template,
				UpdateStrategy:      apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
				MinReadySeconds:     int32(-1),
			},
		}
		Strategy.PrepareForCreate(ctx, ps)
		errs := Strategy.Validate(ctx, ps)
		if len(errs) == 0 {
			t.Errorf("expected failure when MinReadySeconds is not positive number but got no error %v", errs)
		}
		expectedCreateErrorString := "spec.minReadySeconds: Invalid value: -1: must be greater than or equal to 0"
		if errs[0].Error() != expectedCreateErrorString {
			t.Errorf("mismatched error string %v", errs[0].Error())
		}
		// Test updation
		newMinReadySeconds := int32(50)
		// Just Spec.Replicas is allowed to change
		validPs := &apps.StatefulSet{
			ObjectMeta: metav1.ObjectMeta{Name: ps.Name, Namespace: ps.Namespace, ResourceVersion: "1", Generation: 1},
			Spec: apps.StatefulSetSpec{
				PodManagementPolicy: apps.OrderedReadyPodManagement,
				Selector:            ps.Spec.Selector,
				Template:            validPodTemplate.Template,
				UpdateStrategy:      apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
				MinReadySeconds:     newMinReadySeconds,
			},
			Status: apps.StatefulSetStatus{Replicas: 4},
		}
		Strategy.PrepareForUpdate(ctx, validPs, ps)
		errs = Strategy.ValidateUpdate(ctx, validPs, ps)
		if len(errs) != 0 {
			t.Errorf("updating spec.Replicas and minReadySeconds is allowed on a statefulset: %v", errs)
		}
		invalidPs := ps
		invalidPs.Spec.MinReadySeconds = int32(-1)
		Strategy.PrepareForUpdate(ctx, validPs, invalidPs)
		errs = Strategy.ValidateUpdate(ctx, validPs, ps)
		if len(errs) != 0 {
			t.Errorf("updating spec.Replicas and minReadySeconds is allowed on a statefulset: %v", errs)
		}
		if validPs.Spec.MinReadySeconds != newMinReadySeconds {
			t.Errorf("expected minReadySeconds to not be changed %v", errs)
		}
	})
	t.Run("when minReadySeconds feature gate is disabled, the minReadySeconds should not be updated",
		func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StatefulSetMinReadySeconds, false)()
			// Test creation
			ps := &apps.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: apps.StatefulSetSpec{
					PodManagementPolicy: apps.OrderedReadyPodManagement,
					Selector:            &metav1.LabelSelector{MatchLabels: validSelector},
					Template:            validPodTemplate.Template,
					UpdateStrategy:      apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
					MinReadySeconds:     int32(-1),
				},
			}
			Strategy.PrepareForCreate(ctx, ps)
			errs := Strategy.Validate(ctx, ps)
			if len(errs) != 0 {
				t.Errorf("StatefulSet creation should not have any issues but found %v", errs)
			}
			if ps.Spec.MinReadySeconds != 0 {
				t.Errorf("if the StatefulSet is created with invalid value we expect it to be defaulted to 0 "+
					"but got %v", ps.Spec.MinReadySeconds)
			}

			// Test Updation
			validPs := &apps.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: ps.Name, Namespace: ps.Namespace, ResourceVersion: "1", Generation: 1},
				Spec: apps.StatefulSetSpec{
					PodManagementPolicy: apps.OrderedReadyPodManagement,
					Selector:            ps.Spec.Selector,
					Template:            validPodTemplate.Template,
					UpdateStrategy:      apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
					MinReadySeconds:     newMinReadySeconds,
				},
				Status: apps.StatefulSetStatus{Replicas: 4},
			}
			Strategy.PrepareForUpdate(ctx, validPs, ps)
			errs = Strategy.ValidateUpdate(ctx, validPs, ps)
			if len(errs) == 0 {
				t.Errorf("updating only spec.Replicas is allowed on a statefulset: %v", errs)
			}
			expectedUpdateErrorString := "spec: Forbidden: updates to statefulset spec for fields other than 'replicas'," +
				" 'template', 'updateStrategy', 'minReadySeconds' and 'persistentVolumeClaimRetentionPolicy' are forbidden"
			if errs[0].Error() != expectedUpdateErrorString {
				t.Errorf("expected error string %v", errs[0].Error())
			}
		})

	validPs = &apps.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{Name: ps.Name, Namespace: ps.Namespace, ResourceVersion: "1", Generation: 1},
		Spec: apps.StatefulSetSpec{
			PodManagementPolicy: apps.OrderedReadyPodManagement,
			Selector:            ps.Spec.Selector,
			Template:            validPodTemplate.Template,
			UpdateStrategy:      apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
			PersistentVolumeClaimRetentionPolicy: &apps.StatefulSetPersistentVolumeClaimRetentionPolicy{
				WhenDeleted: apps.RetainPersistentVolumeClaimRetentionPolicyType,
				WhenScaled:  apps.DeletePersistentVolumeClaimRetentionPolicyType,
			},
		},
		Status: apps.StatefulSetStatus{Replicas: 4},
	}

	t.Run("when StatefulSetAutoDeletePVC feature gate is enabled, PersistentVolumeClaimRetentionPolicy should be updated", func(t *testing.T) {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StatefulSetAutoDeletePVC, true)()
		// Test creation
		ps := &apps.StatefulSet{
			ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
			Spec: apps.StatefulSetSpec{
				PodManagementPolicy: apps.OrderedReadyPodManagement,
				Selector:            ps.Spec.Selector,
				Template:            validPodTemplate.Template,
				UpdateStrategy:      apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
				PersistentVolumeClaimRetentionPolicy: &apps.StatefulSetPersistentVolumeClaimRetentionPolicy{
					WhenDeleted: apps.PersistentVolumeClaimRetentionPolicyType("invalid policy"),
				},
			},
		}
		Strategy.PrepareForCreate(ctx, ps)
		errs := Strategy.Validate(ctx, ps)
		if len(errs) == 0 {
			t.Errorf("expected failure when PersistentVolumeClaimRetentionPolicy is invalid")
		}
		expectedCreateErrorString := "spec.persistentVolumeClaimRetentionPolicy.whenDeleted: Unsupported value: \"invalid policy\": supported values: \"Retain\", \"Delete\""
		if errs[0].Error() != expectedCreateErrorString {
			t.Errorf("mismatched error string %v (expected %v)", errs[0].Error(), expectedCreateErrorString)
		}
		Strategy.PrepareForUpdate(ctx, validPs, ps)
		errs = Strategy.ValidateUpdate(ctx, validPs, ps)
		if len(errs) != 0 {
			t.Errorf("updates to PersistentVolumeClaimRetentionPolicy should be allowed: %v", errs)
		}
		invalidPs := ps
		invalidPs.Spec.PersistentVolumeClaimRetentionPolicy.WhenDeleted = apps.PersistentVolumeClaimRetentionPolicyType("invalid type")
		Strategy.PrepareForUpdate(ctx, validPs, invalidPs)
		errs = Strategy.ValidateUpdate(ctx, validPs, ps)
		if len(errs) != 0 {
			t.Errorf("invalid updates to PersistentVolumeClaimRetentionPolicyType should be allowed: %v", errs)
		}
		if validPs.Spec.PersistentVolumeClaimRetentionPolicy.WhenDeleted != apps.RetainPersistentVolumeClaimRetentionPolicyType || validPs.Spec.PersistentVolumeClaimRetentionPolicy.WhenScaled != apps.DeletePersistentVolumeClaimRetentionPolicyType {
			t.Errorf("expected PersistentVolumeClaimRetentionPolicy to be updated: %v", errs)
		}
	})
	t.Run("when StatefulSetAutoDeletePVC feature gate is disabled, PersistentVolumeClaimRetentionPolicy should not be updated", func(t *testing.T) {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StatefulSetAutoDeletePVC, true)()
		// Test creation
		ps := &apps.StatefulSet{
			ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
			Spec: apps.StatefulSetSpec{
				PodManagementPolicy: apps.OrderedReadyPodManagement,
				Selector:            ps.Spec.Selector,
				Template:            validPodTemplate.Template,
				UpdateStrategy:      apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
				PersistentVolumeClaimRetentionPolicy: &apps.StatefulSetPersistentVolumeClaimRetentionPolicy{
					WhenDeleted: apps.RetainPersistentVolumeClaimRetentionPolicyType,
					WhenScaled:  apps.DeletePersistentVolumeClaimRetentionPolicyType,
				},
			},
		}
		Strategy.PrepareForCreate(ctx, ps)
		errs := Strategy.Validate(ctx, ps)
		if len(errs) != 0 {
			t.Errorf("unexpected failure with PersistentVolumeClaimRetentionPolicy: %v", errs)
		}
		if ps.Spec.PersistentVolumeClaimRetentionPolicy.WhenDeleted != apps.RetainPersistentVolumeClaimRetentionPolicyType || ps.Spec.PersistentVolumeClaimRetentionPolicy.WhenScaled != apps.DeletePersistentVolumeClaimRetentionPolicyType {
			t.Errorf("expected invalid PersistentVolumeClaimRetentionPolicy to be defaulted to Retain, but got %v", ps.Spec.PersistentVolumeClaimRetentionPolicy)
		}
		Strategy.PrepareForUpdate(ctx, validPs, ps)
		errs = Strategy.ValidateUpdate(ctx, validPs, ps)
		if len(errs) != 0 {
			t.Errorf("updates to PersistentVolumeClaimRetentionPolicy should be allowed: %v", errs)
		}
		invalidPs := ps
		invalidPs.Spec.PersistentVolumeClaimRetentionPolicy.WhenDeleted = apps.PersistentVolumeClaimRetentionPolicyType("invalid type")
		Strategy.PrepareForUpdate(ctx, validPs, invalidPs)
		errs = Strategy.ValidateUpdate(ctx, validPs, ps)
		if len(errs) != 0 {
			t.Errorf("should ignore updates to PersistentVolumeClaimRetentionPolicyType")
		}
	})

	validPs.Spec.Selector = &metav1.LabelSelector{MatchLabels: map[string]string{"a": "bar"}}
	Strategy.PrepareForUpdate(ctx, validPs, ps)
	errs = Strategy.ValidateUpdate(ctx, validPs, ps)
	if len(errs) == 0 {
		t.Errorf("expected a validation error since updates are disallowed on statefulsets.")
	}
}

func TestStatefulsetDefaultGarbageCollectionPolicy(t *testing.T) {
	// Make sure we correctly implement the interface.
	// Otherwise a typo could silently change the default.
	var gcds rest.GarbageCollectionDeleteStrategy = Strategy
	tests := []struct {
		requestInfo      genericapirequest.RequestInfo
		expectedGCPolicy rest.GarbageCollectionPolicy
		isNilRequestInfo bool
	}{
		{
			genericapirequest.RequestInfo{
				APIGroup:   "apps",
				APIVersion: "v1beta1",
				Resource:   "statefulsets",
			},
			rest.OrphanDependents,
			false,
		},
		{
			genericapirequest.RequestInfo{
				APIGroup:   "apps",
				APIVersion: "v1beta2",
				Resource:   "statefulsets",
			},
			rest.OrphanDependents,
			false,
		},
		{
			genericapirequest.RequestInfo{
				APIGroup:   "apps",
				APIVersion: "v1",
				Resource:   "statefulsets",
			},
			rest.DeleteDependents,
			false,
		},
		{
			expectedGCPolicy: rest.DeleteDependents,
			isNilRequestInfo: true,
		},
	}

	for _, test := range tests {
		context := genericapirequest.NewContext()
		if !test.isNilRequestInfo {
			context = genericapirequest.WithRequestInfo(context, &test.requestInfo)
		}
		if got, want := gcds.DefaultGarbageCollectionPolicy(context), test.expectedGCPolicy; got != want {
			t.Errorf("%s/%s: DefaultGarbageCollectionPolicy() = %#v, want %#v", test.requestInfo.APIGroup,
				test.requestInfo.APIVersion, got, want)
		}
	}
}

func TestStatefulSetStatusStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !StatusStrategy.NamespaceScoped() {
		t.Errorf("StatefulSet must be namespace scoped")
	}
	if StatusStrategy.AllowCreateOnUpdate() {
		t.Errorf("StatefulSet should not allow create on update")
	}
	validSelector := map[string]string{"a": "b"}
	validPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: validSelector,
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
	}
	oldPS := &apps.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault, ResourceVersion: "10"},
		Spec: apps.StatefulSetSpec{
			Replicas:       3,
			Selector:       &metav1.LabelSelector{MatchLabels: validSelector},
			Template:       validPodTemplate.Template,
			UpdateStrategy: apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
		},
		Status: apps.StatefulSetStatus{
			Replicas: 1,
		},
	}
	newPS := &apps.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault, ResourceVersion: "9"},
		Spec: apps.StatefulSetSpec{
			Replicas:       1,
			Selector:       &metav1.LabelSelector{MatchLabels: validSelector},
			Template:       validPodTemplate.Template,
			UpdateStrategy: apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
		},
		Status: apps.StatefulSetStatus{
			Replicas: 2,
		},
	}
	StatusStrategy.PrepareForUpdate(ctx, newPS, oldPS)
	if newPS.Status.Replicas != 2 {
		t.Errorf("StatefulSet status updates should allow change of pods: %v", newPS.Status.Replicas)
	}
	if newPS.Spec.Replicas != 3 {
		t.Errorf("StatefulSet status updates should not clobber spec: %v", newPS.Spec)
	}
	errs := StatusStrategy.ValidateUpdate(ctx, newPS, oldPS)
	if len(errs) != 0 {
		t.Errorf("unexpected error %v", errs)
	}
}

// generateStatefulSetWithMinReadySeconds generates a StatefulSet with min values
func generateStatefulSetWithMinReadySeconds(minReadySeconds int32) *apps.StatefulSet {
	return &apps.StatefulSet{
		Spec: apps.StatefulSetSpec{
			MinReadySeconds: minReadySeconds,
		},
	}
}

// TestDropStatefulSetDisabledFields tests if the drop functionality is working fine or not
func TestDropStatefulSetDisabledFields(t *testing.T) {
	testCases := []struct {
		name                  string
		enableMinReadySeconds bool
		ss                    *apps.StatefulSet
		oldSS                 *apps.StatefulSet
		expectedSS            *apps.StatefulSet
	}{
		{
			name:                  "no minReadySeconds, no update",
			enableMinReadySeconds: false,
			ss:                    &apps.StatefulSet{},
			oldSS:                 nil,
			expectedSS:            &apps.StatefulSet{},
		},
		{
			name:                  "no minReadySeconds, irrespective of the current value, set to default value of 0",
			enableMinReadySeconds: false,
			ss:                    generateStatefulSetWithMinReadySeconds(2000),
			oldSS:                 nil,
			expectedSS:            &apps.StatefulSet{Spec: apps.StatefulSetSpec{MinReadySeconds: int32(0)}},
		},
		{
			name:                  "no minReadySeconds, oldSS field set to 100, no update",
			enableMinReadySeconds: false,
			ss:                    generateStatefulSetWithMinReadySeconds(2000),
			oldSS:                 generateStatefulSetWithMinReadySeconds(100),
			expectedSS:            generateStatefulSetWithMinReadySeconds(2000),
		},
		{
			name:                  "no minReadySeconds, oldSS field set to -1(invalid value), update to zero",
			enableMinReadySeconds: false,
			ss:                    generateStatefulSetWithMinReadySeconds(2000),
			oldSS:                 generateStatefulSetWithMinReadySeconds(-1),
			expectedSS:            generateStatefulSetWithMinReadySeconds(0),
		},
		{
			name:                  "no minReadySeconds, oldSS field set to 0, no update",
			enableMinReadySeconds: false,
			ss:                    generateStatefulSetWithMinReadySeconds(2000),
			oldSS:                 generateStatefulSetWithMinReadySeconds(0),
			expectedSS:            generateStatefulSetWithMinReadySeconds(2000),
		},
		{
			name:                  "set minReadySeconds, no update",
			enableMinReadySeconds: true,
			ss:                    generateStatefulSetWithMinReadySeconds(10),
			oldSS:                 generateStatefulSetWithMinReadySeconds(20),
			expectedSS:            generateStatefulSetWithMinReadySeconds(10),
		},
		{
			name:                  "set minReadySeconds, oldSS field set to nil",
			enableMinReadySeconds: true,
			ss:                    generateStatefulSetWithMinReadySeconds(10),
			oldSS:                 nil,
			expectedSS:            generateStatefulSetWithMinReadySeconds(10),
		},
		{
			name:                  "set minReadySeconds, oldSS field is set to 0",
			enableMinReadySeconds: true,
			ss:                    generateStatefulSetWithMinReadySeconds(10),
			oldSS:                 generateStatefulSetWithMinReadySeconds(0),
			expectedSS:            generateStatefulSetWithMinReadySeconds(10),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StatefulSetMinReadySeconds, tc.enableMinReadySeconds)()
			old := tc.oldSS.DeepCopy()

			dropStatefulSetDisabledFields(tc.ss, tc.oldSS)

			// old obj should never be changed
			if !reflect.DeepEqual(tc.oldSS, old) {
				t.Fatalf("old ds changed: %v", diff.ObjectReflectDiff(tc.oldSS, old))
			}

			if !reflect.DeepEqual(tc.ss, tc.expectedSS) {
				t.Fatalf("unexpected ds spec: %v", diff.ObjectReflectDiff(tc.expectedSS, tc.ss))
			}
		})
	}
}
