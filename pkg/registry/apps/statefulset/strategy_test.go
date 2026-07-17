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
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

var ignoreErrValueDetail = cmpopts.IgnoreFields(field.Error{}, "BadValue", "Detail")

func TestStatefulSetStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("StatefulSet must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate(context.Background()) {
		t.Errorf("StatefulSet should not allow create on update")
	}

	validSelector := map[string]string{"a": "b"}
	validPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: validSelector,
			},
			Spec: api.PodSpec{
				RestartPolicy:                 api.RestartPolicyAlways,
				DNSPolicy:                     api.DNSClusterFirst,
				Containers:                    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
				TerminationGracePeriodSeconds: ptr.To[int64](30),
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
	t.Run("StatefulSet minReadySeconds field validations on creation and updation", func(t *testing.T) {
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

	t.Run("StatefulSet revisionHistoryLimit field validations on creation and updation", func(t *testing.T) {
		// Test creation
		oldSts := &apps.StatefulSet{
			ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
			Spec: apps.StatefulSetSpec{
				PodManagementPolicy:  apps.OrderedReadyPodManagement,
				Selector:             &metav1.LabelSelector{MatchLabels: validSelector},
				Template:             validPodTemplate.Template,
				UpdateStrategy:       apps.StatefulSetUpdateStrategy{Type: apps.RollingUpdateStatefulSetStrategyType},
				RevisionHistoryLimit: ptr.To(int32(-2)),
			},
		}

		warnings := Strategy.WarningsOnCreate(ctx, oldSts)
		if len(warnings) != 1 {
			t.Errorf("expected one warning got %v", warnings)
		}
		if warnings[0] != "spec.revisionHistoryLimit: a negative value retains all historical revisions; a value >= 0 is recommended" {
			t.Errorf("unexpected warning: %v", warnings)
		}
		oldSts.Spec.RevisionHistoryLimit = ptr.To(int32(2))
		newSts := oldSts.DeepCopy()
		newSts.Spec.RevisionHistoryLimit = ptr.To(int32(-2))
		warnings = Strategy.WarningsOnUpdate(ctx, newSts, oldSts)
		if len(warnings) != 1 {
			t.Errorf("expected one warning got %v", warnings)
		}
		if warnings[0] != "spec.revisionHistoryLimit: a negative value retains all historical revisions; a value >= 0 is recommended" {
			t.Errorf("unexpected warning: %v", warnings)
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

	t.Run("PersistentVolumeClaimRetentionPolicy should be updated", func(t *testing.T) {
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
	t.Run("PersistentVolumeClaimRetentionPolicy should not be updated", func(t *testing.T) {
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
			t.Errorf("unexpected failure with PersistentVolumeClaimRetentionPolicy: %v", errs)
		}
	})

	validPs.Spec.Selector = &metav1.LabelSelector{MatchLabels: map[string]string{"a": "bar"}}
	Strategy.PrepareForUpdate(ctx, validPs, ps)
	errs = Strategy.ValidateUpdate(ctx, validPs, ps)
	if len(errs) == 0 {
		t.Errorf("expected a validation error since updates are disallowed on statefulsets.")
	}
}

func TestStatefulSetStatusStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !StatusStrategy.NamespaceScoped() {
		t.Errorf("StatefulSet must be namespace scoped")
	}
	if StatusStrategy.AllowCreateOnUpdate(context.Background()) {
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

func makeStatefulSetWithMaxUnavailable(maxUnavailable *int) *apps.StatefulSet {
	rollingUpdate := apps.RollingUpdateStatefulSetStrategy{}
	if maxUnavailable != nil {
		maxUnavailableIntStr := intstr.FromInt32(int32(*maxUnavailable))
		rollingUpdate = apps.RollingUpdateStatefulSetStrategy{
			MaxUnavailable: &maxUnavailableIntStr,
		}
	}

	return &apps.StatefulSet{
		Spec: apps.StatefulSetSpec{
			UpdateStrategy: apps.StatefulSetUpdateStrategy{
				Type:          apps.RollingUpdateStatefulSetStrategyType,
				RollingUpdate: &rollingUpdate,
			},
		},
	}
}

func makeValidStatefulSetWithUpdateStrategy(strategyType apps.StatefulSetUpdateStrategyType) *apps.StatefulSet {
	validSelector := map[string]string{"a": "b"}
	return &apps.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: apps.StatefulSetSpec{
			PodManagementPolicy: apps.OrderedReadyPodManagement,
			UpdateStrategy:      apps.StatefulSetUpdateStrategy{Type: strategyType},
			Selector:            &metav1.LabelSelector{MatchLabels: validSelector},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: validSelector},
				Spec: api.PodSpec{
					RestartPolicy:                 api.RestartPolicyAlways,
					DNSPolicy:                     api.DNSClusterFirst,
					Containers:                    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
					TerminationGracePeriodSeconds: ptr.To[int64](30),
				},
			},
		},
		Status: apps.StatefulSetStatus{Replicas: 3},
	}
}

// TestDropStatefulSetDisabledFields tests if the drop functionality is working fine or not
func TestDropStatefulSetDisabledFields(t *testing.T) {
	testCases := []struct {
		name                 string
		enableMaxUnavailable bool
		ss                   *apps.StatefulSet
		oldSS                *apps.StatefulSet
		expectedSS           *apps.StatefulSet
	}{
		{
			name:       "MaxUnavailable not enabled, field not used",
			ss:         makeStatefulSetWithMaxUnavailable(nil),
			oldSS:      nil,
			expectedSS: makeStatefulSetWithMaxUnavailable(nil),
		},
		{
			name:                 "MaxUnavailable not enabled, field used in new, not in old",
			enableMaxUnavailable: false,
			ss:                   makeStatefulSetWithMaxUnavailable(ptr.To(3)),
			oldSS:                nil,
			expectedSS:           makeStatefulSetWithMaxUnavailable(nil),
		},
		{
			name:                 "MaxUnavailable not enabled, field used in old and new",
			enableMaxUnavailable: false,
			ss:                   makeStatefulSetWithMaxUnavailable(ptr.To(3)),
			oldSS:                makeStatefulSetWithMaxUnavailable(ptr.To(3)),
			expectedSS:           makeStatefulSetWithMaxUnavailable(ptr.To(3)),
		},
		{
			name:                 "MaxUnavailable enabled, field used in new only",
			enableMaxUnavailable: true,
			ss:                   makeStatefulSetWithMaxUnavailable(ptr.To(3)),
			oldSS:                nil,
			expectedSS:           makeStatefulSetWithMaxUnavailable(ptr.To(3)),
		},
		{
			name:                 "MaxUnavailable enabled, field used in both old and new",
			enableMaxUnavailable: true,
			ss:                   makeStatefulSetWithMaxUnavailable(ptr.To(1)),
			oldSS:                makeStatefulSetWithMaxUnavailable(ptr.To(3)),
			expectedSS:           makeStatefulSetWithMaxUnavailable(ptr.To(1)),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MaxUnavailableStatefulSet, tc.enableMaxUnavailable)
			old := tc.oldSS.DeepCopy()

			dropStatefulSetDisabledFields(tc.ss, tc.oldSS)

			// old obj should never be changed
			if diff := cmp.Diff(tc.oldSS, old); diff != "" {
				t.Fatalf("%v: old statefulSet changed: %v", tc.name, diff)
			}

			if diff := cmp.Diff(tc.expectedSS, tc.ss); diff != "" {
				t.Fatalf("%v: unexpected statefulSet spec: %v, want %v, got %v", tc.name, diff, tc.expectedSS, tc.ss)
			}
		})
	}
}

func TestStatefulSetStrategy_RecreateStrategy_Validate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	cases := map[string]struct {
		enableRecreateStrategyFG bool
		set                      *apps.StatefulSet
		wantErrs                 field.ErrorList
	}{
		"Validate create with Recreate strategy when feature gate is enabled": {
			enableRecreateStrategyFG: true,
			set:                      makeValidStatefulSetWithUpdateStrategy(apps.RecreateStatefulSetStrategyType),
		},
		"Validate create with Recreate strategy when feature gate is disabled": {
			enableRecreateStrategyFG: false,
			set:                      makeValidStatefulSetWithUpdateStrategy(apps.RecreateStatefulSetStrategyType),
			wantErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "updateStrategy"), nil, ""),
			},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StatefulSetRecreateStrategy, tc.enableRecreateStrategyFG)

			Strategy.PrepareForCreate(ctx, tc.set)
			if tc.set.Status.Replicas != 0 {
				t.Error("StatefulSet should not allow setting status.replicas on create")
			}
			errs := Strategy.Validate(ctx, tc.set)
			if diff := cmp.Diff(tc.wantErrs, errs, ignoreErrValueDetail); diff != "" {
				t.Errorf("Unexpected errors (-want,+got):\n%s", diff)
			}
		})
	}

}

func TestStatefulSetStrategy_RecreateStrategy_ValidateUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	cases := map[string]struct {
		enableRecreateStrategyFG bool
		set                      *apps.StatefulSet
		update                   func(*apps.StatefulSet)
		wantErrs                 field.ErrorList
	}{
		"Rolling to Recreate, RecreateFG on - allowed": {
			enableRecreateStrategyFG: true,
			set:                      makeValidStatefulSetWithUpdateStrategy(apps.RollingUpdateStatefulSetStrategyType),
			update: func(ss *apps.StatefulSet) {
				ss.ObjectMeta.ResourceVersion = "1"
				ss.Spec.UpdateStrategy.Type = apps.RecreateStatefulSetStrategyType
			},
		},
		"Rolling to Recreate, RecreateFG off - forbidden": {
			enableRecreateStrategyFG: false,
			set:                      makeValidStatefulSetWithUpdateStrategy(apps.RollingUpdateStatefulSetStrategyType),
			update: func(ss *apps.StatefulSet) {
				ss.ObjectMeta.ResourceVersion = "1"
				ss.Spec.UpdateStrategy.Type = apps.RecreateStatefulSetStrategyType
			},
			wantErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "updateStrategy"), nil, ""),
			},
		},
		"OnDelete to Recreate, RecreateFG on - allowed": {
			enableRecreateStrategyFG: true,
			set:                      makeValidStatefulSetWithUpdateStrategy(apps.OnDeleteStatefulSetStrategyType),
			update: func(ss *apps.StatefulSet) {
				ss.ObjectMeta.ResourceVersion = "1"
				ss.Spec.UpdateStrategy.Type = apps.RecreateStatefulSetStrategyType
			},
		},
		"OnDelete to Recreate, RecreateFG off - forbidden": {
			enableRecreateStrategyFG: false,
			set:                      makeValidStatefulSetWithUpdateStrategy(apps.OnDeleteStatefulSetStrategyType),
			update: func(ss *apps.StatefulSet) {
				ss.ObjectMeta.ResourceVersion = "1"
				ss.Spec.UpdateStrategy.Type = apps.RecreateStatefulSetStrategyType
			},
			wantErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "updateStrategy"), nil, ""),
			},
		},
		"Recreate to Recreate, RecreateFG off - allowed": {
			enableRecreateStrategyFG: false,
			set:                      makeValidStatefulSetWithUpdateStrategy(apps.RecreateStatefulSetStrategyType),
			update: func(ss *apps.StatefulSet) {
				ss.ObjectMeta.ResourceVersion = "1"
				ss.Spec.UpdateStrategy.Type = apps.RecreateStatefulSetStrategyType
			},
		},
		"Recreate to Recreate, RecreateFG on - allowed": {
			enableRecreateStrategyFG: true,
			set:                      makeValidStatefulSetWithUpdateStrategy(apps.RecreateStatefulSetStrategyType),
			update: func(ss *apps.StatefulSet) {
				ss.ObjectMeta.ResourceVersion = "1"
				ss.Spec.UpdateStrategy.Type = apps.RecreateStatefulSetStrategyType
			},
		},
		"Recreate to RollingUpdate, RecreateFG on - allowed": {
			enableRecreateStrategyFG: true,
			set:                      makeValidStatefulSetWithUpdateStrategy(apps.RecreateStatefulSetStrategyType),
			update: func(ss *apps.StatefulSet) {
				ss.ObjectMeta.ResourceVersion = "1"
				ss.Spec.UpdateStrategy.Type = apps.RollingUpdateStatefulSetStrategyType
			},
		},
		"Recreate to RollingUpdate, RecreateFG off - allowed": {
			enableRecreateStrategyFG: false,
			set:                      makeValidStatefulSetWithUpdateStrategy(apps.RecreateStatefulSetStrategyType),
			update: func(ss *apps.StatefulSet) {
				ss.ObjectMeta.ResourceVersion = "1"
				ss.Spec.UpdateStrategy.Type = apps.RollingUpdateStatefulSetStrategyType
			},
		},
		"Recreate to OnDelete, RecreateFG on - allowed": {
			enableRecreateStrategyFG: true,
			set:                      makeValidStatefulSetWithUpdateStrategy(apps.RecreateStatefulSetStrategyType),
			update: func(ss *apps.StatefulSet) {
				ss.ObjectMeta.ResourceVersion = "1"
				ss.Spec.UpdateStrategy.Type = apps.OnDeleteStatefulSetStrategyType
			},
		},
		"Recreate to OnDelete, RecreateFG off - allowed": {
			enableRecreateStrategyFG: false,
			set:                      makeValidStatefulSetWithUpdateStrategy(apps.RecreateStatefulSetStrategyType),
			update: func(ss *apps.StatefulSet) {
				ss.ObjectMeta.ResourceVersion = "1"
				ss.Spec.UpdateStrategy.Type = apps.OnDeleteStatefulSetStrategyType
			},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StatefulSetRecreateStrategy, tc.enableRecreateStrategyFG)

			newSet := tc.set.DeepCopy()
			tc.update(newSet)
			Strategy.PrepareForUpdate(ctx, newSet, tc.set)
			errs := Strategy.ValidateUpdate(ctx, newSet, tc.set)
			if diff := cmp.Diff(tc.wantErrs, errs, ignoreErrValueDetail); diff != "" {
				t.Errorf("Unexpected errors (-want,+got):\n%s", diff)
			}
		})
	}

}
