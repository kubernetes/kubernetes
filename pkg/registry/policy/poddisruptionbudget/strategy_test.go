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

package poddisruptionbudget

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/features"
)

type podHealthyPolicyStrategyTestCase struct {
	name                                             string
	enablePodHealthyPolicy                           bool
	disablePDBPodHealthyPolicyFeatureGateAfterCreate bool
	podHealthyPolicy                                 policy.PodHealthyPolicy
	expectedPodHealthyPolicy                         policy.PodHealthyPolicy
	expectedValidationErr                            bool
	updatePodHealthyPolicy                           policy.PodHealthyPolicy
	expectedUpdatePodHealthyPolicy                   policy.PodHealthyPolicy
	expectedValidationUpdateErr                      bool
}

func TestPodDisruptionBudgetStrategy(t *testing.T) {
	tests := map[string]bool{
		"PodDisruptionBudget strategy with PDBPodHealthyPolicy feature gate disabled": false,
		"PodDisruptionBudget strategy with PDBPodHealthyPolicy feature gate enabled":  true,
	}

	for name, enablePodHealthyPolicy := range tests {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PDBPodHealthyPolicy, enablePodHealthyPolicy)()
			testPodDisruptionBudgetStrategy(t)
		})
	}

	healthyPolicyTests := []podHealthyPolicyStrategyTestCase{
		{
			name:                   "PodDisruptionBudget strategy with FeatureGate disabled should remove podHealthyPolicy",
			enablePodHealthyPolicy: false,
			podHealthyPolicy:       policy.PodRunning,
			updatePodHealthyPolicy: policy.PodRunning,
		},
		{
			name:                   "PodDisruptionBudget strategy with FeatureGate disabled should remove invalid podHealthyPolicy",
			enablePodHealthyPolicy: false,
			podHealthyPolicy:       "Invalid",
			updatePodHealthyPolicy: "Invalid",
		},
		{
			name:                   "PodDisruptionBudget strategy with FeatureGate enabled",
			enablePodHealthyPolicy: true,
		},
		{
			name:                           "PodDisruptionBudget strategy with FeatureGate enabled should respect podHealthyPolicy",
			enablePodHealthyPolicy:         true,
			podHealthyPolicy:               policy.PodReady,
			expectedPodHealthyPolicy:       policy.PodReady,
			updatePodHealthyPolicy:         policy.PodRunning,
			expectedUpdatePodHealthyPolicy: policy.PodRunning,
		},
		{
			name:                   "PodDisruptionBudget strategy with FeatureGate enabled should fail invalid podHealthyPolicy",
			enablePodHealthyPolicy: true,
			podHealthyPolicy:       "Invalid",
			expectedValidationErr:  true,
		},
		{
			name:                        "PodDisruptionBudget strategy with FeatureGate enabled should fail invalid podHealthyPolicy when updated",
			enablePodHealthyPolicy:      true,
			updatePodHealthyPolicy:      "Invalid",
			expectedValidationUpdateErr: true,
		},
		{
			name:                   "PodDisruptionBudget strategy with podHealthyPolicy should be preserved when feature gate is disabled",
			enablePodHealthyPolicy: true,
			disablePDBPodHealthyPolicyFeatureGateAfterCreate: true,
			podHealthyPolicy:               policy.PodReady,
			expectedPodHealthyPolicy:       policy.PodReady,
			updatePodHealthyPolicy:         policy.PodRunning,
			expectedUpdatePodHealthyPolicy: policy.PodReady,
		},
	}

	for _, tc := range healthyPolicyTests {
		t.Run(tc.name, func(t *testing.T) {
			testPodDisruptionBudgetStrategyWithPodHealthyPolicy(t, tc)
		})
	}
}

func testPodDisruptionBudgetStrategyWithPodHealthyPolicy(t *testing.T, tc podHealthyPolicyStrategyTestCase) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PDBPodHealthyPolicy, tc.enablePodHealthyPolicy)()
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("PodDisruptionBudget must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("PodDisruptionBudget should not allow create on update")
	}

	validSelector := map[string]string{"a": "b"}
	minAvailable := intstr.FromInt(3)
	pdb := &policy.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: policy.PodDisruptionBudgetSpec{
			MinAvailable:     &minAvailable,
			Selector:         &metav1.LabelSelector{MatchLabels: validSelector},
			PodHealthyPolicy: tc.podHealthyPolicy,
		},
	}

	Strategy.PrepareForCreate(ctx, pdb)
	errs := Strategy.Validate(ctx, pdb)
	if len(errs) != 0 {
		if !tc.expectedValidationErr {
			t.Errorf("Unexpected error validating %v", errs)
		}
		return // no point going further when we have invalid PDB
	}
	if len(errs) == 0 && tc.expectedValidationErr {
		t.Errorf("Expected error validating")
	}
	if pdb.Spec.PodHealthyPolicy != tc.expectedPodHealthyPolicy {
		t.Errorf("Unexpected PodHealthyPolicy set: expected %v, got %v", tc.expectedPodHealthyPolicy, pdb.Spec.PodHealthyPolicy)
	}
	if tc.disablePDBPodHealthyPolicyFeatureGateAfterCreate {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PDBPodHealthyPolicy, false)()
	}

	newPdb := &policy.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{Name: pdb.Name, Namespace: pdb.Namespace},
		Spec:       pdb.Spec,
	}
	if len(tc.updatePodHealthyPolicy) > 0 {
		newPdb.Spec.PodHealthyPolicy = tc.updatePodHealthyPolicy
	}

	// Nothing in Spec changes: OK
	Strategy.PrepareForUpdate(ctx, newPdb, pdb)
	errs = Strategy.ValidateUpdate(ctx, newPdb, pdb)

	if len(errs) != 0 {
		if !tc.expectedValidationUpdateErr {
			t.Errorf("Unexpected error updating PodDisruptionBudget %v", errs)
		}
		return // no point going further when we have invalid PDB
	}
	if len(errs) == 0 && tc.expectedValidationUpdateErr {
		t.Errorf("Expected error updating PodDisruptionBudget")
	}
	if newPdb.Spec.PodHealthyPolicy != tc.expectedUpdatePodHealthyPolicy {
		t.Errorf("Unexpected PodHealthyPolicy set: expected %v, got %v", tc.expectedUpdatePodHealthyPolicy, newPdb.Spec.PodHealthyPolicy)
	}
}

func testPodDisruptionBudgetStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("PodDisruptionBudget must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("PodDisruptionBudget should not allow create on update")
	}

	validSelector := map[string]string{"a": "b"}
	minAvailable := intstr.FromInt(3)
	pdb := &policy.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: policy.PodDisruptionBudgetSpec{
			MinAvailable: &minAvailable,
			Selector:     &metav1.LabelSelector{MatchLabels: validSelector},
		},
	}

	Strategy.PrepareForCreate(ctx, pdb)
	errs := Strategy.Validate(ctx, pdb)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}

	newPdb := &policy.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{Name: pdb.Name, Namespace: pdb.Namespace},
		Spec:       pdb.Spec,
		Status: policy.PodDisruptionBudgetStatus{
			DisruptionsAllowed: 1,
			CurrentHealthy:     3,
			DesiredHealthy:     3,
			ExpectedPods:       3,
		},
	}

	// Nothing in Spec changes: OK
	Strategy.PrepareForUpdate(ctx, newPdb, pdb)
	errs = Strategy.ValidateUpdate(ctx, newPdb, pdb)
	if len(errs) != 0 {
		t.Errorf("Unexpected error updating PodDisruptionBudget.")
	}

	// Changing the selector?  OK
	newPdb.Spec.Selector = &metav1.LabelSelector{MatchLabels: map[string]string{"a": "bar"}}
	Strategy.PrepareForUpdate(ctx, newPdb, pdb)
	errs = Strategy.ValidateUpdate(ctx, newPdb, pdb)
	if len(errs) != 0 {
		t.Errorf("Expected no error on changing selector on poddisruptionbudgets.")
	}
	newPdb.Spec.Selector = pdb.Spec.Selector

	// Changing MinAvailable?  OK
	newMinAvailable := intstr.FromString("28%")
	newPdb.Spec.MinAvailable = &newMinAvailable
	Strategy.PrepareForUpdate(ctx, newPdb, pdb)
	errs = Strategy.ValidateUpdate(ctx, newPdb, pdb)
	if len(errs) != 0 {
		t.Errorf("Expected no error updating MinAvailable on poddisruptionbudgets.")
	}

	// Changing MinAvailable to MaxAvailable? OK
	maxUnavailable := intstr.FromString("28%")
	newPdb.Spec.MaxUnavailable = &maxUnavailable
	newPdb.Spec.MinAvailable = nil
	Strategy.PrepareForUpdate(ctx, newPdb, pdb)
	errs = Strategy.ValidateUpdate(ctx, newPdb, pdb)
	if len(errs) != 0 {
		t.Errorf("Expected no error updating replacing MinAvailable with MaxUnavailable on poddisruptionbudgets.")
	}
}

func TestPodDisruptionBudgetStatusStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !StatusStrategy.NamespaceScoped() {
		t.Errorf("PodDisruptionBudgetStatus must be namespace scoped")
	}
	if StatusStrategy.AllowCreateOnUpdate() {
		t.Errorf("PodDisruptionBudgetStatus should not allow create on update")
	}

	oldMinAvailable := intstr.FromInt(3)
	newMinAvailable := intstr.FromInt(2)

	validSelector := map[string]string{"a": "b"}
	oldPdb := &policy.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault, ResourceVersion: "10"},
		Spec: policy.PodDisruptionBudgetSpec{
			Selector:     &metav1.LabelSelector{MatchLabels: validSelector},
			MinAvailable: &oldMinAvailable,
		},
		Status: policy.PodDisruptionBudgetStatus{
			DisruptionsAllowed: 1,
			CurrentHealthy:     3,
			DesiredHealthy:     3,
			ExpectedPods:       3,
		},
	}
	newPdb := &policy.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault, ResourceVersion: "9"},
		Spec: policy.PodDisruptionBudgetSpec{
			Selector:     &metav1.LabelSelector{MatchLabels: validSelector},
			MinAvailable: &newMinAvailable,
		},
		Status: policy.PodDisruptionBudgetStatus{
			DisruptionsAllowed: 0,
			CurrentHealthy:     2,
			DesiredHealthy:     3,
			ExpectedPods:       3,
		},
	}
	StatusStrategy.PrepareForUpdate(ctx, newPdb, oldPdb)
	if newPdb.Status.CurrentHealthy != 2 {
		t.Errorf("PodDisruptionBudget status updates should allow change of CurrentHealthy: %v", newPdb.Status.CurrentHealthy)
	}
	if newPdb.Spec.MinAvailable.IntValue() != 3 {
		t.Errorf("PodDisruptionBudget status updates should not clobber spec: %v", newPdb.Spec)
	}
	errs := StatusStrategy.ValidateUpdate(ctx, newPdb, oldPdb)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}
}

func TestPodDisruptionBudgetStatusValidationByApiVersion(t *testing.T) {
	testCases := map[string]struct {
		apiVersion string
		validation bool
	}{
		"policy/v1beta1 should not do update validation": {
			apiVersion: "v1beta1",
			validation: false,
		},
		"policy/v1 should do update validation": {
			apiVersion: "v1",
			validation: true,
		},
		"policy/some-version should do update validation": {
			apiVersion: "some-version",
			validation: true,
		},
	}

	for tn, tc := range testCases {
		t.Run(tn, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIGroup:   "policy",
					APIVersion: tc.apiVersion,
				})

			oldMaxUnavailable := intstr.FromInt(2)
			newMaxUnavailable := intstr.FromInt(3)
			oldPdb := &policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault, ResourceVersion: "10"},
				Spec: policy.PodDisruptionBudgetSpec{
					Selector:       &metav1.LabelSelector{MatchLabels: map[string]string{"a": "b"}},
					MaxUnavailable: &oldMaxUnavailable,
				},
				Status: policy.PodDisruptionBudgetStatus{
					DisruptionsAllowed: 1,
				},
			}
			newPdb := &policy.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault, ResourceVersion: "9"},
				Spec: policy.PodDisruptionBudgetSpec{
					Selector:     &metav1.LabelSelector{MatchLabels: map[string]string{"a": "b"}},
					MinAvailable: &newMaxUnavailable,
				},
				Status: policy.PodDisruptionBudgetStatus{
					DisruptionsAllowed: -1, // This is not allowed, so should trigger validation error.
				},
			}

			errs := StatusStrategy.ValidateUpdate(ctx, newPdb, oldPdb)
			hasErrors := len(errs) > 0
			if !tc.validation && hasErrors {
				t.Errorf("Validation failed when no validation should happen")
			}
			if tc.validation && !hasErrors {
				t.Errorf("Expected validation errors but didn't get any")
			}
		})
	}
}

func TestDropDisabledFields(t *testing.T) {
	tests := map[string]struct {
		oldSpec                *policy.PodDisruptionBudgetSpec
		newSpec                *policy.PodDisruptionBudgetSpec
		expectNewSpec          *policy.PodDisruptionBudgetSpec
		enablePodHealthyPolicy bool
	}{
		"disabled clears pod healthy policy": {
			enablePodHealthyPolicy: false,
			oldSpec:                nil,
			newSpec:                specWithPodHealthyPolicy(policy.PodRunning),
			expectNewSpec:          specWithPodHealthyPolicy(""),
		},
		"disabled does not allow updating pod healthy policy": {
			enablePodHealthyPolicy: false,
			oldSpec:                specWithPodHealthyPolicy(""),
			newSpec:                specWithPodHealthyPolicy(policy.PodRunning),
			expectNewSpec:          specWithPodHealthyPolicy(""),
		},
		"disabled preserves old pod healthy policy when both old and new have it": {
			enablePodHealthyPolicy: false,
			oldSpec:                specWithPodHealthyPolicy(policy.PodRunning),
			newSpec:                specWithPodHealthyPolicy(policy.PodRunning),
			expectNewSpec:          specWithPodHealthyPolicy(policy.PodRunning),
		},
		"disabled does not update pod healthy policy": {
			enablePodHealthyPolicy: false,
			oldSpec:                specWithPodHealthyPolicy(policy.PodRunning),
			newSpec:                specWithPodHealthyPolicy(policy.PodRunning),
			expectNewSpec:          specWithPodHealthyPolicy(policy.PodRunning),
		},
		"enabled preserve pod healthy policy": {
			enablePodHealthyPolicy: true,
			oldSpec:                nil,
			newSpec:                specWithPodHealthyPolicy(policy.PodRunning),
			expectNewSpec:          specWithPodHealthyPolicy(policy.PodRunning),
		},
		"enabled allows updating pod healthy policy": {
			enablePodHealthyPolicy: true,
			oldSpec:                specWithPodHealthyPolicy(""),
			newSpec:                specWithPodHealthyPolicy(policy.PodRunning),
			expectNewSpec:          specWithPodHealthyPolicy(policy.PodRunning),
		},
		"enabled preserve pod healthy policy when both old and new have it": {
			enablePodHealthyPolicy: true,
			oldSpec:                specWithPodHealthyPolicy(policy.PodRunning),
			newSpec:                specWithPodHealthyPolicy(policy.PodRunning),
			expectNewSpec:          specWithPodHealthyPolicy(policy.PodRunning),
		},
		"enabled updates pod healthy policy": {
			enablePodHealthyPolicy: true,
			oldSpec:                specWithPodHealthyPolicy(policy.PodRunning),
			newSpec:                specWithPodHealthyPolicy(policy.PodReady),
			expectNewSpec:          specWithPodHealthyPolicy(policy.PodReady),
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PDBPodHealthyPolicy, tc.enablePodHealthyPolicy)()

			oldSpecBefore := tc.oldSpec.DeepCopy()
			dropDisabledFields(tc.newSpec, tc.oldSpec)
			if !reflect.DeepEqual(tc.newSpec, tc.expectNewSpec) {
				t.Error(cmp.Diff(tc.newSpec, tc.expectNewSpec))
			}
			if !reflect.DeepEqual(tc.oldSpec, oldSpecBefore) {
				t.Error(cmp.Diff(tc.oldSpec, oldSpecBefore))
			}
		})
	}
}

func specWithPodHealthyPolicy(podHealthyPolicy policy.PodHealthyPolicy) *policy.PodDisruptionBudgetSpec {
	return &policy.PodDisruptionBudgetSpec{
		PodHealthyPolicy: podHealthyPolicy,
	}
}
