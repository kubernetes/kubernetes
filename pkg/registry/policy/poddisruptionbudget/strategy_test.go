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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/utils/ptr"
)

func TestPodDisruptionBudgetStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("PodDisruptionBudget must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("PodDisruptionBudget should not allow create on update")
	}

	validSelector := map[string]string{"a": "b"}
	minAvailable := intstr.FromInt32(3)
	pdb := &policy.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: policy.PodDisruptionBudgetSpec{
			MinAvailable:               &minAvailable,
			Selector:                   &metav1.LabelSelector{MatchLabels: validSelector},
			UnhealthyPodEvictionPolicy: ptr.To(policy.AlwaysAllow),
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

	// Changing UnhealthyPodEvictionPolicy? OK
	newPdb.Spec.UnhealthyPodEvictionPolicy = ptr.To(policy.IfHealthyBudget)
	Strategy.PrepareForUpdate(ctx, newPdb, pdb)
	errs = Strategy.ValidateUpdate(ctx, newPdb, pdb)
	if len(errs) != 0 {
		t.Errorf("Expected no error on changing UnhealthyPodEvictionPolicy on poddisruptionbudgets.")
	}
	if *newPdb.Spec.UnhealthyPodEvictionPolicy != policy.IfHealthyBudget {
		t.Errorf("Unexpected UnhealthyPodEvictionPolicy: expected %v, got %v", *newPdb.Spec.UnhealthyPodEvictionPolicy, policy.IfHealthyBudget)
	}

	// Changing to invalid UnhealthyPodEvictionPolicy.
	newPdb.Spec.UnhealthyPodEvictionPolicy = ptr.To(policy.UnhealthyPodEvictionPolicyType("invalid"))
	Strategy.PrepareForUpdate(ctx, newPdb, pdb)
	errs = Strategy.ValidateUpdate(ctx, newPdb, pdb)
	if len(errs) == 0 {
		t.Errorf("Expected error on changing to invalid UnhealthyPodEvictionPolicy on poddisruptionbudgets.")
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

	oldMinAvailable := intstr.FromInt32(3)
	newMinAvailable := intstr.FromInt32(2)

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

			oldMaxUnavailable := intstr.FromInt32(2)
			newMaxUnavailable := intstr.FromInt32(3)
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
