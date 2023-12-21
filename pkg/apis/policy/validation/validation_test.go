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

package validation

import (
	"fmt"
	"testing"
	"time"

	policyv1beta1 "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/policy"
)

func TestValidatePodDisruptionBudgetSpec(t *testing.T) {
	minAvailable := intstr.FromString("0%")
	maxUnavailable := intstr.FromString("10%")

	spec := policy.PodDisruptionBudgetSpec{
		MinAvailable:   &minAvailable,
		MaxUnavailable: &maxUnavailable,
	}
	errs := ValidatePodDisruptionBudgetSpec(spec, PodDisruptionBudgetValidationOptions{true}, field.NewPath("foo"))
	if len(errs) == 0 {
		t.Errorf("unexpected success for %v", spec)
	}
}

func TestValidateMinAvailablePodDisruptionBudgetSpec(t *testing.T) {
	successCases := []intstr.IntOrString{
		intstr.FromString("0%"),
		intstr.FromString("1%"),
		intstr.FromString("100%"),
		intstr.FromInt32(0),
		intstr.FromInt32(1),
		intstr.FromInt32(100),
	}
	for _, c := range successCases {
		spec := policy.PodDisruptionBudgetSpec{
			MinAvailable: &c,
		}
		errs := ValidatePodDisruptionBudgetSpec(spec, PodDisruptionBudgetValidationOptions{true}, field.NewPath("foo"))
		if len(errs) != 0 {
			t.Errorf("unexpected failure %v for %v", errs, spec)
		}
	}

	failureCases := []intstr.IntOrString{
		intstr.FromString("1.1%"),
		intstr.FromString("nope"),
		intstr.FromString("-1%"),
		intstr.FromString("101%"),
		intstr.FromInt32(-1),
	}
	for _, c := range failureCases {
		spec := policy.PodDisruptionBudgetSpec{
			MinAvailable: &c,
		}
		errs := ValidatePodDisruptionBudgetSpec(spec, PodDisruptionBudgetValidationOptions{true}, field.NewPath("foo"))
		if len(errs) == 0 {
			t.Errorf("unexpected success for %v", spec)
		}
	}
}

func TestValidateMinAvailablePodAndMaxUnavailableDisruptionBudgetSpec(t *testing.T) {
	c1 := intstr.FromString("10%")
	c2 := intstr.FromInt32(1)

	spec := policy.PodDisruptionBudgetSpec{
		MinAvailable:   &c1,
		MaxUnavailable: &c2,
	}
	errs := ValidatePodDisruptionBudgetSpec(spec, PodDisruptionBudgetValidationOptions{true}, field.NewPath("foo"))
	if len(errs) == 0 {
		t.Errorf("unexpected success for %v", spec)
	}
}

func TestValidateUnhealthyPodEvictionPolicyDisruptionBudgetSpec(t *testing.T) {
	c1 := intstr.FromString("10%")
	alwaysAllowPolicy := policy.AlwaysAllow
	invalidPolicy := policy.UnhealthyPodEvictionPolicyType("Invalid")

	testCases := []struct {
		name      string
		pdbSpec   policy.PodDisruptionBudgetSpec
		expectErr bool
	}{{
		name: "valid nil UnhealthyPodEvictionPolicy",
		pdbSpec: policy.PodDisruptionBudgetSpec{
			MinAvailable:               &c1,
			UnhealthyPodEvictionPolicy: nil,
		},
		expectErr: false,
	}, {
		name: "valid UnhealthyPodEvictionPolicy",
		pdbSpec: policy.PodDisruptionBudgetSpec{
			MinAvailable:               &c1,
			UnhealthyPodEvictionPolicy: &alwaysAllowPolicy,
		},
		expectErr: false,
	}, {
		name: "empty UnhealthyPodEvictionPolicy",
		pdbSpec: policy.PodDisruptionBudgetSpec{
			MinAvailable:               &c1,
			UnhealthyPodEvictionPolicy: new(policy.UnhealthyPodEvictionPolicyType),
		},
		expectErr: true,
	}, {
		name: "invalid UnhealthyPodEvictionPolicy",
		pdbSpec: policy.PodDisruptionBudgetSpec{
			MinAvailable:               &c1,
			UnhealthyPodEvictionPolicy: &invalidPolicy,
		},
		expectErr: true,
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := ValidatePodDisruptionBudgetSpec(tc.pdbSpec, PodDisruptionBudgetValidationOptions{true}, field.NewPath("foo"))
			if len(errs) == 0 && tc.expectErr {
				t.Errorf("unexpected success for %v", tc.pdbSpec)
			}
			if len(errs) != 0 && !tc.expectErr {
				t.Errorf("unexpected failure for %v", tc.pdbSpec)
			}
		})
	}
}

func TestValidatePodDisruptionBudgetStatus(t *testing.T) {
	const expectNoErrors = false
	const expectErrors = true
	testCases := []struct {
		name                string
		pdbStatus           policy.PodDisruptionBudgetStatus
		expectErrForVersion map[schema.GroupVersion]bool
	}{{
		name: "DisruptionsAllowed: 10",
		pdbStatus: policy.PodDisruptionBudgetStatus{
			DisruptionsAllowed: 10,
		},
		expectErrForVersion: map[schema.GroupVersion]bool{
			policy.SchemeGroupVersion:        expectNoErrors,
			policyv1beta1.SchemeGroupVersion: expectNoErrors,
		},
	}, {
		name: "CurrentHealthy: 5",
		pdbStatus: policy.PodDisruptionBudgetStatus{
			CurrentHealthy: 5,
		},
		expectErrForVersion: map[schema.GroupVersion]bool{
			policy.SchemeGroupVersion:        expectNoErrors,
			policyv1beta1.SchemeGroupVersion: expectNoErrors,
		},
	}, {
		name: "DesiredHealthy: 3",
		pdbStatus: policy.PodDisruptionBudgetStatus{
			DesiredHealthy: 3,
		},
		expectErrForVersion: map[schema.GroupVersion]bool{
			policy.SchemeGroupVersion:        expectNoErrors,
			policyv1beta1.SchemeGroupVersion: expectNoErrors,
		},
	}, {
		name: "ExpectedPods: 2",
		pdbStatus: policy.PodDisruptionBudgetStatus{
			ExpectedPods: 2,
		},
		expectErrForVersion: map[schema.GroupVersion]bool{
			policy.SchemeGroupVersion:        expectNoErrors,
			policyv1beta1.SchemeGroupVersion: expectNoErrors,
		},
	}, {
		name: "DisruptionsAllowed: -10",
		pdbStatus: policy.PodDisruptionBudgetStatus{
			DisruptionsAllowed: -10,
		},
		expectErrForVersion: map[schema.GroupVersion]bool{
			policy.SchemeGroupVersion:        expectErrors,
			policyv1beta1.SchemeGroupVersion: expectNoErrors,
		},
	}, {
		name: "CurrentHealthy: -5",
		pdbStatus: policy.PodDisruptionBudgetStatus{
			CurrentHealthy: -5,
		},
		expectErrForVersion: map[schema.GroupVersion]bool{
			policy.SchemeGroupVersion:        expectErrors,
			policyv1beta1.SchemeGroupVersion: expectNoErrors,
		},
	}, {
		name: "DesiredHealthy: -3",
		pdbStatus: policy.PodDisruptionBudgetStatus{
			DesiredHealthy: -3,
		},
		expectErrForVersion: map[schema.GroupVersion]bool{
			policy.SchemeGroupVersion:        expectErrors,
			policyv1beta1.SchemeGroupVersion: expectNoErrors,
		},
	}, {
		name: "ExpectedPods: -2",
		pdbStatus: policy.PodDisruptionBudgetStatus{
			ExpectedPods: -2,
		},
		expectErrForVersion: map[schema.GroupVersion]bool{
			policy.SchemeGroupVersion:        expectErrors,
			policyv1beta1.SchemeGroupVersion: expectNoErrors,
		},
	}, {
		name: "Conditions valid",
		pdbStatus: policy.PodDisruptionBudgetStatus{
			Conditions: []metav1.Condition{{
				Type:   policyv1beta1.DisruptionAllowedCondition,
				Status: metav1.ConditionTrue,
				LastTransitionTime: metav1.Time{
					Time: time.Now().Add(-5 * time.Minute),
				},
				Reason:             policyv1beta1.SufficientPodsReason,
				Message:            "message",
				ObservedGeneration: 3,
			}},
		},
		expectErrForVersion: map[schema.GroupVersion]bool{
			policy.SchemeGroupVersion:        expectNoErrors,
			policyv1beta1.SchemeGroupVersion: expectNoErrors,
		},
	}, {
		name: "Conditions not valid",
		pdbStatus: policy.PodDisruptionBudgetStatus{
			Conditions: []metav1.Condition{{
				Type:   policyv1beta1.DisruptionAllowedCondition,
				Status: metav1.ConditionTrue,
			}, {
				Type:   policyv1beta1.DisruptionAllowedCondition,
				Status: metav1.ConditionFalse,
			}},
		},
		expectErrForVersion: map[schema.GroupVersion]bool{
			policy.SchemeGroupVersion:        expectErrors,
			policyv1beta1.SchemeGroupVersion: expectErrors,
		},
	}}

	for _, tc := range testCases {
		for apiVersion, expectErrors := range tc.expectErrForVersion {
			t.Run(fmt.Sprintf("apiVersion: %s, %s", apiVersion.String(), tc.name), func(t *testing.T) {
				errors := ValidatePodDisruptionBudgetStatusUpdate(tc.pdbStatus, policy.PodDisruptionBudgetStatus{},
					field.NewPath("status"), apiVersion)
				errCount := len(errors)

				if errCount > 0 && !expectErrors {
					t.Errorf("unexpected failure %v for %v", errors, tc.pdbStatus)
				}

				if errCount == 0 && expectErrors {
					t.Errorf("expected errors but didn't one for %v", tc.pdbStatus)
				}
			})
		}
	}
}

func TestIsValidSysctlPattern(t *testing.T) {
	valid := []string{
		"a.b.c.d",
		"a",
		"a_b",
		"a-b",
		"abc",
		"abc.def",
		"*",
		"a.*",
		"*",
		"abc*",
		"a.abc*",
		"a.b.*",
		"a/b/c/d",
		"a/*",
		"a/b/*",
		"a.b/c*",
		"a.b/c.d",
		"a/b.c/d",
	}
	invalid := []string{
		"",
		"ä",
		"a_",
		"_",
		"_a",
		"_a._b",
		"__",
		"-",
		".",
		"a.",
		".a",
		"a.b.",
		"a*.b",
		"a*b",
		"*a",
		"Abc",
		"/",
		"a/",
		"/a",
		"a*/b",
		func(n int) string {
			x := make([]byte, n)
			for i := range x {
				x[i] = byte('a')
			}
			return string(x)
		}(256),
	}
	for _, s := range valid {
		if !IsValidSysctlPattern(s) {
			t.Errorf("%q expected to be a valid sysctl pattern", s)
		}
	}
	for _, s := range invalid {
		if IsValidSysctlPattern(s) {
			t.Errorf("%q expected to be an invalid sysctl pattern", s)
		}
	}
}
