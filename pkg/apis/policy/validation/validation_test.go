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
	"testing"

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
	errs := ValidatePodDisruptionBudgetSpec(spec, field.NewPath("foo"))
	if len(errs) == 0 {
		t.Errorf("unexpected success for %v", spec)
	}
}

func TestValidateMinAvailablePodDisruptionBudgetSpec(t *testing.T) {
	successCases := []intstr.IntOrString{
		intstr.FromString("0%"),
		intstr.FromString("1%"),
		intstr.FromString("100%"),
		intstr.FromInt(0),
		intstr.FromInt(1),
		intstr.FromInt(100),
	}
	for _, c := range successCases {
		spec := policy.PodDisruptionBudgetSpec{
			MinAvailable: &c,
		}
		errs := ValidatePodDisruptionBudgetSpec(spec, field.NewPath("foo"))
		if len(errs) != 0 {
			t.Errorf("unexpected failure %v for %v", errs, spec)
		}
	}

	failureCases := []intstr.IntOrString{
		intstr.FromString("1.1%"),
		intstr.FromString("nope"),
		intstr.FromString("-1%"),
		intstr.FromString("101%"),
		intstr.FromInt(-1),
	}
	for _, c := range failureCases {
		spec := policy.PodDisruptionBudgetSpec{
			MinAvailable: &c,
		}
		errs := ValidatePodDisruptionBudgetSpec(spec, field.NewPath("foo"))
		if len(errs) == 0 {
			t.Errorf("unexpected success for %v", spec)
		}
	}
}

func TestValidateMinAvailablePodAndMaxUnavailableDisruptionBudgetSpec(t *testing.T) {
	c1 := intstr.FromString("10%")
	c2 := intstr.FromInt(1)

	spec := policy.PodDisruptionBudgetSpec{
		MinAvailable:   &c1,
		MaxUnavailable: &c2,
	}
	errs := ValidatePodDisruptionBudgetSpec(spec, field.NewPath("foo"))
	if len(errs) == 0 {
		t.Errorf("unexpected success for %v", spec)
	}
}

func TestValidatePodDisruptionBudgetStatus(t *testing.T) {
	successCases := []policy.PodDisruptionBudgetStatus{
		{PodDisruptionsAllowed: 10},
		{CurrentHealthy: 5},
		{DesiredHealthy: 3},
		{ExpectedPods: 2}}
	for _, c := range successCases {
		errors := ValidatePodDisruptionBudgetStatus(c, field.NewPath("status"))
		if len(errors) > 0 {
			t.Errorf("unexpected failure %v for %v", errors, c)
		}
	}
	failureCases := []policy.PodDisruptionBudgetStatus{
		{PodDisruptionsAllowed: -10},
		{CurrentHealthy: -5},
		{DesiredHealthy: -3},
		{ExpectedPods: -2}}
	for _, c := range failureCases {
		errors := ValidatePodDisruptionBudgetStatus(c, field.NewPath("status"))
		if len(errors) == 0 {
			t.Errorf("unexpected success for %v", c)
		}
	}
}

func TestValidatePodDisruptionBudgetUpdate(t *testing.T) {
	c1 := intstr.FromString("10%")
	c2 := intstr.FromInt(1)
	c3 := intstr.FromInt(2)
	oldPdb := &policy.PodDisruptionBudget{}
	pdb := &policy.PodDisruptionBudget{}
	testCases := []struct {
		generations []int64
		name        string
		specs       []policy.PodDisruptionBudgetSpec
		status      []policy.PodDisruptionBudgetStatus
		ok          bool
	}{
		{
			name:        "only update status",
			generations: []int64{int64(2), int64(3)},
			specs: []policy.PodDisruptionBudgetSpec{
				{
					MinAvailable:   &c1,
					MaxUnavailable: &c2,
				},
				{
					MinAvailable:   &c1,
					MaxUnavailable: &c2,
				},
			},
			status: []policy.PodDisruptionBudgetStatus{
				{
					PodDisruptionsAllowed: 10,
					CurrentHealthy:        5,
					ExpectedPods:          2,
				},
				{
					PodDisruptionsAllowed: 8,
					CurrentHealthy:        5,
					DesiredHealthy:        3,
				},
			},
			ok: true,
		},
		{
			name:        "only update pdb spec",
			generations: []int64{int64(2), int64(3)},
			specs: []policy.PodDisruptionBudgetSpec{
				{
					MaxUnavailable: &c2,
				},
				{
					MinAvailable:   &c1,
					MaxUnavailable: &c3,
				},
			},
			status: []policy.PodDisruptionBudgetStatus{
				{
					PodDisruptionsAllowed: 10,
				},
				{
					PodDisruptionsAllowed: 10,
				},
			},
			ok: false,
		},
		{
			name:        "update spec and status",
			generations: []int64{int64(2), int64(3)},
			specs: []policy.PodDisruptionBudgetSpec{
				{
					MaxUnavailable: &c2,
				},
				{
					MinAvailable:   &c1,
					MaxUnavailable: &c3,
				},
			},
			status: []policy.PodDisruptionBudgetStatus{
				{
					PodDisruptionsAllowed: 10,
					CurrentHealthy:        5,
					ExpectedPods:          2,
				},
				{
					PodDisruptionsAllowed: 8,
					CurrentHealthy:        5,
					DesiredHealthy:        3,
				},
			},
			ok: false,
		},
	}

	for i, tc := range testCases {
		oldPdb.Spec = tc.specs[0]
		oldPdb.Generation = tc.generations[0]
		oldPdb.Status = tc.status[0]

		pdb.Spec = tc.specs[1]
		pdb.Generation = tc.generations[1]
		oldPdb.Status = tc.status[1]

		errs := ValidatePodDisruptionBudgetUpdate(oldPdb, pdb)
		if tc.ok && len(errs) > 0 {
			t.Errorf("[%d:%s] unexpected errors: %v", i, tc.name, errs)
		} else if !tc.ok && len(errs) == 0 {
			t.Errorf("[%d:%s] expected errors: %v", i, tc.name, errs)
		}
	}
}
