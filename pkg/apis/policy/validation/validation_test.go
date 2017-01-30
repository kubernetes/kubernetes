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

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func TestValidatePodDisruptionBudgetSpec(t *testing.T) {
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
			MinAvailable: c,
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
			MinAvailable: c,
		}
		errs := ValidatePodDisruptionBudgetSpec(spec, field.NewPath("foo"))
		if len(errs) == 0 {
			t.Errorf("unexpected success for %v", spec)
		}
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
