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

package workloadbuilder

import (
	"testing"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestValidateSchedulingPolicy(t *testing.T) {
	tests := []struct {
		name     string
		policy   *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy
		allowed  []SchedulingPolicyOption
		wantErrs int
		wantType field.ErrorType
	}{
		{
			name:     "nil policy is always valid",
			policy:   nil,
			allowed:  []SchedulingPolicyOption{BasicPolicy, GangPolicy},
			wantErrs: 0,
		},
		{
			name:     "basic allowed",
			policy:   &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{}},
			allowed:  []SchedulingPolicyOption{BasicPolicy, GangPolicy},
			wantErrs: 0,
		},
		{
			name:     "basic forbidden when not allow-listed",
			policy:   &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{}},
			allowed:  []SchedulingPolicyOption{GangPolicy},
			wantErrs: 1,
			wantType: field.ErrorTypeForbidden,
		},
		{
			name:     "gang allowed",
			policy:   &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: ptr.To[int32](4)}},
			allowed:  []SchedulingPolicyOption{GangPolicy},
			wantErrs: 0,
		},
		{
			name:     "gang forbidden when not allow-listed",
			policy:   &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{}},
			allowed:  []SchedulingPolicyOption{BasicPolicy},
			wantErrs: 1,
			wantType: field.ErrorTypeForbidden,
		},
		{
			name:     "gang with invalid minCount",
			policy:   &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: ptr.To[int32](0)}},
			allowed:  []SchedulingPolicyOption{GangPolicy},
			wantErrs: 1,
			wantType: field.ErrorTypeInvalid,
		},
		{
			name:     "no policy set",
			policy:   &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{},
			allowed:  []SchedulingPolicyOption{BasicPolicy, GangPolicy},
			wantErrs: 1,
			wantType: field.ErrorTypeRequired,
		},
		{
			name: "both policies set",
			policy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
				Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{},
				Gang:  &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: ptr.To[int32](1)},
			},
			allowed:  []SchedulingPolicyOption{BasicPolicy, GangPolicy},
			wantErrs: 1,
			wantType: field.ErrorTypeInvalid,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := ValidateSchedulingPolicy(tt.policy, field.NewPath("policy"), tt.allowed...)
			if len(errs) != tt.wantErrs {
				t.Fatalf("expected %d error(s), got %d: %v", tt.wantErrs, len(errs), errs)
			}
			if tt.wantErrs > 0 && tt.wantType != "" && errs[0].Type != tt.wantType {
				t.Errorf("expected error type %s, got %s", tt.wantType, errs[0].Type)
			}
		})
	}
}

func TestValidateDisruptionMode(t *testing.T) {
	tests := []struct {
		name     string
		mode     *schedulingv1alpha3.WorkloadPodGroupDisruptionMode
		allowed  []DisruptionModeOption
		wantErrs int
		wantType field.ErrorType
	}{
		{
			name:     "nil mode is always valid",
			mode:     nil,
			allowed:  []DisruptionModeOption{SingleMode, AllMode},
			wantErrs: 0,
		},
		{
			name:     "single allowed",
			mode:     &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{Single: &schedulingv1alpha3.WorkloadPodGroupSingleDisruptionMode{}},
			allowed:  []DisruptionModeOption{SingleMode},
			wantErrs: 0,
		},
		{
			name:     "all forbidden when not allow-listed",
			mode:     &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{}},
			allowed:  []DisruptionModeOption{SingleMode},
			wantErrs: 1,
			wantType: field.ErrorTypeForbidden,
		},
		{
			name:     "no mode set",
			mode:     &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{},
			allowed:  []DisruptionModeOption{SingleMode, AllMode},
			wantErrs: 1,
			wantType: field.ErrorTypeRequired,
		},
		{
			name: "both modes set",
			mode: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{
				Single: &schedulingv1alpha3.WorkloadPodGroupSingleDisruptionMode{},
				All:    &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{},
			},
			allowed:  []DisruptionModeOption{SingleMode, AllMode},
			wantErrs: 1,
			wantType: field.ErrorTypeInvalid,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := ValidateDisruptionMode(tt.mode, field.NewPath("disruption"), tt.allowed...)
			if len(errs) != tt.wantErrs {
				t.Fatalf("expected %d error(s), got %d: %v", tt.wantErrs, len(errs), errs)
			}
			if tt.wantErrs > 0 && tt.wantType != "" && errs[0].Type != tt.wantType {
				t.Errorf("expected error type %s, got %s", tt.wantType, errs[0].Type)
			}
		})
	}
}
