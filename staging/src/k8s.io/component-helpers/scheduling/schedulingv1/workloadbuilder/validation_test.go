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
	"context"
	"testing"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidateSchedulingPolicy(t *testing.T) {
	basic := &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{}
	gang := func(minCount int32) *schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy {
		return &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: new(minCount)}
	}

	tests := []struct {
		name      string
		policy    *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy
		oldPolicy *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy
		update    bool
		allowed   []SchedulingPolicyOption
		wantErrs  int
		wantType  field.ErrorType
	}{
		{
			name:     "nil policy is always valid",
			policy:   nil,
			allowed:  []SchedulingPolicyOption{BasicPolicy, GangPolicy},
			wantErrs: 0,
		},
		{
			name:     "basic allowed",
			policy:   &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Basic: basic},
			allowed:  []SchedulingPolicyOption{BasicPolicy, GangPolicy},
			wantErrs: 0,
		},
		{
			name:     "basic forbidden when not allow-listed",
			policy:   &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Basic: basic},
			allowed:  []SchedulingPolicyOption{GangPolicy},
			wantErrs: 1,
			wantType: field.ErrorTypeForbidden,
		},
		{
			name:     "gang allowed",
			policy:   &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: gang(4)},
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
			name:     "empty policy fails union validation",
			policy:   &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{},
			allowed:  []SchedulingPolicyOption{BasicPolicy, GangPolicy},
			wantErrs: 1,
			wantType: field.ErrorTypeInvalid,
		},
		{
			name: "both policies set flags the non-allow-listed one and the union violation",
			policy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
				Basic: basic,
				Gang:  gang(1),
			},
			allowed:  []SchedulingPolicyOption{BasicPolicy},
			wantErrs: 2,
			wantType: field.ErrorTypeForbidden,
		},
		{
			name:     "gang minCount below minimum",
			policy:   &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: gang(0)},
			allowed:  []SchedulingPolicyOption{GangPolicy},
			wantErrs: 1,
			wantType: field.ErrorTypeInvalid,
		},
		{
			name:      "update with unchanged gang policy is ratcheted",
			policy:    &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: gang(2)},
			oldPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: gang(2)},
			update:    true,
			allowed:   []SchedulingPolicyOption{GangPolicy},
			wantErrs:  0,
		},
		{
			name:      "update cannot switch basic to gang",
			policy:    &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: gang(2)},
			oldPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Basic: basic},
			update:    true,
			allowed:   []SchedulingPolicyOption{BasicPolicy, GangPolicy},
			wantErrs:  2, // basic is immutable, gang cannot be set once created
			wantType:  field.ErrorTypeInvalid,
		},
		{
			name:      "update cannot switch gang to basic",
			policy:    &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Basic: basic},
			oldPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: gang(2)},
			update:    true,
			allowed:   []SchedulingPolicyOption{BasicPolicy, GangPolicy},
			wantErrs:  2, // basic is immutable, gang cannot be cleared once set
			wantType:  field.ErrorTypeInvalid,
		},
		{
			name:      "update revalidates a changed minCount",
			policy:    &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: gang(0)},
			oldPolicy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: gang(2)},
			update:    true,
			allowed:   []SchedulingPolicyOption{GangPolicy},
			wantErrs:  1,
			wantType:  field.ErrorTypeInvalid,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			var errs field.ErrorList
			if tt.update {
				errs = ValidateSchedulingPolicyUpdate(ctx, tt.policy, tt.oldPolicy, field.NewPath("policy"), tt.allowed...)
			} else {
				errs = ValidateSchedulingPolicy(ctx, tt.policy, field.NewPath("policy"), tt.allowed...)
			}
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
	single := &schedulingv1alpha3.WorkloadPodGroupSingleDisruptionMode{}
	all := &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{}

	tests := []struct {
		name     string
		mode     *schedulingv1alpha3.WorkloadPodGroupDisruptionMode
		oldMode  *schedulingv1alpha3.WorkloadPodGroupDisruptionMode
		update   bool
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
			mode:     &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{Single: single},
			allowed:  []DisruptionModeOption{SingleMode},
			wantErrs: 0,
		},
		{
			name:     "all forbidden when not allow-listed",
			mode:     &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{All: all},
			allowed:  []DisruptionModeOption{SingleMode},
			wantErrs: 1,
			wantType: field.ErrorTypeForbidden,
		},
		{
			name:     "empty mode fails union validation",
			mode:     &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{},
			allowed:  []DisruptionModeOption{SingleMode, AllMode},
			wantErrs: 1,
			wantType: field.ErrorTypeInvalid,
		},
		{
			name: "both modes set flags the non-allow-listed one and the union violation",
			mode: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{
				Single: single,
				All:    all,
			},
			allowed:  []DisruptionModeOption{AllMode},
			wantErrs: 2,
			wantType: field.ErrorTypeForbidden,
		},
		{
			name:     "update can switch single to all",
			mode:     &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{All: all},
			oldMode:  &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{Single: single},
			update:   true,
			allowed:  []DisruptionModeOption{SingleMode, AllMode},
			wantErrs: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			var errs field.ErrorList
			if tt.update {
				errs = ValidateDisruptionModeUpdate(ctx, tt.mode, tt.oldMode, field.NewPath("disruption"), tt.allowed...)
			} else {
				errs = ValidateDisruptionMode(ctx, tt.mode, field.NewPath("disruption"), tt.allowed...)
			}
			if len(errs) != tt.wantErrs {
				t.Fatalf("expected %d error(s), got %d: %v", tt.wantErrs, len(errs), errs)
			}
			if tt.wantErrs > 0 && tt.wantType != "" && errs[0].Type != tt.wantType {
				t.Errorf("expected error type %s, got %s", tt.wantType, errs[0].Type)
			}
		})
	}
}
