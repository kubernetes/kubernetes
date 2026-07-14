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
	"k8s.io/apimachinery/pkg/api/operation"

	"context"

	"testing"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestBuilderValidate(t *testing.T) {
	allPolicies := []SchedulingPolicyOption{BasicPolicy, GangPolicy}
	allModes := []DisruptionModeOption{SingleMode, AllMode}
	gang := func() PolicyInput {
		return PolicyInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: ptr.To[int32](2)}}}
	}
	basic := func() PolicyInput { return PolicyInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{}}} }

	tests := []struct {
		name          string
		root          *WorkloadItem
		allowPolicies []SchedulingPolicyOption
		allowModes    []DisruptionModeOption
		wantErrs      int
		wantType      field.ErrorType
	}{
		{
			name:          "nil tree fails to compile",
			root:          nil,
			allowPolicies: allPolicies,
			allowModes:    allModes,
			wantErrs:      1,
			wantType:      field.ErrorTypeInvalid,
		},
		{
			name:          "basic allowed",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: basic()}},
			allowPolicies: allPolicies,
			allowModes:    allModes,
		},
		{
			name:          "basic forbidden when not allow-listed",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: basic()}},
			allowPolicies: []SchedulingPolicyOption{GangPolicy},
			allowModes:    allModes,
			wantErrs:      1,
			wantType:      field.ErrorTypeForbidden,
		},
		{
			name:          "gang allowed",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang()}},
			allowPolicies: []SchedulingPolicyOption{GangPolicy},
			allowModes:    allModes,
		},
		{
			name:          "gang forbidden when not allow-listed",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang()}},
			allowPolicies: []SchedulingPolicyOption{BasicPolicy},
			allowModes:    allModes,
			wantErrs:      1,
			wantType:      field.ErrorTypeForbidden,
		},
		{
			name:          "single disruption allowed",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang(), DisruptionMode: DisruptionModeInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{Single: &schedulingv1alpha3.WorkloadPodGroupSingleDisruptionMode{}}}}},
			allowPolicies: allPolicies,
			allowModes:    []DisruptionModeOption{SingleMode},
		},
		{
			name:          "all disruption forbidden when not allow-listed",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang(), DisruptionMode: DisruptionModeInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{}}}}},
			allowPolicies: allPolicies,
			allowModes:    []DisruptionModeOption{SingleMode},
			wantErrs:      1,
			wantType:      field.ErrorTypeForbidden,
		},
		{
			name:          "basic with all disruption is rejected by cross-field rule",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: basic(), DisruptionMode: DisruptionModeInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{}}}}},
			allowPolicies: allPolicies,
			allowModes:    allModes,
			wantErrs:      1,
			wantType:      field.ErrorTypeInvalid,
		},
		{
			name:          "gang with all disruption is valid",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang(), DisruptionMode: DisruptionModeInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{}}}}},
			allowPolicies: allPolicies,
			allowModes:    allModes,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := NewBuilder(tt.root, BuildOptions{
				Owner:                  jobOwner(),
				AllowedPolicies:        tt.allowPolicies,
				AllowedDisruptionModes: tt.allowModes,
			})
			errs := b.Validate(context.Background(), operation.Operation{Type: operation.Create}, nil, nil)
			if len(errs) != tt.wantErrs {
				t.Fatalf("expected %d error(s), got %d: %v", tt.wantErrs, len(errs), errs)
			}
			if tt.wantErrs > 0 && tt.wantType != "" && errs[0].Type != tt.wantType {
				t.Errorf("expected error type %s, got %s", tt.wantType, errs[0].Type)
			}
		})
	}
}
