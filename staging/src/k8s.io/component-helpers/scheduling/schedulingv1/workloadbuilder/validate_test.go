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
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

func TestBuilderValidate(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	allPolicies := []SchedulingPolicyOption{BasicPolicy, GangPolicy}
	allModes := []DisruptionModeOption{SingleMode, AllMode}
	gang := func() PolicyInput {
		return PolicyInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: ptr.To[int32](2)}}}
	}
	basic := func() PolicyInput {
		return PolicyInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{}}}
	}

	tests := []struct {
		name             string
		root             *WorkloadItem
		oldRoot          *WorkloadItem
		allowPolicies    []SchedulingPolicyOption
		allowModes       []DisruptionModeOption
		disableDV        bool
		wantErrs         int
		wantType         field.ErrorType
		wantErrFieldPath string
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
		{
			name: "gang minCount below minimum fails declarative validation at injected path",
			root: &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: PolicyInput{
				PathElements: []string{"schedulingPolicy"},
				PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: ptr.To[int32](0)}},
			}}},
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.scheduling.schedulingPolicy.gang.minCount",
		},
		{
			name: "declarative validation is skipped when disabled",
			root: &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: PolicyInput{
				PathElements: []string{"schedulingPolicy"},
				PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: ptr.To[int32](0)}},
			}}},
			allowPolicies: allPolicies,
			allowModes:    allModes,
			disableDV:     true,
		},
		{
			name: "disruptionMode with single and all fails union declarative validation",
			root: &WorkloadItem{Name: "job", Input: WorkloadInput{
				Policy: gang(),
				DisruptionMode: DisruptionModeInput{
					PathElements: []string{"disruptionMode"},
					PodGroupData: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{
						Single: &schedulingv1alpha3.WorkloadPodGroupSingleDisruptionMode{},
						All:    &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{},
					},
				},
			}},
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.scheduling.disruptionMode",
		},
		{
			name: "too many topology constraints fails declarative validation",
			root: &WorkloadItem{Name: "job", Input: WorkloadInput{
				Policy: gang(),
				Constraints: ConstraintsInput{
					PathElements: []string{"schedulingConstraints"},
					PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
						Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "zone"}, {Key: "rack"}},
					},
				},
			}},
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeTooMany,
			wantErrFieldPath: "spec.scheduling.schedulingConstraints.topology",
		},
		{
			name: "resourceClaim with no source fails union declarative validation at indexed path",
			root: &WorkloadItem{Name: "job", Input: WorkloadInput{
				Policy: gang(),
				ResourceClaims: ResourceClaimsInput{
					PathElements: []string{"resourceClaims"},
					PodGroupData: []schedulingv1alpha3.WorkloadPodGroupResourceClaim{{Name: "claim0"}},
				},
			}},
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.scheduling.resourceClaims[0]",
		},
		{
			name:          "update with matching root name is accepted",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang()}},
			oldRoot:       &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang()}},
			allowPolicies: allPolicies,
			allowModes:    allModes,
		},
		{
			name:             "update with mismatched root name is an internal error",
			root:             &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang()}},
			oldRoot:          &WorkloadItem{Name: "stale-job", Input: WorkloadInput{Policy: gang()}},
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInternal,
			wantErrFieldPath: "spec.scheduling",
		},
		{
			name:          "update clearing policy skips declarative validation for the new nil policy",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{}},
			oldRoot:       &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang()}},
			allowPolicies: allPolicies,
			allowModes:    allModes,
		},
		{
			name: "update clearing constraints skips declarative validation for the new nil constraints",
			root: &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang()}},
			oldRoot: &WorkloadItem{Name: "job", Input: WorkloadInput{
				Policy: gang(),
				Constraints: ConstraintsInput{
					PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
						Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "zone"}},
					},
				},
			}},
			allowPolicies: allPolicies,
			allowModes:    allModes,
		},
		{
			name: "update clearing disruptionMode skips declarative validation for the new nil disruptionMode",
			root: &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang()}},
			oldRoot: &WorkloadItem{Name: "job", Input: WorkloadInput{
				Policy: gang(),
				DisruptionMode: DisruptionModeInput{
					PodGroupData: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{
						Single: &schedulingv1alpha3.WorkloadPodGroupSingleDisruptionMode{},
					},
				},
			}},
			allowPolicies: allPolicies,
			allowModes:    allModes,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := NewBuilder(tt.root, BuildOptions{
				Owner:                        jobOwner(),
				AllowedPolicies:              tt.allowPolicies,
				AllowedDisruptionModes:       tt.allowModes,
				DisableDeclarativeValidation: tt.disableDV,
			})
			errs := b.Validate(ctx, field.NewPath("spec", "scheduling"), ValidationInput{OldRoot: tt.oldRoot})
			if len(errs) != tt.wantErrs {
				t.Fatalf("expected %d error(s), got %d: %v", tt.wantErrs, len(errs), errs)
			}
			if tt.wantErrs > 0 && tt.wantType != "" && errs[0].Type != tt.wantType {
				t.Errorf("expected error type %s, got %s", tt.wantType, errs[0].Type)
			}
			if tt.wantErrFieldPath != "" && errs[0].Field != tt.wantErrFieldPath {
				t.Errorf("expected error field %q, got %q", tt.wantErrFieldPath, errs[0].Field)
			}
		})
	}
}
