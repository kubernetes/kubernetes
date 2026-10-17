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
)

var (
	schedulingPath = field.NewPath("spec", "scheduling")
)

func TestBuilderValidate(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	allPolicies := []SchedulingPolicyOption{BasicPolicy, GangPolicy}
	allModes := []DisruptionModeOption{SingleMode, AllMode}
	gang := func() PolicyInput {
		return PolicyInput{PodGroupData: gangData(2)}
	}
	basic := func() PolicyInput {
		return PolicyInput{PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{}}}
	}
	compositeGang := func() PolicyInput {
		return PolicyInput{CompositePodGroupData: compositeGangData(2)}
	}
	compositeBasic := func() PolicyInput {
		return PolicyInput{CompositePodGroupData: &schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.WorkloadCompositePodGroupBasicSchedulingPolicy{}}}
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
			name:             "empty item name is rejected",
			root:             &WorkloadItem{Name: "", Path: schedulingPath, Input: WorkloadInput{Policy: basic()}},
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInternal,
			wantErrFieldPath: "spec.scheduling",
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
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang(), DisruptionMode: singleDisruption()}},
			allowPolicies: allPolicies,
			allowModes:    []DisruptionModeOption{SingleMode},
		},
		{
			name:          "all disruption forbidden when not allow-listed",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang(), DisruptionMode: allDisruption()}},
			allowPolicies: allPolicies,
			allowModes:    []DisruptionModeOption{SingleMode},
			wantErrs:      1,
			wantType:      field.ErrorTypeForbidden,
		},
		{
			name:          "basic with all disruption is rejected by cross-field rule",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: basic(), DisruptionMode: allDisruption()}},
			allowPolicies: allPolicies,
			allowModes:    allModes,
			wantErrs:      1,
			wantType:      field.ErrorTypeInvalid,
		},
		{
			name:          "gang with all disruption is valid",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang(), DisruptionMode: allDisruption()}},
			allowPolicies: allPolicies,
			allowModes:    allModes,
		},
		{
			name:             "gang minCount below minimum fails declarative validation at injected path",
			root:             newWorkloadItem("job", withPath(schedulingPath), withGang(0)),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.scheduling.schedulingPolicy.gang.minCount",
		},
		{
			name:             "nil item Path reports building-block errors without a root prefix",
			root:             newWorkloadItem("job", withGang(0)),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "schedulingPolicy.gang.minCount",
		},
		{
			name:          "declarative validation is skipped when disabled",
			root:          newWorkloadItem("job", withPath(schedulingPath), withGang(0)),
			allowPolicies: allPolicies,
			allowModes:    allModes,
			disableDV:     true,
		},
		{
			name: "disruptionMode with single and all fails union declarative validation",
			root: &WorkloadItem{Name: "job", Path: schedulingPath, Input: WorkloadInput{
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
			root: &WorkloadItem{Name: "job", Path: schedulingPath, Input: WorkloadInput{
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
			root: &WorkloadItem{Name: "job", Path: schedulingPath, Input: WorkloadInput{
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
			name:          "composite gang allowed",
			root:          compositeRoot(WorkloadInput{Policy: compositeGang()}),
			allowPolicies: []SchedulingPolicyOption{GangPolicy},
			allowModes:    allModes,
		},
		{
			name:          "composite basic forbidden when not allow-listed",
			root:          compositeRoot(WorkloadInput{Policy: compositeBasic()}),
			allowPolicies: []SchedulingPolicyOption{GangPolicy},
			allowModes:    allModes,
			wantErrs:      1,
			wantType:      field.ErrorTypeForbidden,
		},
		{
			name: "composite gang minGroupCount below minimum fails declarative validation at injected path",
			root: compositeRoot(WorkloadInput{Policy: PolicyInput{
				PathElements:          []string{"schedulingPolicy"},
				CompositePodGroupData: compositeGangData(0),
			}}),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.scheduling.schedulingPolicy.gang.minGroupCount",
		},
		{
			name: "composite policy with basic and gang fails union declarative validation",
			root: compositeRoot(WorkloadInput{Policy: PolicyInput{
				PathElements: []string{"schedulingPolicy"},
				CompositePodGroupData: &schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy{
					Basic: &schedulingv1alpha3.WorkloadCompositePodGroupBasicSchedulingPolicy{},
					Gang:  &schedulingv1alpha3.WorkloadCompositePodGroupGangSchedulingPolicy{MinGroupCount: new(int32(2))},
				},
			}}),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.scheduling.schedulingPolicy",
		},
		{
			name: "composite policy set with a leaf policy fails the input union",
			root: compositeRoot(WorkloadInput{Policy: PolicyInput{
				PathElements:          []string{"schedulingPolicy"},
				CompositePodGroupData: compositeGangData(2),
				PodGroupData:          gangData(2),
			}}),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInternal,
			wantErrFieldPath: "spec.scheduling",
		},
		{
			name: "composite policy set with a leaf disruptionMode fails the input union",
			root: compositeRoot(WorkloadInput{
				Policy: PolicyInput{
					PathElements:          []string{"schedulingPolicy"},
					CompositePodGroupData: compositeGangData(2),
				},
				DisruptionMode: singleDisruption(),
			}),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInternal,
			wantErrFieldPath: "spec.scheduling",
		},
		{
			name: "input union is enforced even when declarative validation is disabled",
			root: compositeRoot(WorkloadInput{Policy: PolicyInput{
				PathElements:          []string{"schedulingPolicy"},
				CompositePodGroupData: compositeGangData(2),
				PodGroupData:          gangData(2),
			}}),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			disableDV:        true,
			wantErrs:         1,
			wantType:         field.ErrorTypeInternal,
			wantErrFieldPath: "spec.scheduling",
		},
		{
			name: "invalid leaf child in a composite tree is validated recursively",
			root: newWorkloadItem("root", withPath(schedulingPath), withInput(WorkloadInput{Policy: compositeBasic()}),
				withChildren(newWorkloadItem("child", withPath(schedulingPath), withGang(0)))),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.scheduling.schedulingPolicy.gang.minCount",
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
			root:             &WorkloadItem{Name: "job", Path: schedulingPath, Input: WorkloadInput{Policy: gang()}},
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
				Policy:         gang(),
				DisruptionMode: singleDisruption(),
			}},
			allowPolicies: allPolicies,
			allowModes:    allModes,
		},
		{
			// A child that keeps its name and position correlates with its old
			// counterpart, so its unchanged (and still invalid) minCount is
			// ratcheted and not re-validated on update.
			name:          "update of an unchanged child at the same position is ratcheted",
			root:          newWorkloadItem("root", withChildren(newWorkloadItem("leaf", withGang(0)))),
			oldRoot:       newWorkloadItem("root", withChildren(newWorkloadItem("leaf", withGang(0)))),
			allowPolicies: allPolicies,
			allowModes:    allModes,
		},
		{
			// The same-named "leaf" moves up a level (root>mid>leaf becomes
			// root>leaf), so it does not correlate with any old child at the root
			// level. It is validated as an addition, surfacing its invalid minCount.
			name:             "update of a child that changed position is validated as an addition",
			root:             newWorkloadItem("root", withPath(schedulingPath), withChildren(newWorkloadItem("leaf", withPath(schedulingPath), withGang(0)))),
			oldRoot:          newWorkloadItem("root", withChildren(newWorkloadItem("mid", withChildren(newWorkloadItem("leaf", withGang(0)))))),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.scheduling.schedulingPolicy.gang.minCount",
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
			errs := b.Validate(ctx, ValidationInput{OldRoot: tt.oldRoot})
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

type itemOption func(*WorkloadItem)

func newWorkloadItem(name string, opts ...itemOption) *WorkloadItem {
	item := &WorkloadItem{Name: name}
	for _, opt := range opts {
		opt(item)
	}
	return item
}

func withPath(path *field.Path) itemOption {
	return func(i *WorkloadItem) {
		i.Path = path
	}
}

func withInput(in WorkloadInput) itemOption {
	return func(i *WorkloadItem) {
		i.Input = in
	}
}

func withGang(minCount int32) itemOption {
	return func(i *WorkloadItem) {
		i.Input.Policy = PolicyInput{
			PathElements: []string{"schedulingPolicy"},
			PodGroupData: gangData(minCount),
		}
	}
}

func withChildren(children ...*WorkloadItem) itemOption {
	return func(i *WorkloadItem) {
		i.Children = children
	}
}

func compositeRoot(in WorkloadInput) *WorkloadItem {
	return newWorkloadItem("root", withPath(schedulingPath), withInput(in),
		withChildren(newWorkloadItem("child", withPath(schedulingPath), withGang(2))))
}

func singleDisruption() DisruptionModeInput {
	return DisruptionModeInput{
		PodGroupData: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{
			Single: &schedulingv1alpha3.WorkloadPodGroupSingleDisruptionMode{},
		},
	}
}

func allDisruption() DisruptionModeInput {
	return DisruptionModeInput{
		PodGroupData: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{
			All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{},
		},
	}
}

func gangData(minCount int32) *schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy {
	return &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
		Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{
			MinCount: new(minCount),
		},
	}
}

func compositeGangData(minGroupCount int32) *schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy {
	return &schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy{
		Gang: &schedulingv1alpha3.WorkloadCompositePodGroupGangSchedulingPolicy{
			MinGroupCount: new(minGroupCount),
		},
	}
}
