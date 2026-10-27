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
	"strings"
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
		wantErrDetail    string
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
			wantErrDetail:    "workload item name cannot be empty",
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
			name:          "unset policy (defaulting to basic) forbidden when basic not allow-listed",
			root:          &WorkloadItem{Name: "job"},
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
			name:          "single disruption forbidden when not allow-listed",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang(), DisruptionMode: singleDisruption()}},
			allowPolicies: allPolicies,
			allowModes:    []DisruptionModeOption{AllMode},
			wantErrs:      1,
			wantType:      field.ErrorTypeForbidden,
		},
		{
			name:          "unset disruption (defaulting to single) forbidden when single not allow-listed",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{Policy: gang()}},
			allowPolicies: allPolicies,
			allowModes:    []DisruptionModeOption{AllMode},
			wantErrs:      1,
			wantType:      field.ErrorTypeForbidden,
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
			name:          "unset policy (defaulting to basic) with all disruption is rejected by cross-field rule",
			root:          &WorkloadItem{Name: "job", Input: WorkloadInput{DisruptionMode: allDisruption()}},
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
			wantErrDetail:    "sets leaf building block: policy",
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
			wantErrDetail:    "sets leaf building block: disruptionMode",
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
			wantErrDetail:    "sets leaf building block: policy",
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
			wantErrDetail:    "does not match new root name",
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
		{
			name: "tree at max allowed depth is accepted",
			root: newWorkloadItem("cpg1", withPath(schedulingPath), withCompositeGang(1),
				withChildren(newWorkloadItem("cpg2", withPath(schedulingPath), withCompositeGang(1),
					withChildren(newWorkloadItem("cpg3", withPath(schedulingPath), withCompositeGang(1),
						withChildren(newWorkloadItem("leaf", withPath(schedulingPath), withGang(2)))))))),
			allowPolicies: allPolicies,
			allowModes:    allModes,
		},
		{
			name: "tree exceeding max allowed depth is rejected",
			root: newWorkloadItem("cpg1", withPath(schedulingPath), withCompositeGang(1),
				withChildren(newWorkloadItem("cpg2", withPath(schedulingPath), withCompositeGang(1),
					withChildren(newWorkloadItem("cpg3", withPath(schedulingPath), withCompositeGang(1),
						withChildren(newWorkloadItem("cpg4", withPath(schedulingPath), withCompositeGang(1),
							withChildren(newWorkloadItem("leaf", withPath(schedulingPath), withGang(2)))))))))),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInternal,
			wantErrFieldPath: "spec.scheduling",
			wantErrDetail:    "exceeds maximum allowed tree depth",
		},
		{
			name: "composite item exceeding max allowed leaf children is rejected",
			root: newWorkloadItem("root", withPath(schedulingPath), withInput(WorkloadInput{Policy: compositeBasic()}),
				withChildren(
					newWorkloadItem("leaf-1", withBasic()),
					newWorkloadItem("leaf-2", withBasic()),
					newWorkloadItem("leaf-3", withBasic()),
					newWorkloadItem("leaf-4", withBasic()),
					newWorkloadItem("leaf-5", withBasic()),
					newWorkloadItem("leaf-6", withBasic()),
					newWorkloadItem("leaf-7", withBasic()),
					newWorkloadItem("leaf-8", withBasic()),
					newWorkloadItem("leaf-9", withBasic()),
				)),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInternal,
			wantErrFieldPath: "spec.scheduling",
			wantErrDetail:    "leaf children, exceeding maximum allowed",
		},
		{
			name: "composite item exceeding max allowed composite children is rejected",
			root: newWorkloadItem("root", withPath(schedulingPath), withInput(WorkloadInput{Policy: compositeBasic()}),
				withChildren(
					newWorkloadItem("cpg-1", withChildren(newWorkloadItem("leaf-1", withBasic()))),
					newWorkloadItem("cpg-2", withChildren(newWorkloadItem("leaf-2", withBasic()))),
					newWorkloadItem("cpg-3", withChildren(newWorkloadItem("leaf-3", withBasic()))),
					newWorkloadItem("cpg-4", withChildren(newWorkloadItem("leaf-4", withBasic()))),
					newWorkloadItem("cpg-5", withChildren(newWorkloadItem("leaf-5", withBasic()))),
					newWorkloadItem("cpg-6", withChildren(newWorkloadItem("leaf-6", withBasic()))),
					newWorkloadItem("cpg-7", withChildren(newWorkloadItem("leaf-7", withBasic()))),
					newWorkloadItem("cpg-8", withChildren(newWorkloadItem("leaf-8", withBasic()))),
					newWorkloadItem("cpg-9", withChildren(newWorkloadItem("leaf-9", withBasic()))),
				)),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInternal,
			wantErrFieldPath: "spec.scheduling",
			wantErrDetail:    "composite children, exceeding maximum allowed",
		},
		{
			name: "composite basic parent with basic and gang children is valid",
			root: newWorkloadItem("root", withPath(schedulingPath), withInput(WorkloadInput{Policy: compositeBasic()}),
				withChildren(
					newWorkloadItem("basic-child", withPath(schedulingPath), withBasic()),
					newWorkloadItem("gang-child", withPath(schedulingPath), withGang(2)),
				)),
			allowPolicies: allPolicies,
			allowModes:    allModes,
		},
		{
			name: "composite gang parent with basic leaf child is rejected",
			root: newWorkloadItem("root", withPath(schedulingPath), withCompositeGang(1),
				withChildren(newWorkloadItem("child", withPath(field.NewPath("spec", "childScheduling")), withBasic()))),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.childScheduling.schedulingPolicy",
		},
		{
			name: "composite gang parent with unset leaf child policy (defaulting to basic) is rejected",
			root: newWorkloadItem("root", withPath(schedulingPath), withCompositeGang(1),
				withChildren(newWorkloadItem("child", withPath(field.NewPath("spec", "childScheduling")),
					withInput(WorkloadInput{Policy: PolicyInput{PathElements: []string{"schedulingPolicy"}}})))),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.childScheduling.schedulingPolicy",
		},
		{
			name: "composite gang parent with basic composite child is rejected",
			root: newWorkloadItem("root", withPath(schedulingPath), withCompositeGang(1),
				withChildren(newWorkloadItem("mid", withPath(field.NewPath("spec", "midScheduling")),
					withInput(WorkloadInput{Policy: PolicyInput{PathElements: []string{"schedulingPolicy"}, CompositePodGroupData: &schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.WorkloadCompositePodGroupBasicSchedulingPolicy{}}}}),
					withChildren(newWorkloadItem("leaf", withPath(field.NewPath("spec", "leafScheduling")), withGang(2)))))),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.midScheduling.schedulingPolicy",
		},
		{
			name: "composite gang parent with basic composite child and basic leaf grandchild rejects both",
			root: newWorkloadItem("root", withPath(schedulingPath), withCompositeGang(1),
				withChildren(newWorkloadItem("mid", withPath(field.NewPath("spec", "midScheduling")),
					withInput(WorkloadInput{Policy: PolicyInput{PathElements: []string{"schedulingPolicy"}, CompositePodGroupData: &schedulingv1alpha3.WorkloadCompositePodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.WorkloadCompositePodGroupBasicSchedulingPolicy{}}}}),
					withChildren(newWorkloadItem("leaf", withPath(field.NewPath("spec", "leafScheduling")), withBasic()))))),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         2,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.midScheduling.schedulingPolicy",
		},
		{
			name: "composite gang parent with gang composite child and basic leaf grandchild is rejected at grandchild",
			root: newWorkloadItem("root", withPath(schedulingPath), withCompositeGang(1),
				withChildren(newWorkloadItem("mid", withPath(schedulingPath), withCompositeGang(1),
					withChildren(newWorkloadItem("leaf", withPath(field.NewPath("spec", "leafScheduling")), withBasic()))))),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.leafScheduling.schedulingPolicy",
		},
		{
			name: "composite single disruption parent with single and all disruption children is valid",
			root: newWorkloadItem("root", withPath(schedulingPath),
				withInput(WorkloadInput{Policy: compositeGang(), DisruptionMode: compositeSingleDisruption()}),
				withChildren(
					newWorkloadItem("single-child", withPath(schedulingPath), withGang(2), withLeafDisruption(singleDisruption())),
					newWorkloadItem("all-child", withPath(schedulingPath), withGang(2), withLeafDisruption(allDisruption())),
				)),
			allowPolicies: allPolicies,
			allowModes:    allModes,
		},
		{
			name: "composite all disruption parent with all disruption child is valid",
			root: newWorkloadItem("root", withPath(schedulingPath),
				withInput(WorkloadInput{Policy: compositeGang(), DisruptionMode: compositeAllDisruption()}),
				withChildren(newWorkloadItem("child", withPath(schedulingPath), withGang(2), withLeafDisruption(allDisruption())))),
			allowPolicies: allPolicies,
			allowModes:    allModes,
		},
		{
			name: "composite all disruption parent with single disruption leaf child is rejected",
			root: newWorkloadItem("root", withPath(schedulingPath),
				withInput(WorkloadInput{Policy: compositeGang(), DisruptionMode: compositeAllDisruption()}),
				withChildren(newWorkloadItem("child", withPath(field.NewPath("spec", "childScheduling")),
					withGang(2), withLeafDisruption(singleDisruption())))),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.childScheduling.disruptionMode",
		},
		{
			name: "composite all disruption parent with unset disruption leaf child is rejected",
			root: newWorkloadItem("root", withPath(schedulingPath),
				withInput(WorkloadInput{Policy: compositeGang(), DisruptionMode: compositeAllDisruption()}),
				withChildren(newWorkloadItem("child", withPath(field.NewPath("spec", "childScheduling")),
					withGang(2), withLeafDisruption(DisruptionModeInput{PathElements: []string{"disruptionMode"}})))),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.childScheduling.disruptionMode",
		},
		{
			name: "composite all disruption parent with single disruption composite child is rejected",
			root: newWorkloadItem("root", withPath(schedulingPath),
				withInput(WorkloadInput{Policy: compositeGang(), DisruptionMode: compositeAllDisruption()}),
				withChildren(newWorkloadItem("mid", withPath(field.NewPath("spec", "midScheduling")),
					withInput(WorkloadInput{
						Policy: compositeGang(),
						DisruptionMode: DisruptionModeInput{
							PathElements:          []string{"disruptionMode"},
							CompositePodGroupData: &schedulingv1alpha3.WorkloadCompositePodGroupDisruptionMode{Single: &schedulingv1alpha3.WorkloadCompositePodGroupSingleDisruptionMode{}},
						},
					}),
					withChildren(newWorkloadItem("leaf", withPath(schedulingPath), withGang(2), withLeafDisruption(allDisruption())))))),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.midScheduling.disruptionMode",
		},
		{
			name: "matching priorityClassName across the tree is valid",
			root: newWorkloadItem("root", withPath(schedulingPath), withCompositeGang(1), withPriorityClassName("high-priority"),
				withChildren(
					newWorkloadItem("child-1", withPath(schedulingPath), withGang(2), withPriorityClassName("high-priority")),
					newWorkloadItem("child-2", withPath(schedulingPath), withGang(2), withPriorityClassName("high-priority")),
				)),
			allowPolicies: allPolicies,
			allowModes:    allModes,
		},
		{
			name: "mismatched priorityClassName across the tree reports a single error at the root path",
			root: newWorkloadItem("root", withPath(schedulingPath), withCompositeGang(1), withPriorityClassName("high-priority"),
				withChildren(
					newWorkloadItem("child-1", withPath(field.NewPath("spec", "child1")), withGang(2), withPriorityClassName("low-priority")),
					newWorkloadItem("child-2", withPath(field.NewPath("spec", "child2")), withGang(2), withPriorityClassName("other-priority")),
				)),
			allowPolicies:    allPolicies,
			allowModes:       allModes,
			wantErrs:         1,
			wantType:         field.ErrorTypeInvalid,
			wantErrFieldPath: "spec.scheduling",
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
			if tt.wantErrDetail != "" && !strings.Contains(errs[0].Detail, tt.wantErrDetail) {
				t.Errorf("expected error detail containing %q, got %q", tt.wantErrDetail, errs[0].Detail)
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

func withBasic() itemOption {
	return func(i *WorkloadItem) {
		i.Input.Policy = PolicyInput{
			PathElements: []string{"schedulingPolicy"},
			PodGroupData: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
				Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{},
			},
		}
	}
}

func withCompositeGang(minGroupCount int32) itemOption {
	return func(i *WorkloadItem) {
		i.Input.Policy = PolicyInput{
			PathElements:          []string{"schedulingPolicy"},
			CompositePodGroupData: compositeGangData(minGroupCount),
		}
	}
}

func withLeafDisruption(dm DisruptionModeInput) itemOption {
	return func(i *WorkloadItem) {
		dm.PathElements = []string{"disruptionMode"}
		i.Input.DisruptionMode = dm
	}
}

func withPriorityClassName(name string) itemOption {
	return func(i *WorkloadItem) {
		if i.DefaultConfig == nil {
			i.DefaultConfig = &SchedulingConfig{}
		}
		i.DefaultConfig.PriorityClassName = name
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

func compositeSingleDisruption() DisruptionModeInput {
	return DisruptionModeInput{
		CompositePodGroupData: &schedulingv1alpha3.WorkloadCompositePodGroupDisruptionMode{
			Single: &schedulingv1alpha3.WorkloadCompositePodGroupSingleDisruptionMode{},
		},
	}
}

func compositeAllDisruption() DisruptionModeInput {
	return DisruptionModeInput{
		CompositePodGroupData: &schedulingv1alpha3.WorkloadCompositePodGroupDisruptionMode{
			All: &schedulingv1alpha3.WorkloadCompositePodGroupAllDisruptionMode{},
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
