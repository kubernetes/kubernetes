/*
Copyright 2025 The Kubernetes Authors.

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

package workload

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/test/coverage"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	registry "k8s.io/kubernetes/pkg/registry/scheduling/workload"
	"k8s.io/kubernetes/test/declarative_validation/meta"

	// Ensure all API groups are registered with the scheme
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
)

func TestDeclarativeValidate(t *testing.T) {

	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "workloads",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input                           scheduling.Workload
		enableTopologyAwareScheduling   bool
		enableDRAWorkloadResourceClaims bool
		enablePodGroupPreemptionPolicy  bool
		enableCompositePodGroup         bool
		expectedErrs                    field.ErrorList
	}{
		"valid": {
			input: mkValidWorkload(),
		},
		"empty podGroupTemplates": {
			input: mkValidWorkload(clearPodGroupTemplates()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "", "must specify one of: `podGroupTemplates`").WithOrigin("union"),
			},
		},
		"too many podGroupTemplates": {
			input:        mkValidWorkload(setManyPodGroupTemplates(scheduling.WorkloadMaxPodGroupTemplates + 1)),
			expectedErrs: field.ErrorList{field.TooMany(field.NewPath("spec", "podGroupTemplates"), scheduling.WorkloadMaxPodGroupTemplates+1, scheduling.WorkloadMaxPodGroupTemplates).WithOrigin("maxItems")},
		},
		"empty podGroupTemplate name": {
			input:        mkValidWorkload(setPodGroupName(0, "")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplates").Index(0).Child("name"), "")},
		},
		"invalid podGroupTemplate name": {
			input:        mkValidWorkload(setPodGroupName(0, "Invalid_Name")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name")},
		},
		"duplicate podGroupTemplate names": {
			input:        mkValidWorkload(addPodGroupTemplate("main")),
			expectedErrs: field.ErrorList{field.Duplicate(field.NewPath("spec", "podGroupTemplates").Index(1), nil)},
		},
		// Declarative validation treats 0 as "missing" and returns Required error
		// instead of checking minimum constraint and returning Invalid error.
		"gang minCount zero": {
			input:        mkValidWorkload(setPodGroupMinCount(0, 0)),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy", "gang", "minCount"), "")},
		},
		"gang minCount negative": {
			input:        mkValidWorkload(setPodGroupMinCount(0, -1)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy", "gang", "minCount"), nil, "").WithOrigin("minimum")},
		},
		"valid with controllerRef": {
			input: mkValidWorkload(setControllerRef("apps", "Deployment", "my-deployment")),
		},
		"controllerRef with empty APIGroup": {
			input: mkValidWorkload(setControllerRef("", "Pod", "my-pod")),
		},
		"controllerRef invalid APIGroup": {
			input:        mkValidWorkload(setControllerRef("invalid_api_group", "Deployment", "my-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "apiGroup"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"controllerRef too long APIGroup": {
			input:        mkValidWorkload(setControllerRef(strings.Repeat("g", 254), "Deployment", "my-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "apiGroup"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"controllerRef missing kind": {
			input:        mkValidWorkload(setControllerRef("apps", "", "my-deployment")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "controllerRef", "kind"), "")},
		},
		"controllerRef invalid kind with slash": {
			input:        mkValidWorkload(setControllerRef("apps", "Deploy/ment", "my-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "kind"), nil, "").WithOrigin("format=k8s-path-segment-name")},
		},
		"controllerRef missing name": {
			input:        mkValidWorkload(setControllerRef("apps", "Deployment", "")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "controllerRef", "name"), "")},
		},
		"controllerRef invalid name": {
			input:        mkValidWorkload(setControllerRef("apps", "Deployment", "/invalid-name")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "name"), nil, "").WithOrigin("format=k8s-path-segment-name")},
		},
		"controllerRef invalid kind with percent": {
			input:        mkValidWorkload(setControllerRef("apps", "Deploy%ment", "my-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "kind"), nil, "").WithOrigin("format=k8s-path-segment-name")},
		},
		"controllerRef invalid name with percent": {
			input:        mkValidWorkload(setControllerRef("apps", "Deployment", "my%deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "name"), nil, "").WithOrigin("format=k8s-path-segment-name")},
		},
		"policy with neither basic nor gang": {
			input:        mkValidWorkload(clearPodGroupPolicy(0)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy"), nil, "").WithOrigin("union")},
		},
		"policy with both basic and gang": {
			input:        mkValidWorkload(setBothPolicies(0)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy"), nil, "").WithOrigin("union")},
		},
		"valid with basic policy": {
			input: mkValidWorkload(setBasicPolicy(0)),
		},
		"valid with schedulingConstraints": {
			input:                         mkValidWorkload(addTopologyConstraint(0, "foo")),
			enableTopologyAwareScheduling: true,
		},
		"valid with empty schedulingConstraints": {
			input:                         mkValidWorkload(setSchedulingConstraints(0)),
			enableTopologyAwareScheduling: true,
		},
		"with multiple topology constraints": {
			input:                         mkValidWorkload(addTopologyConstraint(0, "foo"), addTopologyConstraint(0, "bar")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.TooMany(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints", "topology"), 2, 1).WithOrigin("maxItems")},
		},
		"with empty topology key": {
			input:                         mkValidWorkload(addTopologyConstraint(0, "")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints", "topology").Index(0).Child("key"), "")},
		},
		"valid with topology key with DNS prefix": {
			input:                         mkValidWorkload(addTopologyConstraint(0, "example.com/Foo")),
			enableTopologyAwareScheduling: true,
		},
		"valid with topology key with prefix with max length": {
			input:                         mkValidWorkload(addTopologyConstraint(0, strings.Repeat("a", 253)+"/"+strings.Repeat("b", 63))),
			enableTopologyAwareScheduling: true,
		},
		"with topology key with prefix exceending max prefix length": {
			input:                         mkValidWorkload(addTopologyConstraint(0, strings.Repeat("a", 254)+"/foo")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key")},
		},
		"with topology key with prefix exceending max name length": {
			input:                         mkValidWorkload(addTopologyConstraint(0, "foo/"+strings.Repeat("b", 64))),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key")},
		},
		"with topology key without prefix exceeding max length": {
			input:                         mkValidWorkload(addTopologyConstraint(0, strings.Repeat("b", 64))),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key")},
		},
		"with topology key with invalid characters": {
			input:                         mkValidWorkload(addTopologyConstraint(0, "Example.com/Foo")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key")},
		},
		"disruption mode single": {
			input: mkValidWorkload(setDisruptionModeSingle(0)),
		},
		"disruption mode all": {
			input: mkValidWorkload(setDisruptionModeAll(0)),
		},
		"disruption mode with neither single nor all": {
			input:        mkValidWorkload(setDisruptionModeNeither(0)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("disruptionMode"), nil, "").WithOrigin("union")},
		},
		"disruption mode with both single and all": {
			input:        mkValidWorkload(setDisruptionModeBoth(0)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("disruptionMode"), nil, "").WithOrigin("union")},
		},
		"valid priorityClassName": {
			input: mkValidWorkload(setPriorityClassName(0, "high-priority")),
		},
		"invalid priorityClassName": {
			input: mkValidWorkload(setPriorityClassName(0, "high/priority")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("priorityClassName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"valid priority": {
			input: mkValidWorkload(setPriority(0, 1000)),
		},
		"valid negative priority": {
			input: mkValidWorkload(setPriority(0, -2147483648)),
		},
		"too high priority": {
			input: mkValidWorkload(setPriority(0, scheduling.HighestUserDefinablePriority+1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("priority"), nil, "").WithOrigin("maximum"),
			},
		},
		"valid PreemptionPolicy": {
			input:                          mkValidWorkload(setPreemptionPolicy(0, scheduling.PreemptLowerPriority)),
			enablePodGroupPreemptionPolicy: true,
		},
		"invalid PreemptionPolicy": {
			input:                          mkValidWorkload(setPreemptionPolicy(0, scheduling.PreemptionPolicy("Invalid"))),
			enablePodGroupPreemptionPolicy: true,
			expectedErrs: field.ErrorList{
				field.NotSupported(field.NewPath("spec", "podGroupTemplates").Index(0).Child("preemptionPolicy"), scheduling.PreemptionPolicy("Invalid"), []string{"Never", "PreemptLowerPriority"}),
			},
		},
		"forbidden PreemptionPolicy when PodGroupPreemptionPolicy is disabled": {
			input: mkValidWorkload(setPreemptionPolicy(0, scheduling.PreemptLowerPriority)),
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0).Child("preemptionPolicy"), ""),
			},
		},

		"ok resourceClaimName reference": {
			input: mkValidWorkload(addResourceClaims(scheduling.PodGroupResourceClaim{Name: "claim", ResourceClaimName: new("resource-claim")})),
		},
		"ok resourceClaimTemplateName reference": {
			input: mkValidWorkload(addResourceClaims(scheduling.PodGroupResourceClaim{Name: "claim", ResourceClaimTemplateName: new("resource-claim-template")})),
		},
		"ok multiple claims": {
			input: mkValidWorkload(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "claim-1", ResourceClaimName: new("resource-claim-1")},
				scheduling.PodGroupResourceClaim{Name: "claim-2", ResourceClaimName: new("resource-claim-2")},
			)),
		},
		"claim name with prefix": {
			input: mkValidWorkload(addResourceClaims(scheduling.PodGroupResourceClaim{Name: "../my-claim", ResourceClaimName: new("resource-claim")})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name"),
			},
		},
		"claim name with path": {
			input: mkValidWorkload(addResourceClaims(scheduling.PodGroupResourceClaim{Name: "my/claim", ResourceClaimName: new("resource-claim")})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name"),
			},
		},
		"duplicate claim entries": {
			input: mkValidWorkload(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimName: new("resource-claim-1")},
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimName: new("resource-claim-2")},
			)),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(1), nil),
			},
		},
		"resource claim source empty": {
			input: mkValidWorkload(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim"},
			)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(0), nil, "").WithOrigin("union"),
			},
		},
		"resource claim reference and template": {
			input: mkValidWorkload(addResourceClaims(
				scheduling.PodGroupResourceClaim{
					Name:                      "my-claim",
					ResourceClaimName:         new("resource-claim"),
					ResourceClaimTemplateName: new("resource-claim-template"),
				},
			)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(0), nil, "").WithOrigin("union"),
			},
		},
		"invalid claim reference name": {
			input: mkValidWorkload(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimName: new(".foo_bar")},
			)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(0).Child("resourceClaimName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"invalid claim template name": {
			input: mkValidWorkload(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new(".foo_bar")},
			)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(0).Child("resourceClaimTemplateName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"too many claims": {
			input: mkValidWorkload(func(pg *scheduling.Workload) {
				for i := range scheduling.MaxPodGroupResourceClaims + 1 {
					pg.Spec.PodGroupTemplates[0].ResourceClaims = append(pg.Spec.PodGroupTemplates[0].ResourceClaims, scheduling.PodGroupResourceClaim{
						Name:              "my-claim-" + strconv.Itoa(i),
						ResourceClaimName: new("resource-claim"),
					})
				}
			}),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims"), scheduling.MaxPodGroupResourceClaims+1, scheduling.MaxPodGroupResourceClaims).WithOrigin("maxItems"),
			},
		},
		"empty claim name": {
			input: mkValidWorkload(addResourceClaims(scheduling.PodGroupResourceClaim{Name: "", ResourceClaimName: new("resource-claim")})),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(0).Child("name"), ""),
			},
		},
		"schedulingConstraints set with TAS disabled": {
			input:        mkValidWorkload(setSchedulingConstraints(0)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints"), "")},
		},
		"empty compositePodGroupTemplates": {
			input:                         mkValidWorkload(setEmptyCompositePodGroupTemplates()),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "", "must specify one of: `podGroupTemplates`, `compositePodGroupTemplates`").WithOrigin("union"),
			},
		},
		"both podGroupTemplates and compositePodGroupTemplates": {
			input:                         mkValidWorkload(setBothTemplates()),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "{podGroupTemplates, compositePodGroupTemplates}", "must specify exactly one of: `podGroupTemplates`, `compositePodGroupTemplates`").WithOrigin("union"),
			},
		},
		"forbidden compositePodGroupTemplates": {
			input:        mkValidWorkload(addCompositePodGroupTemplate("cpg-1")),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates"), "")},
		},
		"too many compositePodGroupTemplates": {
			input:                         mkValidWorkload(setManyCompositePodGroupTemplates(scheduling.WorkloadMaxPodGroupTemplates + 1)),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs:                  field.ErrorList{field.TooMany(field.NewPath("spec", "compositePodGroupTemplates"), scheduling.WorkloadMaxPodGroupTemplates+1, scheduling.WorkloadMaxPodGroupTemplates).WithOrigin("maxItems")},
		},
		"duplicate compositePodGroupTemplates": {
			input:                         mkValidWorkload(addCompositePodGroupTemplate("main")),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Duplicate(field.NewPath("spec", "compositePodGroupTemplates").Index(1), nil)},
		},
		"invalid compositePodGroupTemplate name": {
			input:                         mkValidWorkload(setCompositePodGroupName(0, "Invalid_Name")),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name")},
		},
		"empty compositePodGroupTemplate name": {
			input:                         mkValidWorkload(setCompositePodGroupName(0, "")),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs:                  field.ErrorList{field.Required(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("name"), "")},
		},
		"too many nested compositePodGroupTemplates": {
			input:                         mkValidWorkload(setNestedManyCompositePodGroupTemplates(0, scheduling.WorkloadMaxPodGroupTemplates+1)),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.TooMany(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("compositePodGroupTemplates"), scheduling.WorkloadMaxPodGroupTemplates+1, scheduling.WorkloadMaxPodGroupTemplates).WithOrigin("maxItems")},
		},
		"duplicate nested compositePodGroupTemplates": {
			input:                         mkValidWorkload(addNestedCompositePodGroupTemplate(0, "sub", "sub")),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs:                  field.ErrorList{field.Duplicate(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("compositePodGroupTemplates").Index(1), nil)},
		},
		"cpg empty children": {
			input: mkValidWorkload(func(obj *scheduling.Workload) {
				addCompositePodGroupTemplate()(obj)
				obj.Spec.CompositePodGroupTemplates[0].PodGroupTemplates = nil
			}),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0), "cpg-0", "must have at least one child PodGroupTemplate or CompositePodGroupTemplate").MarkFromImperative(),
			},
		},
		"cpg empty priorityClassName": {
			input:                         mkValidWorkload(setCPGPriorityClassName(0, "my-priority")),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec"), nil, "detected multiple priority configurations").MarkFromImperative()},
		},
		"cpg invalid priorityClassName": {
			input:                         mkValidWorkload(setCPGPriorityClassName(0, "Invalid_Class"), setNestedPGPriorityClassName(0, 0, "Invalid_Class")),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("priorityClassName"), nil, "").WithOrigin("format=k8s-long-name"),
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("priorityClassName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"cpg forbidden priority": {
			input: mkValidWorkload(setCPGPriority(0, 100)),
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates"), ""),
				field.Invalid(field.NewPath("spec"), nil, "detected multiple priority configurations").MarkFromImperative(),
			},
		},
		"cpg policy missing gang": {
			input:                         mkValidWorkload(setCPGPolicyEmpty(0)),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("schedulingPolicy"), nil, "").WithOrigin("union")},
		},
		"cpg policy missing gang minGroupCount": {
			input:                         mkValidWorkload(setCPGMinGroupCount(0, 0)),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Required(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("schedulingPolicy", "gang", "minGroupCount"), "")},
		},
		"cpg policy invalid gang minGroupCount": {
			input:                         mkValidWorkload(setCPGMinGroupCount(0, -1)),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("schedulingPolicy", "gang", "minGroupCount"), nil, "").WithOrigin("minimum")},
		},
		"too many nested podGroupTemplates": {
			input:                         mkValidWorkload(setNestedManyPodGroupTemplates(0, scheduling.WorkloadMaxPodGroupTemplates+1)),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.TooMany(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates"), scheduling.WorkloadMaxPodGroupTemplates+1, scheduling.WorkloadMaxPodGroupTemplates).WithOrigin("maxItems")},
		},
		"duplicate nested podGroupTemplates": {
			input:                         mkValidWorkload(addNestedPodGroupTemplate(0, "sub-pg", "sub-pg")),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs:                  field.ErrorList{field.Duplicate(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(2), nil)},
		},
		"nested pg empty name": {
			input:                         mkValidWorkload(setNestedPGName(0, 0, "")),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs:                  field.ErrorList{field.Required(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("name"), "")},
		},
		"nested pg invalid name": {
			input:                         mkValidWorkload(setNestedPGName(0, 0, "Invalid_Name")),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name")},
		},
		"nested pg invalid disruptionMode union": {
			input:                         mkValidWorkload(setNestedPGDisruptionModeBoth(0, 0)),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("disruptionMode"), nil, "").WithOrigin("union")},
		},
		"nested pg forbidden priorityClassName": {
			input: mkValidWorkload(setNestedPGPriorityClassName(0, 0, "my-priority")),
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates"), ""),
				field.Invalid(field.NewPath("spec"), nil, "detected multiple priority configurations").MarkFromImperative(),
			},
		},
		"nested pg invalid priorityClassName": {
			input:                         mkValidWorkload(setCPGPriorityClassName(0, "Invalid_Class"), setNestedPGPriorityClassName(0, 0, "Invalid_Class")),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("priorityClassName"), nil, "").WithOrigin("format=k8s-long-name"),
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("priorityClassName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"nested pg forbidden priority": {
			input: mkValidWorkload(setNestedPGPriority(0, 0, 100)),
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates"), ""),
				field.Invalid(field.NewPath("spec"), nil, "detected multiple priority configurations").MarkFromImperative(),
			},
		},
		"nested pg invalid priority max": {
			input:                         mkValidWorkload(setCPGPriority(0, scheduling.HighestUserDefinablePriority+1), setNestedPGPriority(0, 0, scheduling.HighestUserDefinablePriority+1)),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("priority"), nil, "").WithOrigin("maximum"),
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("priority"), nil, "").WithOrigin("maximum"),
			},
		},
		"nested pg too many resourceClaims": {
			input:                           mkValidWorkload(setNestedPGManyResourceClaims(0, 0, scheduling.MaxPodGroupResourceClaims+1)),
			enableTopologyAwareScheduling:   true,
			enableCompositePodGroup:         true,
			enableDRAWorkloadResourceClaims: true,
			expectedErrs:                    field.ErrorList{field.TooMany(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("resourceClaims"), scheduling.MaxPodGroupResourceClaims+1, scheduling.MaxPodGroupResourceClaims).WithOrigin("maxItems")},
		},
		"nested pg duplicate resourceClaims": {
			input:                           mkValidWorkload(addNestedPGResourceClaim(0, 0, "claim1", "claim1")),
			enableTopologyAwareScheduling:   true,
			enableCompositePodGroup:         true,
			enableDRAWorkloadResourceClaims: true,
			expectedErrs:                    field.ErrorList{field.Duplicate(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("resourceClaims").Index(1), nil)},
		},
		"nested pg resourceClaim invalid union": {
			input:                           mkValidWorkload(setNestedPGResourceClaimBoth(0, 0, "claim1")),
			enableTopologyAwareScheduling:   true,
			enableCompositePodGroup:         true,
			enableDRAWorkloadResourceClaims: true,
			expectedErrs:                    field.ErrorList{field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("resourceClaims").Index(0), nil, "").WithOrigin("union")},
		},
		"nested pg resourceClaim missing name": {
			input:                           mkValidWorkload(addNestedPGResourceClaimEmptyName(0, 0)),
			enableTopologyAwareScheduling:   true,
			enableCompositePodGroup:         true,
			enableDRAWorkloadResourceClaims: true,
			expectedErrs:                    field.ErrorList{field.Required(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("resourceClaims").Index(0).Child("name"), "")},
		},
		"nested pg resourceClaim invalid name": {
			input:                           mkValidWorkload(addNestedPGResourceClaimInvalidName(0, 0, "Invalid_Name")),
			enableTopologyAwareScheduling:   true,
			enableCompositePodGroup:         true,
			enableDRAWorkloadResourceClaims: true,
			expectedErrs:                    field.ErrorList{field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("resourceClaims").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name")},
		},
		"nested pg resourceClaim invalid resourceClaimName": {
			input:                           mkValidWorkload(addNestedPGResourceClaimInvalidRef(0, 0, "Invalid_Ref")),
			enableTopologyAwareScheduling:   true,
			enableCompositePodGroup:         true,
			enableDRAWorkloadResourceClaims: true,
			expectedErrs:                    field.ErrorList{field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("resourceClaims").Index(0).Child("resourceClaimName"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"nested pg resourceClaim invalid resourceClaimTemplateName": {
			input:                           mkValidWorkload(addNestedPGResourceClaimInvalidTplRef(0, 0, "Invalid_Tpl")),
			enableTopologyAwareScheduling:   true,
			enableCompositePodGroup:         true,
			enableDRAWorkloadResourceClaims: true,
			expectedErrs:                    field.ErrorList{field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("resourceClaims").Index(0).Child("resourceClaimTemplateName"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"nested pg forbidden schedulingConstraints": {
			input:        mkValidWorkload(setNestedPGSchedulingConstraints(0, 0)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates"), "")},
		},
		"nested pg schedulingConstraints too many topology": {
			input:                         mkValidWorkload(setNestedPGSchedulingConstraintsMany(0, 0, 2)),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs:                  field.ErrorList{field.TooMany(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("schedulingConstraints", "topology"), 2, 1).WithOrigin("maxItems")},
		},
		"nested pg schedulingConstraints topology missing key": {
			input:                         mkValidWorkload(setNestedPGSchedulingConstraintsEmptyKey(0, 0)),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs:                  field.ErrorList{field.Required(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("schedulingConstraints", "topology").Index(0).Child("key"), "")},
		},
		"nested pg schedulingConstraints topology invalid key": {
			input:                         mkValidWorkload(setNestedPGSchedulingConstraintsInvalidKey(0, 0, "invalid/key/slash/format")),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("schedulingConstraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key")},
		},
		"nested pg policy invalid union": {
			input:                         mkValidWorkload(setNestedPGPolicyBoth(0, 0)),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("schedulingPolicy"), nil, "").WithOrigin("union")},
		},
		"nested pg policy missing gang minCount": {
			input:                         mkValidWorkload(setNestedPGMinCount(0, 0, 0)),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs:                  field.ErrorList{field.Required(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("schedulingPolicy", "gang", "minCount"), "")},
		},
		"nested pg policy invalid gang minCount": {
			input:                         mkValidWorkload(setNestedPGMinCount(0, 0, -1)),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("schedulingPolicy", "gang", "minCount"), nil, "").WithOrigin("minimum")},
		},
		"cpg tree depth exceeds limit": {
			input:                         mkValidWorkload(setCompositePodGroupTreeDepth(4)),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0), nil, "maximum tree depth is 4").MarkFromImperative(),
			},
		},
		"mismatched PriorityClassName in root podGroupTemplates": {
			input: mkValidWorkload(
				setPriorityClassName(0, "high-priority"),
				addPodGroupTemplate("second-pg"),
				setPriorityClassName(1, "low-priority"),
			),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), nil, "detected multiple priority configurations").MarkFromImperative(),
			},
		},
		"mismatched Priority in root podGroupTemplates": {
			input: mkValidWorkload(
				setPriority(0, 100),
				addPodGroupTemplate("second-pg"),
				setPriority(1, 200),
			),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), nil, "detected multiple priority configurations").MarkFromImperative(),
			},
		},
		"mismatched PriorityClassName in cpg hierarchy": {
			input: mkValidWorkload(
				clearPodGroupTemplates(),
				addCompositePodGroupTemplate("cpg-root"),
				setCPGPriorityClassName(0, "high-priority"),
				addNestedPodGroupTemplate(0, "child-pg-1", "child-pg-2"),
				setNestedPGPriorityClassName(0, 0, "low-priority"),
			),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), nil, "detected multiple priority configurations").MarkFromImperative(),
			},
		},
		"mismatched Priority in cpg hierarchy": {
			input: mkValidWorkload(
				clearPodGroupTemplates(),
				addCompositePodGroupTemplate("cpg-root"),
				setCPGPriority(0, 100),
				addNestedPodGroupTemplate(0, "child-pg-1", "child-pg-2"),
				setNestedPGPriority(0, 0, 200),
			),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), nil, "detected multiple priority configurations").MarkFromImperative(),
			},
		},
		"valid nested pg PreemptionPolicy": {
			input:                          mkValidWorkload(setNestedPGPreemptionPolicy(0, 0, scheduling.PreemptLowerPriority)),
			enableTopologyAwareScheduling:  true,
			enableCompositePodGroup:        true,
			enablePodGroupPreemptionPolicy: true,
		},
		"invalid nested pg PreemptionPolicy": {
			input:                          mkValidWorkload(setNestedPGPreemptionPolicy(0, 0, scheduling.PreemptionPolicy("Invalid"))),
			enableTopologyAwareScheduling:  true,
			enableCompositePodGroup:        true,
			enablePodGroupPreemptionPolicy: true,
			expectedErrs: field.ErrorList{
				field.NotSupported(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("preemptionPolicy"), scheduling.PreemptionPolicy("Invalid"), []string{"Never", "PreemptLowerPriority"}),
			},
		},
		"forbidden nested pg PreemptionPolicy when PodGroupPreemptionPolicy is disabled": {
			input:                          mkValidWorkload(setNestedPGPreemptionPolicy(0, 0, scheduling.PreemptLowerPriority)),
			enableTopologyAwareScheduling:  true,
			enableCompositePodGroup:        true,
			enablePodGroupPreemptionPolicy: false,
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("preemptionPolicy"), ""),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: tc.enableTopologyAwareScheduling,
				features.DRAWorkloadResourceClaims:       tc.enableDRAWorkloadResourceClaims,
				features.PodGroupPreemptionPolicy:        tc.enablePodGroupPreemptionPolicy,
				features.CompositePodGroup:               tc.enableCompositePodGroup,
			})
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}

	obj := mkValidWorkload()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func TestDeclarativeValidateUpdate(t *testing.T) {

	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		oldObj                          scheduling.Workload
		updateObj                       scheduling.Workload
		enableTopologyAwareScheduling   bool
		enableDRAWorkloadResourceClaims bool
		enablePodGroupPreemptionPolicy  bool
		enableCompositePodGroup         bool
		expectedErrs                    field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1")),
		},
		"valid update with unchanged controllerRef": {
			oldObj:    mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "my-deployment")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "my-deployment")),
		},
		"valid update with minCount changed": {
			oldObj:    mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "my-deployment")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "my-deployment"), setPodGroupMinCount(0, 10)),
		},
		"set controllerRef": {
			oldObj:       mkValidWorkload(setResourceVersion("1")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "different-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef"), nil, "").WithOrigin("immutable")},
		},
		"invalid update controllerRef": {
			oldObj:       mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "my-deployment")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "different-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef"), nil, "").WithOrigin("immutable")},
		},
		"unset controllerRef": {
			oldObj:       mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "different-deployment")),
			updateObj:    mkValidWorkload(setResourceVersion("1")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef"), nil, "").WithOrigin("immutable")},
		},
		"invalid update empty podGroupTemplates": {
			oldObj:                        mkValidWorkload(setResourceVersion("1")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), setEmptyPodGroupTemplates()),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "podGroupTemplates"), "").WithOrigin("update"),
				field.Invalid(field.NewPath("spec"), "", "must specify one of: `podGroupTemplates`, `compositePodGroupTemplates`").WithOrigin("union"),
			},
		},
		"change podGroupTemplate name": {
			oldObj:    mkValidWorkload(setResourceVersion("1"), addPodGroupTemplate("worker1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), addPodGroupTemplate("worker2")),
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(1), "").WithOrigin("update"),
				field.Forbidden(field.NewPath("spec", "podGroupTemplates"), "").WithOrigin("update"),
			},
		},
		"invalid update too many podGroupTemplates": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setManyPodGroupTemplates(scheduling.WorkloadMaxPodGroupTemplates+1)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "podGroupTemplates"), scheduling.WorkloadMaxPodGroupTemplates+1, scheduling.WorkloadMaxPodGroupTemplates).WithOrigin("maxItems"),
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0), "").WithOrigin("update"),
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(1), "").WithOrigin("update"),
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(2), "").WithOrigin("update"),
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(3), "").WithOrigin("update"),
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(4), "").WithOrigin("update"),
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(5), "").WithOrigin("update"),
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(6), "").WithOrigin("update"),
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(7), "").WithOrigin("update"),
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(8), "").WithOrigin("update"),
				field.Forbidden(field.NewPath("spec", "podGroupTemplates"), "").WithOrigin("update"),
			},
		},
		"add podGroupTemplate": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), addPodGroupTemplate("worker1")),
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(1), "").WithOrigin("update"),
			},
		},
		"remove podGroupTemplate": {
			oldObj:    mkValidWorkload(setResourceVersion("1"), addPodGroupTemplate("worker1")),
			updateObj: mkValidWorkload(setResourceVersion("1")),
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "podGroupTemplates"), "").WithOrigin("update"),
			},
		},
		"invalid update with neither basic nor gang": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), clearPodGroupPolicy(0)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy"), nil, "").WithOrigin("union"),
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy", "gang"), nil, "").WithOrigin("update"),
			},
		},
		"invalid update with both basic and gang": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setBothPolicies(0)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy"), nil, "").WithOrigin("union"),
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy", "basic"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update of gang minCount": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setPodGroupMinCount(0, -1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy", "gang", "minCount"), nil, "").WithOrigin("minimum"),
			},
		},
		"invalid update from gang to basic policy": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setBasicPolicy(0)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy", "basic"), nil, "").WithOrigin("immutable"),
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy", "gang"), nil, "").WithOrigin("update"),
			},
		},
		"valid update with unchanged scheduling constraints": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			enableTopologyAwareScheduling: true,
		},
		"invalid update to scheduling constraints": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), setSchedulingConstraints(0)),
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update to topology constraints": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo"), addTopologyConstraint(0, "bar")),
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update to topology key": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "bar")),
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
		"valid update with unchanged scheduling constraints with TAS disabled": {
			oldObj:    mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj: mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
		},
		"invalid update to scheduling constraints with TAS disabled": {
			oldObj:    mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setSchedulingConstraints(0)),
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints"), ""),
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update to topology constraints with TAS disabled": {
			oldObj:    mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj: mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo"), addTopologyConstraint(0, "bar")),
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints"), ""),
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update to topology key with TAS disabled": {
			oldObj:    mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj: mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "bar")),
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints"), ""),
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid add of resource claims, DRA workload resource claims disabled": {
			oldObj: mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid add of resource claims, DRA workload resource claims enabled": {
			oldObj: mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			enableDRAWorkloadResourceClaims: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update of resource claims, DRA workload resource claims disabled": {
			oldObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			updateObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-other-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update of resource claims, DRA workload resource claims enabled": {
			oldObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			updateObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-other-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			enableDRAWorkloadResourceClaims: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid remove of resource claims, DRA workload resource claims disabled": {
			oldObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			updateObj: mkValidWorkload(setResourceVersion("1")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid remove of resource claims, DRA workload resource claims enabled": {
			oldObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			updateObj:                       mkValidWorkload(setResourceVersion("1")),
			enableDRAWorkloadResourceClaims: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update of disruption mode": {
			oldObj:    mkValidWorkload(setResourceVersion("1"), setDisruptionModeSingle(0)),
			updateObj: mkValidWorkload(setResourceVersion("1"), setDisruptionModeAll(0)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("disruptionMode"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update of priority class name": {
			oldObj:       mkValidWorkload(setResourceVersion("1"), setPriorityClassName(0, "low-priority")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), setPriorityClassName(0, "high-priority")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("priorityClassName"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of priority": {
			oldObj:    mkValidWorkload(setResourceVersion("1"), setPriority(0, 1000)),
			updateObj: mkValidWorkload(setResourceVersion("1"), setPriority(0, 2000)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("priority"), nil, "").WithOrigin("immutable"),
			},
		},
		"valid update with unchanged PreemptionPolicy": {
			oldObj:                         mkValidWorkload(setResourceVersion("1"), setPreemptionPolicy(0, scheduling.PreemptLowerPriority)),
			updateObj:                      mkValidWorkload(setResourceVersion("1"), setPreemptionPolicy(0, scheduling.PreemptLowerPriority)),
			enablePodGroupPreemptionPolicy: true,
		},
		"invalid update of PreemptionPolicy": {
			oldObj:                         mkValidWorkload(setResourceVersion("1"), setPreemptionPolicy(0, scheduling.PreemptLowerPriority)),
			updateObj:                      mkValidWorkload(setResourceVersion("1"), setPreemptionPolicy(0, scheduling.PreemptNever)),
			enablePodGroupPreemptionPolicy: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("preemptionPolicy"), nil, "").WithOrigin("immutable"),
			},
		},
		"valid update with unchanged PreemptionPolicy when PodGroupPreemptionPolicy is disabled": {
			oldObj:                         mkValidWorkload(setResourceVersion("1"), setPreemptionPolicy(0, scheduling.PreemptLowerPriority)),
			updateObj:                      mkValidWorkload(setResourceVersion("1"), setPreemptionPolicy(0, scheduling.PreemptLowerPriority)),
			enablePodGroupPreemptionPolicy: false,
		},
		"invalid update of PreemptionPolicy when PodGroupPreemptionPolicy is disabled": {
			oldObj:                         mkValidWorkload(setResourceVersion("1"), setPreemptionPolicy(0, scheduling.PreemptLowerPriority)),
			updateObj:                      mkValidWorkload(setResourceVersion("1"), setPreemptionPolicy(0, scheduling.PreemptNever)),
			enablePodGroupPreemptionPolicy: false,
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0).Child("preemptionPolicy"), ""),
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("preemptionPolicy"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update add compositePodGroupTemplates": {
			oldObj:                        mkValidWorkload(setResourceVersion("1")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate()),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates").Index(0), "").WithOrigin("update"),
				field.Forbidden(field.NewPath("spec", "podGroupTemplates"), "").WithOrigin("update"),
			},
		},
		"invalid update add nested compositePodGroupTemplates": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate()),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate(), addNestedCompositePodGroupTemplate(0, "sub")),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("compositePodGroupTemplates").Index(0), "").WithOrigin("update"),
			},
		},
		"invalid update remove compositePodGroupTemplates": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate()),
			updateObj:                     mkValidWorkload(setResourceVersion("1")),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates"), "").WithOrigin("update"),
				field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0), "").WithOrigin("update"),
			},
		},
		"invalid update remove nested compositePodGroupTemplates": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate(), addNestedCompositePodGroupTemplate(0, "sub")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate()),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("compositePodGroupTemplates"), "").WithOrigin("update"),
			},
		},
		"invalid update remove nested podGroupTemplates": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate(), addNestedPodGroupTemplate(0, "sub-pg")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate()),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates"), "").WithOrigin("update"),
			},
		},

		"invalid update immutable compositePodGroupTemplates": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate()),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate("another")),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates").Index(1), "").WithOrigin("update"),
			},
		},
		"invalid update immutable nested compositePodGroupTemplates": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate()),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addNestedCompositePodGroupTemplate(0, "sub")),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("compositePodGroupTemplates").Index(0), "").WithOrigin("update"),
			},
		},
		"invalid update immutable cpg priorityClassName": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), setCPGPriorityClassName(0, "p1"), setNestedPGPriorityClassName(0, 0, "p1")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), setCPGPriorityClassName(0, "p2"), setNestedPGPriorityClassName(0, 0, "p2")),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("priorityClassName"), nil, "field is immutable").WithOrigin("immutable"),
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("priorityClassName"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update immutable cpg priority": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), setCPGPriority(0, 1), setNestedPGPriority(0, 0, 1)),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), setCPGPriority(0, 2), setNestedPGPriority(0, 0, 2)),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("priority"), nil, "field is immutable").WithOrigin("immutable"),
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("priority"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update immutable nested podGroupTemplates": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate()),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addNestedPodGroupTemplate(0, "another-pg")),
			enableTopologyAwareScheduling: true,
			enableCompositePodGroup:       true,
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(1), "").WithOrigin("update"),
			},
		},
		"invalid update immutable nested pg disruptionMode": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate(), setNestedPGDisruptionModeSingle(0, 0)),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate(), setNestedPGDisruptionModeAll(0, 0)),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("disruptionMode"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update immutable nested pg priorityClassName": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), setCPGPriorityClassName(0, "p1"), setNestedPGPriorityClassName(0, 0, "p1")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), setCPGPriorityClassName(0, "p2"), setNestedPGPriorityClassName(0, 0, "p2")),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("priorityClassName"), nil, "field is immutable").WithOrigin("immutable"),
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("priorityClassName"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update immutable nested pg priority": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), setCPGPriority(0, 1), setNestedPGPriority(0, 0, 1)),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), setCPGPriority(0, 2), setNestedPGPriority(0, 0, 2)),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("priority"), nil, "field is immutable").WithOrigin("immutable"),
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("priority"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update immutable nested pg resourceClaims": {
			oldObj:                          mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate(), addNestedPGResourceClaim(0, 0, "claim1")),
			updateObj:                       mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate(), addNestedPGResourceClaim(0, 0, "claim2")),
			enableCompositePodGroup:         true,
			enableTopologyAwareScheduling:   true,
			enableDRAWorkloadResourceClaims: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("resourceClaims"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update immutable nested pg schedulingConstraints": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate(), setNestedPGSchedulingConstraints(0, 0)),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate(), setNestedPGSchedulingConstraintsMany(0, 0, 1)),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("schedulingConstraints"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
		"invalid update immutable cpg basic policy": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), setCPGBasicPolicy(0)),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), setCPGPolicyEmpty(0)),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("schedulingPolicy", "basic"), nil, "field is immutable").WithOrigin("immutable"),
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("schedulingPolicy"), nil, "").WithOrigin("union"),
			},
		},
		"invalid update immutable cpg gang policy": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), setCPGGangPolicy(0)),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), setCPGPolicyEmpty(0)),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("schedulingPolicy", "gang"), nil, "field cannot be cleared once set").WithOrigin("update"),
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("schedulingPolicy"), nil, "").WithOrigin("union"),
			},
		},
		"invalid update nested pg from gang to basic policy": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate()),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate(), setNestedPGBasicPolicy(0, 0)),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("schedulingPolicy", "basic"), nil, "field is immutable").WithOrigin("immutable"),
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("schedulingPolicy", "gang"), nil, "field cannot be cleared once set").WithOrigin("update"),
			},
		},
		"invalid update nested pg with neither basic nor gang": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate()),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addCompositePodGroupTemplate(), clearNestedPGPolicy(0, 0)),
			enableCompositePodGroup:       true,
			enableTopologyAwareScheduling: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("schedulingPolicy"), nil, "").WithOrigin("union"),
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("schedulingPolicy", "gang"), nil, "field cannot be cleared once set").WithOrigin("update"),
			},
		},
		"valid update with unchanged nested pg PreemptionPolicy": {
			oldObj:                         mkValidWorkload(setResourceVersion("1"), setNestedPGPreemptionPolicy(0, 0, scheduling.PreemptLowerPriority)),
			updateObj:                      mkValidWorkload(setResourceVersion("1"), setNestedPGPreemptionPolicy(0, 0, scheduling.PreemptLowerPriority)),
			enableTopologyAwareScheduling:  true,
			enableCompositePodGroup:        true,
			enablePodGroupPreemptionPolicy: true,
		},
		"invalid update of nested pg PreemptionPolicy": {
			oldObj:                         mkValidWorkload(setResourceVersion("1"), setNestedPGPreemptionPolicy(0, 0, scheduling.PreemptLowerPriority)),
			updateObj:                      mkValidWorkload(setResourceVersion("1"), setNestedPGPreemptionPolicy(0, 0, scheduling.PreemptNever)),
			enableTopologyAwareScheduling:  true,
			enableCompositePodGroup:        true,
			enablePodGroupPreemptionPolicy: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("preemptionPolicy"), nil, "").WithOrigin("immutable"),
			},
		},
		"valid update with unchanged nested pg PreemptionPolicy when PodGroupPreemptionPolicy is disabled": {
			oldObj:                         mkValidWorkload(setResourceVersion("1"), setNestedPGPreemptionPolicy(0, 0, scheduling.PreemptLowerPriority)),
			updateObj:                      mkValidWorkload(setResourceVersion("1"), setNestedPGPreemptionPolicy(0, 0, scheduling.PreemptLowerPriority)),
			enableTopologyAwareScheduling:  true,
			enableCompositePodGroup:        true,
			enablePodGroupPreemptionPolicy: false,
		},
		"invalid update of nested pg PreemptionPolicy when PodGroupPreemptionPolicy is disabled": {
			oldObj:                         mkValidWorkload(setResourceVersion("1"), setNestedPGPreemptionPolicy(0, 0, scheduling.PreemptLowerPriority)),
			updateObj:                      mkValidWorkload(setResourceVersion("1"), setNestedPGPreemptionPolicy(0, 0, scheduling.PreemptNever)),
			enableTopologyAwareScheduling:  true,
			enableCompositePodGroup:        true,
			enablePodGroupPreemptionPolicy: false,
			expectedErrs: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("preemptionPolicy"), ""),
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("preemptionPolicy"), nil, "").WithOrigin("immutable"),
			},
		},
	}
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "workloads",
		Name:              "valid-workload",
		IsResourceRequest: true,
		Verb:              "update",
	})
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: tc.enableTopologyAwareScheduling,
				features.DRAWorkloadResourceClaims:       tc.enableDRAWorkloadResourceClaims,
				features.PodGroupPreemptionPolicy:        tc.enablePodGroupPreemptionPolicy,
				features.CompositePodGroup:               tc.enableCompositePodGroup,
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, registry.Strategy, tc.expectedErrs)
		})
	}

	updateObj := mkValidWorkload(setResourceVersion("1"))
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())

	// The FieldValueForbidden rule on schedulingConstraints is impossible because it is only forbidden when the
	// TopologyAwareWorkloadScheduling feature gate is disabled, but disabling that feature gate disables the parent
	// CompositePodGroup structure entirely (causing a top-level error before children rules are evaluated).
	gvk := schema.GroupVersionKind{Group: "scheduling.k8s.io", Version: apiVersion, Kind: "Workload"}
	coverage.RecordObservedRules(gvk, field.ErrorList{
		field.Forbidden(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("schedulingConstraints"), ""),
	})
}

func mkValidWorkload(tweaks ...func(obj *scheduling.Workload)) scheduling.Workload {
	obj := scheduling.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-workload",
			Namespace: "default",
		},
		Spec: scheduling.WorkloadSpec{
			PodGroupTemplates: []scheduling.PodGroupTemplate{
				{
					Name: "main",
					SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
						Gang: &scheduling.GangSchedulingPolicy{
							MinCount: 1,
						},
					},
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func setResourceVersion(v string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.ResourceVersion = v
	}
}

func clearPodGroupTemplates() func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = nil
	}
}

func setEmptyPodGroupTemplates() func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = []scheduling.PodGroupTemplate{}
	}
}

func setManyPodGroupTemplates(n int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = make([]scheduling.PodGroupTemplate, n)
		for i := range obj.Spec.PodGroupTemplates {
			obj.Spec.PodGroupTemplates[i] = scheduling.PodGroupTemplate{
				Name: fmt.Sprintf("group-%d", i),
				SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
					Gang: &scheduling.GangSchedulingPolicy{
						MinCount: 1,
					},
				},
			}
		}
	}
}

func addPodGroupTemplate(name string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = append(obj.Spec.PodGroupTemplates, scheduling.PodGroupTemplate{
			Name: name,
			SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
				Gang: &scheduling.GangSchedulingPolicy{
					MinCount: 1,
				},
			},
		})
	}
}

func setPodGroupName(pgIdx int, name string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].Name = name
	}
}

func setPodGroupMinCount(pgIdx, min int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingPolicy.Gang.MinCount = int32(min)
	}
}

func clearPodGroupPolicy(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{}
	}
}

func setBasicPolicy(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
			Basic: &scheduling.BasicSchedulingPolicy{},
		}
	}
}

func setBothPolicies(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
			Basic: &scheduling.BasicSchedulingPolicy{},
			Gang:  &scheduling.GangSchedulingPolicy{MinCount: 1},
		}
	}
}

func setControllerRef(apiGroup, kind, name string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
			APIGroup: apiGroup,
			Kind:     kind,
			Name:     name,
		}
	}
}

func setSchedulingConstraints(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{}
	}
}

func addTopologyConstraint(pgIdx int, topologyKey string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		if obj.Spec.PodGroupTemplates[pgIdx].SchedulingConstraints == nil {
			setSchedulingConstraints(pgIdx)(obj)
		}
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingConstraints.Topology = append(obj.Spec.PodGroupTemplates[pgIdx].SchedulingConstraints.Topology,
			scheduling.TopologyConstraint{Key: topologyKey})
	}
}

func addResourceClaims(claims ...scheduling.PodGroupResourceClaim) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[0].ResourceClaims = append(obj.Spec.PodGroupTemplates[0].ResourceClaims, claims...)
	}
}

func setDisruptionModeSingle(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].DisruptionMode = &scheduling.DisruptionMode{
			Single: &scheduling.SingleDisruptionMode{},
		}
	}
}

func setDisruptionModeAll(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].DisruptionMode = &scheduling.DisruptionMode{
			All: &scheduling.AllDisruptionMode{},
		}
	}
}

func setDisruptionModeNeither(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].DisruptionMode = &scheduling.DisruptionMode{}
	}
}

func setDisruptionModeBoth(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].DisruptionMode = &scheduling.DisruptionMode{
			Single: &scheduling.SingleDisruptionMode{},
			All:    &scheduling.AllDisruptionMode{},
		}
	}
}

func setPriorityClassName(pgIdx int, priorityClassName string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].PriorityClassName = priorityClassName
	}
}

func setPriority(pgIdx int, priority int32) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].Priority = new(priority)
	}
}

func setPreemptionPolicy(pgIdx int, policy scheduling.PreemptionPolicy) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].PreemptionPolicy = &policy
	}
}

func setEmptyCompositePodGroupTemplates() func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = nil
		obj.Spec.CompositePodGroupTemplates = []scheduling.CompositePodGroupTemplate{}
	}
}

func addCompositePodGroupTemplate(names ...string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = nil
		if obj.Spec.CompositePodGroupTemplates == nil {
			obj.Spec.CompositePodGroupTemplates = []scheduling.CompositePodGroupTemplate{{
				Name: "main",
				SchedulingPolicy: scheduling.CompositePodGroupSchedulingPolicy{
					Gang: &scheduling.CompositeGangSchedulingPolicy{
						MinGroupCount: 1,
					},
				},
				PodGroupTemplates: []scheduling.PodGroupTemplate{{
					Name: "worker-main",
					SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
						Gang: &scheduling.GangSchedulingPolicy{MinCount: 1},
					},
				}},
			}}
		}
		for _, name := range names {
			obj.Spec.CompositePodGroupTemplates = append(obj.Spec.CompositePodGroupTemplates, scheduling.CompositePodGroupTemplate{
				Name: name,
				SchedulingPolicy: scheduling.CompositePodGroupSchedulingPolicy{
					Gang: &scheduling.CompositeGangSchedulingPolicy{MinGroupCount: 1},
				},
				PodGroupTemplates: []scheduling.PodGroupTemplate{{
					Name: fmt.Sprintf("worker-%s-%d", name, len(obj.Spec.CompositePodGroupTemplates)),
					SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
						Gang: &scheduling.GangSchedulingPolicy{MinCount: 1},
					},
				}},
			})
		}
	}
}

func setManyCompositePodGroupTemplates(n int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = nil
		obj.Spec.CompositePodGroupTemplates = make([]scheduling.CompositePodGroupTemplate, n)
		for i := range obj.Spec.CompositePodGroupTemplates {
			obj.Spec.CompositePodGroupTemplates[i] = scheduling.CompositePodGroupTemplate{
				Name: fmt.Sprintf("cpg-%d", i),
				SchedulingPolicy: scheduling.CompositePodGroupSchedulingPolicy{
					Gang: &scheduling.CompositeGangSchedulingPolicy{MinGroupCount: 1},
				},
				PodGroupTemplates: []scheduling.PodGroupTemplate{{
					Name: fmt.Sprintf("worker-%d", i),
					SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
						Gang: &scheduling.GangSchedulingPolicy{MinCount: 1},
					},
				}},
			}
		}
	}
}

func setCompositePodGroupName(idx int, name string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[idx].Name = name
	}
}

func setNestedManyCompositePodGroupTemplates(idx, n int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[idx].CompositePodGroupTemplates = make([]scheduling.CompositePodGroupTemplate, n)
		for i := range obj.Spec.CompositePodGroupTemplates[idx].CompositePodGroupTemplates {
			obj.Spec.CompositePodGroupTemplates[idx].CompositePodGroupTemplates[i] = scheduling.CompositePodGroupTemplate{
				Name: fmt.Sprintf("sub-%d", i),
				SchedulingPolicy: scheduling.CompositePodGroupSchedulingPolicy{
					Gang: &scheduling.CompositeGangSchedulingPolicy{MinGroupCount: 1},
				},
				PodGroupTemplates: []scheduling.PodGroupTemplate{{
					Name:             fmt.Sprintf("sub-%d-leaf", i),
					SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{Basic: &scheduling.BasicSchedulingPolicy{}},
				}},
			}
		}
	}
}

func addNestedCompositePodGroupTemplate(idx int, names ...string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		for _, name := range names {
			obj.Spec.CompositePodGroupTemplates[idx].CompositePodGroupTemplates = append(
				obj.Spec.CompositePodGroupTemplates[idx].CompositePodGroupTemplates,
				scheduling.CompositePodGroupTemplate{
					Name: name,
					SchedulingPolicy: scheduling.CompositePodGroupSchedulingPolicy{
						Gang: &scheduling.CompositeGangSchedulingPolicy{MinGroupCount: 1},
					},
					PodGroupTemplates: []scheduling.PodGroupTemplate{{
						Name: fmt.Sprintf("%s-leaf-%d", name, len(obj.Spec.CompositePodGroupTemplates[idx].CompositePodGroupTemplates)),
						SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
							Basic: &scheduling.BasicSchedulingPolicy{},
						},
					}},
				},
			)
		}
	}
}

func setCPGPriorityClassName(idx int, priorityClassName string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[idx].PriorityClassName = priorityClassName
	}
}

func setCPGPriority(idx int, priority int32) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[idx].Priority = &priority
	}
}

func setCPGBasicPolicy(idx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[idx].SchedulingPolicy = scheduling.CompositePodGroupSchedulingPolicy{
			Basic: &scheduling.CompositeBasicSchedulingPolicy{},
		}
	}
}

func setCPGGangPolicy(idx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[idx].SchedulingPolicy = scheduling.CompositePodGroupSchedulingPolicy{
			Gang: &scheduling.CompositeGangSchedulingPolicy{MinGroupCount: 2},
		}
	}
}

func setCPGPolicyEmpty(idx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[idx].SchedulingPolicy = scheduling.CompositePodGroupSchedulingPolicy{}
	}
}

func setCPGMinGroupCount(idx int, min int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[idx].SchedulingPolicy.Gang.MinGroupCount = int32(min)
	}
}

func setNestedManyPodGroupTemplates(idx, n int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[idx].PodGroupTemplates = make([]scheduling.PodGroupTemplate, n)
		for i := range obj.Spec.CompositePodGroupTemplates[idx].PodGroupTemplates {
			obj.Spec.CompositePodGroupTemplates[idx].PodGroupTemplates[i] = scheduling.PodGroupTemplate{
				Name: fmt.Sprintf("pg-%d", i),
				SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
					Gang: &scheduling.GangSchedulingPolicy{MinCount: 1},
				},
			}
		}
	}
}

func addNestedPodGroupTemplate(idx int, names ...string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		for _, name := range names {
			obj.Spec.CompositePodGroupTemplates[idx].PodGroupTemplates = append(
				obj.Spec.CompositePodGroupTemplates[idx].PodGroupTemplates,
				scheduling.PodGroupTemplate{
					Name: name,
					SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
						Gang: &scheduling.GangSchedulingPolicy{MinCount: 1},
					},
				},
			)
		}
	}
}

func setNestedPGName(cidx, pidx int, name string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].Name = name
	}
}

func setNestedPGDisruptionModeBoth(cidx, pidx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].DisruptionMode = &scheduling.DisruptionMode{Single: &scheduling.SingleDisruptionMode{}, All: &scheduling.AllDisruptionMode{}}
	}
}

func setNestedPGPriorityClassName(cidx, pidx int, name string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].PriorityClassName = name
	}
}

func setNestedPGPriority(cidx, pidx int, p int32) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].Priority = &p
	}
}

func setNestedPGPreemptionPolicy(cidx, pidx int, policy scheduling.PreemptionPolicy) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].PreemptionPolicy = &policy
	}
}

func setNestedPGManyResourceClaims(cidx, pidx, n int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].ResourceClaims = make([]scheduling.PodGroupResourceClaim, n)
		for i := range obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].ResourceClaims {
			obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].ResourceClaims[i] = scheduling.PodGroupResourceClaim{
				Name:              fmt.Sprintf("claim-%d", i),
				ResourceClaimName: new("valid-ref"),
			}
		}
	}
}

func addNestedPGResourceClaim(cidx, pidx int, names ...string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		for _, name := range names {
			obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].ResourceClaims = append(
				obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].ResourceClaims,
				scheduling.PodGroupResourceClaim{Name: name, ResourceClaimName: new("valid-ref")},
			)
		}
	}
}

func setNestedPGResourceClaimBoth(cidx, pidx int, name string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].ResourceClaims = []scheduling.PodGroupResourceClaim{
			{Name: name, ResourceClaimName: new("valid-ref"), ResourceClaimTemplateName: new("valid-ref")},
		}
	}
}

func addNestedPGResourceClaimEmptyName(cidx, pidx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].ResourceClaims = []scheduling.PodGroupResourceClaim{
			{ResourceClaimName: new("valid-ref")},
		}
	}
}

func addNestedPGResourceClaimInvalidName(cidx, pidx int, name string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].ResourceClaims = []scheduling.PodGroupResourceClaim{
			{Name: name, ResourceClaimName: new("valid-ref")},
		}
	}
}

func addNestedPGResourceClaimInvalidRef(cidx, pidx int, ref string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].ResourceClaims = []scheduling.PodGroupResourceClaim{
			{Name: "valid-name", ResourceClaimName: &ref},
		}
	}
}

func addNestedPGResourceClaimInvalidTplRef(cidx, pidx int, ref string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].ResourceClaims = []scheduling.PodGroupResourceClaim{
			{Name: "valid-name", ResourceClaimTemplateName: &ref},
		}
	}
}

func setNestedPGSchedulingConstraints(cidx, pidx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{}
	}
}

func setNestedPGSchedulingConstraintsMany(cidx, pidx, n int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
			Topology: make([]scheduling.TopologyConstraint, n),
		}
		for i := range obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].SchedulingConstraints.Topology {
			obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].SchedulingConstraints.Topology[i] = scheduling.TopologyConstraint{Key: fmt.Sprintf("key-%d", i)}
		}
	}
}

func setNestedPGSchedulingConstraintsEmptyKey(cidx, pidx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
			Topology: []scheduling.TopologyConstraint{{}},
		}
	}
}

func setNestedPGSchedulingConstraintsInvalidKey(cidx, pidx int, key string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{
			Topology: []scheduling.TopologyConstraint{{Key: key}},
		}
	}
}

func setNestedPGPolicyBoth(cidx, pidx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
			Basic: &scheduling.BasicSchedulingPolicy{},
			Gang:  &scheduling.GangSchedulingPolicy{MinCount: 1},
		}
	}
}

func setNestedPGMinCount(cidx, pidx int, count int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].SchedulingPolicy.Gang.MinCount = int32(count)
	}
}

func setNestedPGDisruptionModeSingle(cidx, pidx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].DisruptionMode = &scheduling.DisruptionMode{Single: &scheduling.SingleDisruptionMode{}}
	}
}

func setNestedPGDisruptionModeAll(cidx, pidx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].DisruptionMode = &scheduling.DisruptionMode{All: &scheduling.AllDisruptionMode{}}
	}
}

func clearNestedPGPolicy(cidx, pidx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{}
	}
}

func setNestedPGBasicPolicy(cidx, pidx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		addCompositePodGroupTemplate()(obj)
		obj.Spec.CompositePodGroupTemplates[cidx].PodGroupTemplates[pidx].SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
			Basic: &scheduling.BasicSchedulingPolicy{},
		}
	}
}

func setBothTemplates() func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = []scheduling.PodGroupTemplate{{
			Name: "main",
			SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
				Basic: &scheduling.BasicSchedulingPolicy{},
			},
		}}
		obj.Spec.CompositePodGroupTemplates = []scheduling.CompositePodGroupTemplate{{
			Name: "cpg",
			SchedulingPolicy: scheduling.CompositePodGroupSchedulingPolicy{
				Basic: &scheduling.CompositeBasicSchedulingPolicy{},
			},
			PodGroupTemplates: []scheduling.PodGroupTemplate{{
				Name: "cpg-pg",
				SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
					Basic: &scheduling.BasicSchedulingPolicy{},
				},
			}},
		}}
	}
}

func setCompositePodGroupTreeDepth(depth int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = nil
		var buildTree func(level int) []scheduling.CompositePodGroupTemplate
		buildTree = func(level int) []scheduling.CompositePodGroupTemplate {
			if level > depth {
				return nil
			}
			cpgTpl := scheduling.CompositePodGroupTemplate{
				Name: fmt.Sprintf("cpg-%d", level),
				SchedulingPolicy: scheduling.CompositePodGroupSchedulingPolicy{
					Basic: &scheduling.CompositeBasicSchedulingPolicy{},
				},
			}
			if level < depth {
				cpgTpl.CompositePodGroupTemplates = buildTree(level + 1)
			} else {
				cpgTpl.PodGroupTemplates = []scheduling.PodGroupTemplate{{
					Name: "pg-leaf",
					SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
						Basic: &scheduling.BasicSchedulingPolicy{},
					},
				}}
			}
			return []scheduling.CompositePodGroupTemplate{cpgTpl}
		}
		obj.Spec.CompositePodGroupTemplates = buildTree(1)
	}
}
